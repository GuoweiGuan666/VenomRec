# src/data.py

from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from pathlib import Path
from collections import Counter, defaultdict
import glob
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
import yaml
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer
try:
    from .tokenization import P5Tokenizer
except ImportError:  # fallback when executed as a top-level script
    from tokenization import P5Tokenizer  # type: ignore
from typing import Dict, Any, List, Optional
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_PAIR_ROOT = Path(
    os.environ.get("VIP5_TEXT_PAIR_ROOT", str(PROJECT_ROOT / "poison_text_pairs"))
)
DEFAULT_HIST_BASE_DIR = Path(
    os.environ.get("VIP5_HIST_BASE_DIR", str(PROJECT_ROOT / "metrics" / "hist_baseline"))
)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)   


def _counter_to_serializable(counter: Dict[Any, float]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in counter.items()}


def _normalise(counter: Dict[Any, float]) -> Dict[str, float]:
    total = sum(float(v) for v in counter.values())
    if total <= 0:
        return {str(k): 0.0 for k in counter.keys()}
    return {str(k): float(v) / total for k, v in counter.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """Compute KL(p || q) for two probability dictionaries."""

    keys = set(p.keys()) | set(q.keys())
    value = 0.0
    for key in keys:
        pv = float(p.get(key, 0.0))
        qv = float(q.get(key, 0.0))
        if pv <= 0.0:
            continue
        value += pv * math.log(pv / max(qv, eps))
    return float(value)


def _chi_square(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """Compute χ² distance between two count dictionaries."""

    keys = set(p.keys()) | set(q.keys())
    value = 0.0
    for key in keys:
        pv = float(p.get(key, 0.0))
        qv = float(q.get(key, 0.0))
        denom = max(qv, eps)
        value += ((pv - qv) ** 2) / denom
    return float(value)


def _histogram_path(split: str, mode: str, tag: str | None) -> Path:
    label = tag or f"{split}_{mode}"
    override = os.environ.get("VIP5_HIST_BASELINE_PATH")
    if override:
        return Path(override)
    return DEFAULT_HIST_BASE_DIR / f"{label}.json"


def _write_histogram(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_histogram(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _compare_histograms(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    cur_len_counts = {str(k): float(v) for k, v in current.get("length_hist", {}).items()}
    base_len_counts = {str(k): float(v) for k, v in baseline.get("length_hist", {}).items()}
    cur_item_counts = {str(k): float(v) for k, v in current.get("item_hist", {}).items()}
    base_item_counts = {str(k): float(v) for k, v in baseline.get("item_hist", {}).items()}

    cur_len_prob = _normalise(cur_len_counts)
    base_len_prob = _normalise(base_len_counts)
    cur_item_prob = _normalise(cur_item_counts)
    base_item_prob = _normalise(base_item_counts)

    return {
        "kl_length": _kl_divergence(cur_len_prob, base_len_prob),
        "kl_item": _kl_divergence(cur_item_prob, base_item_prob),
        "chi2_length": _chi_square(cur_len_counts, base_len_counts),
        "chi2_item": _chi_square(cur_item_counts, base_item_counts),
    }


def _resolve_poison_file(directory: str, stem: str, atk_suffix: str | None, ext: str) -> tuple[str, str | None]:
    """Locate a poisoned artefact inside *directory*, allowing flexible naming."""

    suffix = atk_suffix or ""
    candidate = os.path.join(directory, f"{stem}_{suffix}.{ext}") if suffix else os.path.join(directory, f"{stem}.{ext}")
    if os.path.exists(candidate):
        normalized_suffix = suffix or None
        return candidate, normalized_suffix

    pattern = os.path.join(directory, f"{stem}_*.{ext}")
    matches = sorted(glob.glob(pattern))
    if matches:
        chosen = matches[0]
        suffix_part = Path(chosen).stem[len(stem):].lstrip("_") or None
        return chosen, suffix_part

    fallback = os.path.join(directory, f"{stem}.{ext}")
    if os.path.exists(fallback):
        return fallback, None

    raise FileNotFoundError(f"Cannot locate {stem} artefact in {directory}")


image_feature_dim_dict = {
    'vitb32': 512,
    'vitb16': 512,
    'vitl14': 768,
    'rn50': 1024,
    'rn101': 512
}

class VIP5_Dataset(Dataset):
    def __init__(
        self,
        all_tasks,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        mode='train',
        split='toys',
        data_root='data',
        feature_root='features',
        sample_type='random'  
    ):  
        # Print basic dataset parameters for easier debugging of poisoning settings.
        atk_mode = getattr(args, "attack_mode", None)
        atk_mr = getattr(args, "mr", None)
        print(f"[VIP5_Dataset] mode={mode}, attack_mode={atk_mode}, mr={atk_mr}")
        if (
            isinstance(atk_mode, str)
            and atk_mr is not None
            and atk_mode.lower() in ("shadowcastattack", "shadowcast")
            and float(atk_mr) == 0
        ):
            print("[WARNING] ShadowCastAttack with mr=0 detected: no poisoning will be applied.")
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.sample_type = sample_type
        # allow overriding heavy direct-task candidate sampling count when
        # running quick smoke tests.
        self.direct_candidate_num = int(os.environ.get("VIP5_DIRECT_CANDIDATES", "99"))
        self.candidate_target_repeat = max(1, int(getattr(self.args, 'candidate_target_repeat', 10)))
        self.candidate_cache_disabled = os.environ.get("VIP5_DISABLE_CANDIDATE_CACHE", "0") == "1"
        self.candidate_cache: dict[str, list[str]] | None = None
        self.candidate_cache_path: str | None = None
        self.eval_target_item: str | None = None
        try:
            override = getattr(self.args, "eval_target_item", None)
            if override is not None:
                self.eval_target_item = str(override)
        except Exception:
            self.eval_target_item = None
        self.image_feature_size_ratio = args.image_feature_size_ratio
        self.image_feature_type = args.image_feature_type
        assert self.image_feature_type in ['vitb32', 'vitb16', 'vitl14', 'rn50', 'rn101']
        self.image_feature_dim = image_feature_dim_dict[self.image_feature_type]
        self.feature_root = feature_root
        self.data_root = data_root
        self.mode = mode
        poison_subdir_cli = getattr(self.args, "poison_subdir", None)
        poison_subdir_env = os.environ.get("VIP5_POISON_SUBDIR")
        poison_subdir = poison_subdir_cli or poison_subdir_env
        if poison_subdir:
            poison_subdir = str(poison_subdir).strip().strip("/\\")
            if not poison_subdir:
                poison_subdir = None
        self.poison_subdir: str | None = poison_subdir
        text_pair_root_override = os.environ.get("VIP5_TEXT_PAIR_ROOT")
        self.poison_text_root = Path(text_pair_root_override) if text_pair_root_override else DEFAULT_TEXT_PAIR_ROOT

        # 1) 直接用命令行传入的 args.attack_mode 和 args.mr
        atk = self.args.attack_mode       # e.g. "RandomInjectionAttack" / "NoAttack" / ...
        mr  = self.args.mr                # e.g. 0.1, 0.2, etc.



        # CamelCase -> snake_case，再去掉末尾 _attack
        import re
        def camel_to_snake(name: str) -> str:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        atk_snake = camel_to_snake(atk).replace("_attack", "")
        # —— 把 "no" 也当作 NoAttack
        if atk_snake == "no":
            atk_snake = "noattack"

        # —— alias 映射：把各种命名规范统一到数据文件夹使用的名称
        alias_map = {
            "direct_boosting": "direct_boost",
            "popular_item_mimicking": "popular_mimicking",
            "shadow_cast": "shadowcast",
            "shadowcast": "shadowcast",
            # 新增攻击 dcip_ieos，保持名称不变但记录日志
            "dcip_ieos": "dcip_ieos",
            # FC 版本：与 poisoned 输出文件名前缀保持一致
            "dcip_ieos_fc": "dcip_ieos_fc",
            "dcip_ieos_fc_ablation_mode1": "dcip_ieos_fc_ablation_mode1",
            "dcip_ieos_shadowcast": "dcip_ieos_shadowcast",
            "direct_boost": "direct_boost",
        }
        normalized = alias_map.get(atk_snake, atk_snake)
        print(f"[VIP5_Dataset] normalized attack_mode: {atk_snake} -> {normalized}")
        atk_snake = normalized
        shadowcast_mode = (
            atk_snake == "dcip_ieos_shadowcast"
            or (self.poison_subdir and "shadowcast" in self.poison_subdir.lower())
        )
        if shadowcast_mode and not self.poison_subdir:
            raise ValueError("Shadowcast mode requires --poison_subdir to locate text pairs.")
        self.shadowcast_mode = shadowcast_mode


        # 数字 MR 转字符串
        # e.g. 0.1 -> "0.1", 0.2 -> "02"
        mr_val = float(mr)
        mr_str = str(mr_val)

        # 在评估阶段（val/test 模式）统一使用干净数据集进行评测。
        # 如果 mr == 0，则无需加载 poisoned 数据，转而使用干净数据路径
        poison_dir_root = os.path.join(self.data_root, self.split, "poisoned")
        if shadowcast_mode:
            use_poisoned = False
        else:
            use_poisoned = bool(self.poison_subdir) or (
                atk_snake not in ("none", "noattack")
                and self.mode == "train"
                and mr_val > 0
            )

        # Lightweight, explicit runtime confirmation for clean vs poisoned data
        print(
            f"[DATA MODE] mode={self.mode}, atk={atk_snake}, mr={mr_val}, "
            f"use_poisoned={use_poisoned}, poison_subdir={self.poison_subdir}"
        )
        if self.mode == 'train' and not use_poisoned:
            # Extra confirmation line requested for NoAttack/mr=0
            print("[CLEAN-CONFIRM] Training uses CLEAN dataset (NoAttack or mr=0).")

        base_suffix = f"{atk_snake}_mr{mr_str}"
        seq_suffix: str | None = None
        if use_poisoned:
            # 有毒文件都在 data/<split>/poisoned 下
            if self.poison_subdir:
                pois = os.path.join(poison_dir_root, self.poison_subdir)
                if not os.path.isdir(pois):
                    raise FileNotFoundError(f"Poison subdirectory not found: {pois}")
            else:
                pois = poison_dir_root
            exp_splits_path, _ = _resolve_poison_file(pois, "exp_splits", base_suffix, "pkl")
            seq_path, seq_suffix = _resolve_poison_file(pois, "sequential_data", base_suffix, "txt")
            idx_path, _ = _resolve_poison_file(pois, "user_id2idx", base_suffix, "pkl")
            name_path, _ = _resolve_poison_file(pois, "user_id2name", base_suffix, "pkl")
        else:
            # NoAttack ：读取原始数据
            base = os.path.join(self.data_root, self.split)
            exp_splits_path = os.path.join(base, "exp_splits.pkl")
            seq_path        = os.path.join(base, "sequential_data.txt")
            idx_path        = os.path.join(base, "user_id2idx.pkl")
            name_path       = os.path.join(base, "user_id2name.pkl")



        # —— DEBUG 打印，确认到底加载的是哪个文件
        print(f"[DEBUG] exp_splits_path = {exp_splits_path}")
        print(f"[DEBUG] seq_path        = {seq_path}")
        print(f"[DEBUG] idx_path        = {idx_path}")
        print(f"[DEBUG] name_path       = {name_path}")
        # —— 轻量存在性校验（仅打印，不中断训练）
        try:
            print(f"[EXISTS] exp_splits: {os.path.exists(exp_splits_path)} | seq: {os.path.exists(seq_path)} | idx: {os.path.exists(idx_path)} | name: {os.path.exists(name_path)}")
        except Exception:
            pass

        self.shadow_users: set[str] = set()
        if use_poisoned:
            try:
                shadow_meta_path, _ = _resolve_poison_file(pois, "shadow_meta", base_suffix, "json")
            except FileNotFoundError:
                shadow_meta_path = None
            if shadow_meta_path and os.path.exists(shadow_meta_path):
                try:
                    with open(shadow_meta_path, "r", encoding="utf-8") as f:
                        shadow_meta = json.load(f)
                    user_entries = shadow_meta.get("users", {})
                    self.shadow_users = {str(uid) for uid in user_entries.keys()}
                    print(f"[SHADOW] Loaded {len(self.shadow_users)} shadow users from {shadow_meta_path}")
                except Exception as exc:
                    print(f"[SHADOW] Failed to load shadow meta {shadow_meta_path}: {exc}")
                    self.shadow_users = set()

            # 预先尝试加载候选缓存：允许通过环境变量覆盖路径
            candidate_cache_path: str | None = None
            if not self.candidate_cache_disabled:
                cache_override = os.environ.get("VIP5_CANDIDATE_CACHE")
                if cache_override:
                    candidate_cache_path = cache_override
                else:
                    if use_poisoned:
                        base_dir = os.path.dirname(seq_path)
                        if seq_suffix:
                            cache_suffix = seq_suffix
                        elif atk_snake not in ("none", "noattack") or mr_val > 0:
                            cache_suffix = f"{atk_snake}_mr{mr_str}"
                        else:
                            cache_suffix = None
                        cache_name = f"candidate_cache_{cache_suffix}.pkl" if cache_suffix else "candidate_cache_clean.pkl"
                    else:
                        base_dir = os.path.join(self.data_root, self.split)
                        cache_name = "candidate_cache_clean.pkl"
                    candidate_cache_path = os.path.join(base_dir, cache_name)

                if candidate_cache_path and os.path.exists(candidate_cache_path):
                    try:
                        with open(candidate_cache_path, "rb") as f:
                            self.candidate_cache = pickle.load(f)
                        # 确保缓存键统一为 str
                        self.candidate_cache = {
                            str(k): [str(x) for x in v]
                            for k, v in self.candidate_cache.items()
                        }
                        self.candidate_cache_path = candidate_cache_path
                        print(f"[CACHE] Loaded candidate cache from {candidate_cache_path}")
                    except Exception as exc:
                        print(f"[CACHE] Failed to load candidate cache {candidate_cache_path}: {exc}")
                        self.candidate_cache = None
                        self.candidate_cache_path = None
                elif candidate_cache_path:
                    print(f"[CACHE] Candidate cache not found at {candidate_cache_path}; using on-the-fly sampling")


        # 加载 split / seq / user 映射
        exp_splits = load_pickle(exp_splits_path)
        # load raw reviews for B-9
        review_base = os.path.join(self.data_root, self.split, "review_splits.pkl")
        if use_poisoned:
            poison_review = os.path.join(pois, "review_splits.pkl")
            review_path = poison_review if os.path.exists(poison_review) else review_base
        else:
            review_path = review_base
        review_splits = load_pickle(review_path)

        def _uid(d: Dict[str, Any]) -> Any:
            return d.get("reviewerID") or d.get("user_id") or d.get("uid")

        def _asin(d: Dict[str, Any]) -> Any:
            return d.get("asin") or d.get("item_id")

        # build a map (user_id, asin) -> reviewText for all splits
        self._review_map = {
            (_uid(r), _asin(r)): r.get("reviewText", "")
            for sp in ["train", "val", "test"] 
            for r in review_splits.get(sp, [])
            if _uid(r) is not None and _asin(r) is not None
        }


        if self.mode == 'train':
            self.exp_data = exp_splits['train']
        elif self.mode == 'val':
            self.exp_data = exp_splits['val']
        elif self.mode == 'test':
            self.exp_data = exp_splits['test']
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")

        self.shadowcast_entries: List[Dict[str, Any]] = []
        self.shadowcast_text_pairs_path: str | None = None
        if self.shadowcast_mode:
            self._inject_shadowcast_pairs()


        # 3) 加载 sequential_data 文件
        #    （路径已经在上面根据攻击模式和 mr 算好了）
        self.sequential_data = ReadLineFromFile(seq_path)

        # —— 只在 poisoned 且 val/test 模式下过滤掉新注入的 fake 用户
        if use_poisoned and self.mode in ("val", "test"):
            # 1) 先一次性读原始 un-poisoned 序列，拿合法用户集合
            orig_path = os.path.join(self.data_root, self.split, "sequential_data.txt")
            orig_users = { line.split()[0] for line in ReadLineFromFile(orig_path) }
            # 2) 用列表推导保留那些合法的 sequential_data
            before_seq = len(ReadLineFromFile(seq_path))
            self.sequential_data = [
                line for line in self.sequential_data
                if line.split()[0] in orig_users
            ]

            print(f"[DEBUG] Val/Test 模式下，剔除了 {before_seq - len(self.sequential_data)} 条 fake 用户数据")

            if self.shadow_users:
                before_shadow = len(self.sequential_data)
                self.sequential_data = [
                    line for line in self.sequential_data
                    if line.split()[0] not in self.shadow_users
                ]
                removed = before_shadow - len(self.sequential_data)
                if removed > 0:
                    print(f"[DEBUG] Removed {removed} compromised users from sequential data for mode={self.mode}")

            # —— **不要** 再去过滤 self.exp_data ！解释任务需要保留所有 review 样本
    

        # 4) 构建 user_items & 统计 item_count 用于采样
        item_count = defaultdict(int)
        user_items = {}
        for line in self.sequential_data:
            user, items_str = line.strip().split(' ', 1)
            items = [int(x) for x in items_str.split()]
            user_items[user] = items
            for it in items:
                item_count[it] += 1
        self.all_item = list(item_count.keys())
        self.all_item_np = np.array(self.all_item, dtype=np.int64)
        counts = np.array(list(item_count.values()), dtype=float)
        if counts.size:
            prob = counts / counts.sum()
            self.probability_np = prob
            self.probability = prob.tolist()
        else:
            self.probability_np = np.array([], dtype=float)
            self.probability = []
        self.user_items = user_items

        hist_label = os.environ.get("VIP5_HIST_LABEL")
        histogram_path = _histogram_path(self.split, self.mode, hist_label)
        compare_path_env = os.environ.get("VIP5_HIST_COMPARE_PATH")
        compare_path = Path(compare_path_env) if compare_path_env else histogram_path
        record_baseline = os.environ.get("VIP5_HIST_RECORD_BASELINE", "0") == "1"
        compare_baseline = os.environ.get("VIP5_HIST_COMPARE_BASELINE", "0") == "1"

        length_counter = Counter(len(items) for items in user_items.values())
        hist_payload = {
            "split": self.split,
            "mode": self.mode,
            "attack": atk_snake,
            "mr": mr_val,
            "num_users": len(user_items),
            "num_items": len(item_count),
            "length_hist": _counter_to_serializable(length_counter),
            "item_hist": _counter_to_serializable(dict(item_count)),
        }

        if record_baseline:
            _write_histogram(histogram_path, hist_payload)
            print(f"[HIST] Baseline histogram saved to {histogram_path}")

        if compare_baseline:
            baseline = _load_histogram(compare_path)
            if baseline is None:
                print(f"[HIST] Baseline file not found at {compare_path}; skipping comparison.")
            else:
                metrics = _compare_histograms(hist_payload, baseline)
                compare_out = Path(
                    os.environ.get(
                        "VIP5_HIST_COMPARE_LOG",
                        str(histogram_path.parent / f"{(hist_label or f'{self.split}_{self.mode}')}_latest.json"),
                    )
                )
                report = {
                    "baseline_path": str(compare_path),
                    "current": hist_payload,
                    "metrics": metrics,
                }
                _write_histogram(compare_out, report)
                print(
                    "[HIST] KL(length)={kl_len:.4e} KL(item)={kl_item:.4e} "
                    "chi2(length)={chi_len:.4e} chi2(item)={chi_item:.4e}".format(
                        kl_len=metrics["kl_length"],
                        kl_item=metrics["kl_item"],
                        chi_len=metrics["chi2_length"],
                        chi_item=metrics["chi2_item"],
                    )
                )

        base_seed = getattr(self.args, "seed", None)
        self._base_seed = int(base_seed) if base_seed is not None else None
        self._rng = np.random.default_rng(self._base_seed)
        self._worker_rngs: dict[int, np.random.Generator] = {}

        # 如果是 test 模式，加载 negative_samples.txt
        if self.mode == 'test':
            neg_path = os.path.join(self.data_root, self.split, 'negative_samples.txt')
            self.negative_samples = ReadLineFromFile(neg_path)


        # 5) 加载 user_id2idx/user_id2name 映射
        #    （路径已经在上面根据攻击模式和 mr 算好了）
        # 在只有 Explanation 任务时，直接动态构建 reviewerID 的映射
        if set(self.task_list.keys()) == {"explanation"}:
            raw_user2id = {}
            self.user_id2name = {}
            for exp in self.exp_data:
                uid = exp.get("reviewerID")
                if uid not in raw_user2id:
                    raw_user2id[uid] = len(raw_user2id)
                    # 如果有 reviewerName，就用它，否则用 ID
                    self.user_id2name[uid] = exp.get("reviewerName", uid)
            print(f"[DEBUG] Explanation-only，动态构建了 {len(raw_user2id)} 个用户映射")
        else:
            # 否则按原逻辑，从文件里读
            if not os.path.exists(idx_path) or not os.path.exists(name_path):
                # 当没有对应映射文件时，若未使用 poisoned 数据（如在评估阶段），
                # 也允许动态构建用户映射以保证流程继续。
                if atk_snake in ("none", "noattack") or not use_poisoned:
                    raw_user2id = {}
                    self.user_id2name = {}
                    for line in self.sequential_data:
                        uid = line.split()[0]
                        if uid not in raw_user2id:
                            raw_user2id[uid] = len(raw_user2id)
                            self.user_id2name[uid] = uid
                    for exp in self.exp_data:
                        reviewer = exp.get("reviewerID")
                        if reviewer not in raw_user2id:
                            raw_user2id[reviewer] = len(raw_user2id)
                            self.user_id2name[reviewer] = reviewer
                    print(f"[WARN] {('NoAttack' if atk_snake in ('none', 'noattack') else 'Clean val/test')} 模式下，动态构建了 {len(raw_user2id)} 个用户映射")
                else:
                    raise FileNotFoundError(
                        f"Missing poisoned mapping files: {idx_path} or {name_path}"
                    )
            else:
                raw_user2id = load_pickle(idx_path)
                self.user_id2name = load_pickle(name_path)



        # —— 只在 val/test 模式下，统一保留所有任务会用到的用户映射 —— 
        if use_poisoned and self.mode in ("val", "test"):
            keep_users = set()
            # sequential/direct 都是从 sequential_data 拿 user_id
            if "sequential" in self.task_list or "direct" in self.task_list:
                keep_users |= { line.split()[0] for line in self.sequential_data }
            # explanation 用到 exp_data 里的 reviewerID
            if "explanation" in self.task_list:
                keep_users |= { exp.get("reviewerID") for exp in self.exp_data }

            # 过滤原始映射，只留下真正会被 __getitem__ 访问到的用户
            raw_user2id = { u: raw_user2id[u] for u in keep_users if u in raw_user2id }
            self.user_id2name = { u: self.user_id2name[u] for u in keep_users if u in self.user_id2name }

            # 重新给过滤后的用户打连续索引
            new_raw_user2id = {}
            new_user_id2name = {}
            for new_idx, u in enumerate(sorted(raw_user2id.keys())):
                new_raw_user2id[u] = new_idx
                new_user_id2name[u] = self.user_id2name[u]
            raw_user2id = new_raw_user2id
            self.user_id2name = new_user_id2name



        # 5.1) 构建 user2id 和 user_list once after filtering
        #       某些生成脚本会直接使用用户 ID 作为索引值，导致索引不再是
        #       从 0 开始的连续区间。这里根据最大索引值初始化列表，
        #       以避免 list assignment index out of range。
        self.user2id = {str(k): int(v) for k, v in raw_user2id.items()}
        max_idx = max(self.user2id.values()) if self.user2id else -1
        self.user_list = [None] * (max_idx + 1)
        for uid, uidx in self.user2id.items():
            if uidx < 0:
                raise ValueError(f"Invalid user index {uidx} for UID {uid}")
            if uidx >= len(self.user_list):
                # 保险起见，动态扩展列表
                self.user_list.extend([None] * (uidx - len(self.user_list) + 1))
            self.user_list[uidx] = uid

        # 部分解释数据中的 reviewerID 可能未包含在原始映射中
        # 这里在构建完初始映射后，再动态补充缺失的用户，避免 __getitem__ 报 KeyError
        for exp in self.exp_data:
            rid = str(exp.get("reviewerID"))
            if rid and rid not in self.user2id:
                new_idx = len(self.user_list)
                self.user2id[rid] = new_idx
                self.user_list.append(rid)
                if rid not in self.user_id2name:
                    self.user_id2name[rid] = exp.get("reviewerName", rid)

        # 5.2) 构建 direct 任务的“有效用户”列表
        self.direct_user_list = [
            uid for uid in self.user_list
            if uid in self.user_items and len(self.user_items[uid]) > 0
        ]

        # 可选：在投毒训练阶段，剔除 fake 用户以保护直推任务的 HR/NDCG 不被污染
        try:
            exclude_fake_direct = bool(getattr(self.args, 'direct_exclude_fake', False))
        except Exception:
            exclude_fake_direct = False
        if use_poisoned and self.mode == 'train' and exclude_fake_direct:
            def _is_fake_user(u: str) -> bool:
                # user_id2name 的键类型可能为 str 或 int，这里直接用原样键查
                try:
                    key = self.key_convert(u) if hasattr(self, 'key_convert') else u
                except Exception:
                    key = u
                name = self.user_id2name.get(key, '')
                if isinstance(name, str) and name.startswith('fake_'):
                    return True
                # 保底：若 uid 自身形如 'fake_1234'
                return isinstance(u, str) and u.startswith('fake_')

            n_before = len(self.direct_user_list)
            self.direct_user_list = [u for u in self.direct_user_list if not _is_fake_user(u)]
            n_after = len(self.direct_user_list)
            try:
                print(f"[DATA] direct_user_list filtered fake users: {n_before} -> {n_after}")
            except Exception:
                pass



        # 6) 加载 datamaps.json，只取 item2id 和 id2item
        datamaps = load_json(os.path.join(self.data_root, self.split, "datamaps.json"))
        self.item2id = datamaps["item2id"]
        self.id2item = datamaps["id2item"]

        # 7) 根据 user_id2name 的 key 类型，确定转换函数
        if self.user_id2name:
            first_key = next(iter(self.user_id2name))
            self.key_convert = int if isinstance(first_key, int) else str
        else:
            self.key_convert = str

        # 8) 加载 meta.json.gz -> meta_data, 构建 meta_dict
        self.meta_data = [m for m in parse(os.path.join(self.data_root, self.split, 'meta.json.gz'))]
        self.meta_dict = { item['asin']: idx for idx, item in enumerate(self.meta_data) }

        # 9) 加载 item2img_dict.pkl
        self.item2img_dict = load_pickle(os.path.join(self.data_root, self.split, 'item2img_dict.pkl'))

        # 准备 datum_info 用于 __getitem__
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'sequential':
                if sum([0 < int(ind.split('-')[1]) <= 6 or int(ind.split('-')[1]) == 9 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][0]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                if sum([6 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    self.total_length += len(self.sequential_data) * self.sample_numbers[key][1]
                    for i in range(self.total_length - curr):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'direct':
                # 只用 direct_user_list 的长度来计算采样数，跳过那些根本没历史的用户
                valid_n = len(self.direct_user_list)
                # 第一组模板
                if sum([0 < int(ind.split('-')[1]) <= 4 for ind in self.task_list[key]]):
                    count = valid_n * self.sample_numbers[key][0]
                    self.total_length += count
                    for i in range(count):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][0]))
                    curr = self.total_length
                # 第二组模板
                if sum([4 < int(ind.split('-')[1]) <= 8 for ind in self.task_list[key]]):
                    count = valid_n * self.sample_numbers[key][1]
                    self.total_length += count
                    for i in range(count):
                        self.datum_info.append((i + curr, key, i // self.sample_numbers[key][1]))
                    curr = self.total_length
            elif key == 'explanation':
                self.total_length += len(self.exp_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError

    def _shadowcast_pairs_path(self) -> Optional[Path]:
        if not self.poison_subdir:
            return None
        path = self.poison_text_root / self.split / "poisoned" / self.poison_subdir / "text_pairs.jsonl"
        return path if path.exists() else None

    @staticmethod
    def _load_shadowcast_pairs_file(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _inject_shadowcast_pairs(self) -> None:
        pair_path = self._shadowcast_pairs_path()
        if pair_path is None:
            print("[SHADOWCAST] poison_subdir not set; skipping shadowcast text pairs injection.")
            return
        records = self._load_shadowcast_pairs_file(pair_path)
        if not records:
            print(f"[SHADOWCAST] No shadowcast text pairs found at {pair_path}")
            return
        self.shadowcast_text_pairs_path = str(pair_path)
        if self.mode != "train":
            self.shadowcast_entries = []
            return
        entries: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            reviewer_id = f"shadowcast_poison_{idx:06d}"
            feature_vec = record.get("feature", [])
            target_asin = str(record.get("target", "0"))
            entry = {
                "reviewerID": reviewer_id,
                "reviewerName": record.get("reviewer_name", reviewer_id),
                "asin": target_asin,
                "summary": "",
                "overall": 0.0,
                "helpful": [0, 0],
                "feature": feature_vec,
                "shadowcast_feature": feature_vec,
                "explanation": record.get("adversarial_text", ""),
                "reviewText": record.get("adversarial_text", ""),
                "shadowcast": True,
                "shadowcast_original_text": record.get("original_text", ""),
            }
            entries.append(entry)
        if entries:
            self.exp_data.extend(entries)
            self.shadowcast_entries = entries
            print(f"[SHADOWCAST] Injected {len(entries)} poison text pairs from {pair_path}")
        else:
            self.shadowcast_entries = []
    
    def gaussian_sampling(self, datum):
        if self.mode == 'train':
            if int(datum['overall']) == 1:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.0+1.4)/2), std=torch.tensor((1.4-1.0)/4)).item(), 1)
            elif int(datum['overall']) == 2:
                sampled_rating = round(torch.normal(mean=torch.tensor((1.5+2.4)/2), std=torch.tensor((2.4-1.5)/4)).item(), 1)
            elif int(datum['overall']) == 3:
                sampled_rating = round(torch.normal(mean=torch.tensor((2.5+3.4)/2), std=torch.tensor((3.4-2.5)/4)).item(), 1)
            elif int(datum['overall']) == 4:
                sampled_rating = round(torch.normal(mean=torch.tensor((3.5+4.4)/2), std=torch.tensor((4.4-3.5)/4)).item(), 1)
            else:
                sampled_rating = round(torch.normal(mean=torch.tensor((4.5+5.0)/2), std=torch.tensor((5.0-4.5)/4)).item(), 1)
            if sampled_rating > 5.0:
                sampled_rating = 5.0
            if sampled_rating < 1.0:
                sampled_rating = 1.0
            return str(sampled_rating)
        else:
            return int(datum['overall'])


    def _sample_candidate_items(self, user_id: str, user_seq, candidate_num, allow_duplicates=True):
        """Sample candidate items avoiding the user's history.

        Always returns ``candidate_num`` entries; if the catalog is too small,
        duplicates are allowed to guarantee progress.
        """

        if candidate_num <= 0:
            return []

        user_seq_set = {str(u) for u in user_seq}
        candidates: list[str] = []

        # 优先尝试使用离线缓存的候选集合
        if self.candidate_cache:
            cached = self.candidate_cache.get(str(user_id))
            if cached:
                for cand in cached:
                    if cand in user_seq_set:
                        continue
                    if not allow_duplicates and cand in candidates:
                        continue
                    candidates.append(cand)
                    if len(candidates) >= candidate_num:
                        break
                if len(candidates) >= candidate_num:
                    return candidates[:candidate_num]

        attempts = 0
        max_attempts = 10
        rng = self._get_rng()
        catalog = self.all_item_np
        catalog_size = catalog.size

        while len(candidates) < candidate_num and attempts < max_attempts and catalog_size > 0:
            needed = candidate_num - len(candidates)
            replace = needed > catalog_size
            sample_ids = rng.choice(catalog, size=needed, replace=replace)
            for item in sample_ids.tolist():
                item_str = str(item)
                if item_str in user_seq_set:
                    continue
                if not allow_duplicates and item_str in candidates:
                    continue
                candidates.append(item_str)
            attempts += 1

        if len(candidates) < candidate_num:
            base_pool = [str(item) for item in catalog.tolist()] or ["0"]
            seen = set(candidates)
            unique_pool = [item for item in base_pool if item not in user_seq_set]
            pool = unique_pool if unique_pool else base_pool
            allow_seen_items = not unique_pool
            allow_dups_fallback = allow_duplicates or len(seen) >= len(set(pool))
            safety = 0
            max_safety = max(10_000, candidate_num * 10)
            while len(candidates) < candidate_num and pool and safety < max_safety:
                safety += 1
                item_str = str(rng.choice(pool))
                if not allow_seen_items and item_str in user_seq_set:
                    continue
                if not allow_dups_fallback and item_str in seen:
                    continue
                candidates.append(item_str)
                if not allow_dups_fallback:
                    seen.add(item_str)
            if len(candidates) < candidate_num and pool:
                if not allow_dups_fallback:
                    # cannot add more unique entries; exit gracefully
                    return candidates[:candidate_num]
                while len(candidates) < candidate_num:
                    candidates.append(str(rng.choice(pool)))

        return candidates[:candidate_num]


    def _is_fake_user(self, user_id: str) -> bool:
        user_str = str(user_id)
        if user_str in getattr(self, "shadow_users", set()):
            return True
        try:
            key = self.key_convert(user_id)  # type: ignore[arg-type]
        except Exception:
            key = user_id
        name = self.user_id2name.get(key, '') if isinstance(self.user_id2name, dict) else ''
        return isinstance(name, str) and name.startswith('fake_')


    def _prepare_candidate_list(self, candidates: List[str], target_item: str, is_fake_user: bool) -> List[str]:
        processed = [str(c) for c in candidates if c is not None]
        target = str(target_item)
        if self.mode == 'train' and is_fake_user:
            repeats = max(1, self.candidate_target_repeat)
            others = [c for c in processed if c != target]
            processed = [target] * repeats + others
        else:
            random.shuffle(processed)
        return processed[:self.direct_candidate_num]


    def _get_rng(self) -> np.random.Generator:
        """Return a per-worker RNG seeded for reproducibility."""
        worker_info = get_worker_info()
        if worker_info is None:
            return self._rng
        worker_id = int(worker_info.id)
        if worker_id not in self._worker_rngs:
            if self._base_seed is None:
                self._worker_rngs[worker_id] = np.random.default_rng()
            else:
                self._worker_rngs[worker_id] = np.random.default_rng(
                    self._base_seed + worker_id + 1
                )
        return self._worker_rngs[worker_id]

    def _load_feature_override(self, asin: str) -> np.ndarray | None:
        """Load per-poison_subdir feature override if present."""
        if not self.poison_subdir:
            return None
        override_dir = os.path.join(
            self.data_root,
            self.split,
            "poisoned",
            self.poison_subdir,
            "features_override",
        )
        if not os.path.isdir(override_dir):
            return None
        path = os.path.join(override_dir, f"{asin}.npy")
        if not os.path.isfile(path):
            return None
        try:
            return np.load(path)
        except Exception:
            warnings.warn(f"[feature-override] failed to load: {path}")
            return None


    def _load_feature(self, item_id: str) -> np.ndarray:
        """Load image feature safely; return zeros if missing."""
        asin = self.id2item.get(item_id)
        if asin is None:
            warnings.warn(f"item_id {item_id} not in mapping")
            return np.zeros(self.image_feature_dim, dtype=np.float32)
        override = self._load_feature_override(str(asin))
        if override is not None:
            return override
        path = os.path.join(
            self.feature_root,
            f"{self.image_feature_type}_features",
            self.split,
            asin + ".npy",
        )
        try:
            return np.load(path)
        except Exception:
            warnings.warn(f"feature file not found: {path}")
            return np.zeros(self.image_feature_dim, dtype=np.float32)

    def _load_exp_feature_vector(self, exp_datum: Dict[str, Any]) -> np.ndarray:
        if exp_datum.get("shadowcast"):
            feature_vec = exp_datum.get("feature") or exp_datum.get("shadowcast_feature")
            arr = np.asarray(feature_vec, dtype=np.float32) if feature_vec is not None else np.zeros((self.image_feature_dim,), dtype=np.float32)
            arr = arr.reshape(-1)
            if arr.size != self.image_feature_dim:
                arr = np.resize(arr, self.image_feature_dim)
            return arr
        asin = str(exp_datum.get("asin", ""))
        if asin:
            override = self._load_feature_override(asin)
            if override is not None:
                return override
        path = os.path.join(
            self.feature_root,
            f"{self.image_feature_type}_features",
            self.split,
            str(exp_datum["asin"]) + ".npy",
        )
        try:
            return np.load(path)
        except Exception:
            warnings.warn(f"[shadowcast] feature file not found: {path}")
            return np.zeros(self.image_feature_dim, dtype=np.float32)
        
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        loss_weight = 1.0
        uid_for_weight = None  # for fake-sample loss attenutation
        candidate_items = None
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            # 使用转换函数将读取的 user_id 转换为映射中键的类型
            uid = self.key_convert(user_id)
            # for loss weighting
            try:
                uid_for_weight = self.key_convert(user_id)
            except Exception:
                uid_for_weight = user_id
            if uid not in self.user_id2name:
                print(f"[WARN] 用户ID {uid} 不在映射中，使用默认 placeholder")
                user_desc = f"synthetic_user_{uid}"
            else:
                user_desc = self.user_id2name[uid]


            is_fake_user = self._is_fake_user(user_id)

            if self.mode == 'train':
                items_only = sequence[1:]
                if is_fake_user and items_only:
                    if len(items_only) >= 3:
                        history = items_only[:-3]
                        target = items_only[-3]
                    else:
                        history = items_only[:-1]
                        target = items_only[-1]
                else:
                    _, *items = sequence
                    L = len(items)
                    if L == 0:
                        history = []
                        target = random.choice(self.all_item)
                    elif L == 1:
                        history = []
                        target = items[0]
                    else:
                        max_h = min(6, L - 1)
                        hlen = random.randint(1, max_h)
                        end_idx = random.randint(hlen - 1, L - 2)
                        start_idx = end_idx - hlen + 1
                        history = items[start_idx: end_idx + 1]
                        target = items[end_idx + 1]
                purchase_history = [str(x) for x in history]
                target_item = str(target)

            elif self.mode == 'val':
                purchase_history = sequence[1:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                purchase_history = sequence[1:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if task_template['id'] in ['A-1', 'A-2', 'A-3']:
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                else:
                    source_text = task_template['source'].format(user_id, ' {}-> '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(purchase_history), self.image_feature_dim), dtype=np.float32)
                for i in range(len(purchase_history)):
                    feats[i] = self._load_feature(purchase_history[i])
            # 以下部分保持原逻辑，未做修改……
            elif task_template['id'] in ['A-4', 'A-5', 'A-6', 'A-9']:
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                else:
                    source_text = task_template['source'].format(user_desc, ' {}-> '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(purchase_history), self.image_feature_dim), dtype=np.float32)
                for i in range(len(purchase_history)):
                    feats[i] = self._load_feature(purchase_history[i])
            elif task_template['id'] == 'A-7':
                symbol_prob = random.random()
                symbol = ' {}, ' if symbol_prob > 0.5 else ' {}-> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                       feats[i] = self._load_feature(purchase_history[i])
                    feats[-1] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    user_seq_set = {str(item) for item in user_seq}
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        needed = candidate_num - len(candidate_samples)
                        catalog_size = self.all_item_np.size
                        replace = needed > catalog_size if catalog_size else True
                        rng = self._get_rng()
                        if catalog_size == 0:
                            sampled = []
                        elif self.sample_type == 'random' or not self.probability_np.size:
                            sampled = rng.choice(self.all_item_np, size=needed, replace=replace)
                        else:
                            sampled = rng.choice(
                                self.all_item_np,
                                size=needed,
                                replace=replace,
                                p=self.probability_np,
                            )
                        sample_ids = [
                            str(item)
                            for item in (sampled.tolist() if isinstance(sampled, np.ndarray) else sampled)
                            if str(item) not in user_seq_set and str(item) not in candidate_samples
                        ]
                        if not sample_ids:
                            break
                        candidate_samples.extend(sample_ids)
                    if len(candidate_samples) < candidate_num:
                        supplement = self._sample_candidate_items(
                            user_id,
                            user_seq,
                            candidate_num - len(candidate_samples),
                            allow_duplicates=False,
                        )
                        for cand in supplement:
                            if cand not in candidate_samples and cand not in user_seq_set:
                                candidate_samples.append(cand)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_id, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = self._load_feature(purchase_history[i])
                    feats[-1] = self._load_feature(candidate_samples[0])
            elif task_template['id'] == 'A-8':
                symbol_prob = random.random()
                symbol = ' {}, ' if symbol_prob > 0.5 else ' {}-> '
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_desc, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = self._load_feature(purchase_history[i])
                    feats[-1] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    user_seq_set = {str(item) for item in user_seq}
                    candidate_samples = []
                    candidate_num = 1
                    while len(candidate_samples) < candidate_num:
                        rng = self._get_rng()
                        catalog_size = self.all_item_np.size
                        if catalog_size == 0:
                            break
                        replace = catalog_size < candidate_num
                        if self.sample_type == 'random' or not self.probability_np.size:
                            sampled = rng.choice(
                                self.all_item_np,
                                size=candidate_num,
                                replace=replace,
                            )
                        else:
                            sampled = rng.choice(
                                self.all_item_np,
                                size=candidate_num,
                                replace=replace,
                                p=self.probability_np,
                            )
                        sample_ids = [
                            str(item)
                            for item in sampled.tolist()
                            if str(item) not in user_seq_set and str(item) not in candidate_samples
                        ]
                        if not sample_ids:
                            break
                        candidate_samples.extend(sample_ids)
                    if len(candidate_samples) < candidate_num:
                        supplement = self._sample_candidate_items(
                            user_id,
                            user_seq,
                            candidate_num - len(candidate_samples),
                            allow_duplicates=False,
                        )
                        for cand in supplement:
                            if cand not in candidate_samples and cand not in user_seq_set:
                                candidate_samples.append(cand)
                    candidate_samples = candidate_samples[:candidate_num]
                    source_text = task_template['source'].format(user_desc, symbol.format('<extra_id_0> ' * self.image_feature_size_ratio).join(purchase_history) + ' <extra_id_0>' * self.image_feature_size_ratio, candidate_samples[0], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(len(purchase_history)+1, self.image_feature_dim), dtype=np.float32)
                    for i in range(len(purchase_history)):
                        feats[i] = self._load_feature(purchase_history[i])
                    feats[-1] = self._load_feature(candidate_samples[0])
            else:
                raise NotImplementedError
                
        elif task_name == 'direct':

            # 从 direct_user_list（而不是全 user_list）拿 user_id
            user_id = self.direct_user_list[datum_idx]
            # 保证 key_conversion 正确
            uid = self.key_convert(user_id)
            # for loss weighting
            try:
                uid_for_weight = self.key_convert(user_id)
            except Exception:
                uid_for_weight = user_id
            # 这个 user_id 一定在 user_items 里
            seq_items = self.user_items.get(user_id, [])
            # 全部转成字符串
            sequence = [str(it) for it in seq_items]
            uid = self.key_convert(user_id)
            if uid not in self.user_id2name:
                print(f"[WARN] 用户ID {uid} 不在映射中，使用默认 placeholder")
                user_desc = f"synthetic_user_{uid}"
            else:
                user_desc = self.user_id2name[uid]



            is_fake_user = self._is_fake_user(user_id)

            if self.mode == 'train':
                items_only = sequence[1:]
                if is_fake_user and items_only:
                    if len(items_only) >= 3:
                        target_item = items_only[-3]
                    else:
                        target_item = items_only[-1]
                else:
                    target_candidates = sequence[1:-2]
                    if not target_candidates:
                        if len(sequence) >= 2:
                            target_candidates = [sequence[-2]]
                        else:
                            target_candidates = [sequence[-1]]
                    target_item = random.choice(target_candidates)
                target_item = str(target_item)


            elif self.mode == 'val':
                target_item = sequence[-2]
            elif self.mode == 'test':
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            if self.mode != 'train':
                target_item = str(target_item)
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['direct'][task_candidates[task_idx]]
            assert task_template['task'] == 'direct'
            if task_template['id'] == 'B-1':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(user_id, target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = self._sample_candidate_items(user_id, user_seq, 1)
                    source_text = task_template['source'].format(
                        user_id, candidate_samples[0],
                        '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>'
                    )
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(candidate_samples[0])
            elif task_template['id'] == 'B-2':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    source_text = task_template['source'].format(target_item, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>', user_desc)
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = self._sample_candidate_items(user_id, user_seq, 1)
                    source_text = task_template['source'].format(
                        candidate_samples[0],
                        '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>',
                        user_desc,
                    )
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(candidate_samples[0])
            elif task_template['id'] == 'B-3':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    asin = self.id2item.get(target_item)
                    if asin and asin in self.meta_dict and 'title' in self.meta_data[self.meta_dict[asin]]:
                        title = self.meta_data[self.meta_dict[asin]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = self._sample_candidate_items(user_id, user_seq, 1)
                    asin = self.id2item.get(candidate_samples[0])
                    if asin and asin in self.meta_dict and 'title' in self.meta_data[self.meta_dict[asin]]:
                        title = self.meta_data[self.meta_dict[asin]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(
                        user_desc,
                        title,
                        '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>',
                    )
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(candidate_samples[0])
            elif task_template['id'] == 'B-4':
                rand_prob = random.random()
                if rand_prob > 0.5:
                    asin = self.id2item.get(target_item)
                    if asin and asin in self.meta_dict and 'title' in self.meta_data[self.meta_dict[asin]]:
                        title = self.meta_data[self.meta_dict[asin]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                    target_text = task_template['target'].format('yes')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(target_item)
                else:
                    user_seq = self.user_items[user_id]
                    candidate_samples = self._sample_candidate_items(user_id, user_seq, 1)
                    asin = self.id2item.get(candidate_samples[0])
                    if asin and asin in self.meta_dict and 'title' in self.meta_data[self.meta_dict[asin]]:
                        title = self.meta_data[self.meta_dict[asin]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(
                        user_id,
                        title,
                        '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>',
                    )
                    target_text = task_template['target'].format('no')
                    feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                    feats[0] = self._load_feature(candidate_samples[0])
            elif task_template['id'] in ['B-5', 'B-6']:
                user_seq = self.user_items[user_id]
                sample_size = self.direct_candidate_num
                append_item = target_item
                override_target = None
                if self.mode in ("val", "test") and self.eval_target_item is not None:
                    override_target = self.eval_target_item
                if self.mode in ("val", "test") and sample_size > 0:
                    sample_size -= 1
                candidate_samples = self._sample_candidate_items(user_id, user_seq, sample_size)
                if override_target is not None:
                    candidate_samples = [c for c in candidate_samples if c != override_target]
                    candidate_samples.append(str(override_target))
                    append_item = str(override_target)
                else:
                    if target_item in candidate_samples:
                        candidate_samples.remove(target_item)
                    candidate_samples.append(target_item)
                candidate_samples = self._prepare_candidate_list(candidate_samples, target_item, is_fake_user)
                candidate_items = list(candidate_samples)
                source_text = task_template['source'].format(user_desc, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(candidate_samples) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(candidate_samples), self.image_feature_dim), dtype=np.float32)
                for i in range(len(candidate_samples)):
                    feats[i] = self._load_feature(candidate_samples[i])
            elif task_template['id'] in ['B-7', 'B-8']:
                user_seq = self.user_items[user_id]
                sample_size = self.direct_candidate_num
                append_item = target_item
                override_target = None
                if self.mode in ("val", "test") and self.eval_target_item is not None:
                    override_target = self.eval_target_item
                if self.mode in ("val", "test") and sample_size > 0:
                    sample_size -= 1
                candidate_samples = self._sample_candidate_items(user_id, user_seq, sample_size)
                if override_target is not None:
                    candidate_samples = [c for c in candidate_samples if c != override_target]
                    candidate_samples.append(str(override_target))
                    append_item = str(override_target)
                else:
                    if target_item in candidate_samples:
                        candidate_samples.remove(target_item)
                    candidate_samples.append(target_item)
                candidate_samples = self._prepare_candidate_list(candidate_samples, target_item, is_fake_user)
                candidate_items = list(candidate_samples)
                source_text = task_template['source'].format(user_id, ' {}, '.format('<extra_id_0> ' * self.image_feature_size_ratio).join(candidate_samples) + ' <extra_id_0>' * self.image_feature_size_ratio)
                target_text = task_template['target'].format(target_item)
                feats = np.zeros(shape=(len(candidate_samples), self.image_feature_dim), dtype=np.float32)
                for i in range(len(candidate_samples)):
                    feats[i] = self._load_feature(candidate_samples[i])
            # Added for B-9 task
            elif task_template['id'] == 'B-9':
                label = 1 if random.random() > 0.5 else 0
                review = self._review_map.get((uid, target_item), "")
                source_text = task_template['source'].format(
                    user_id=user_id,
                    item_id=target_item,
                    item_photo="<extra_id_0> " * (self.image_feature_size_ratio - 1) + "<extra_id_0>",
                    reviewText=review,
                )
                answer_choices = ['no', 'yes']
            

                # Formatting like "{answer_choices[label]}" cannot be handled
                # directly by ``str.format`` when ``label`` is a variable.
                # The template for task B-9 only contains this placeholder,
                # so we manually select the label instead of using ``format``.
                target_text = answer_choices[label]

                feats = np.zeros((1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_feature(target_item)
                out_dict['reviewText'] = review
            else:
                raise NotImplementedError
                
        elif task_name == 'explanation':
            exp_datum = self.exp_data[datum_idx]
            # for loss weighting (explanation 使用 reviewerID)
            try:
                uid_for_weight = self.key_convert(str(exp_datum.get('reviewerID')))
            except Exception:
                uid_for_weight = str(exp_datum.get('reviewerID'))
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)
            task_template = self.all_tasks['explanation'][task_candidates[task_idx]]
            assert task_template['task'] == 'explanation'
            if task_template['id'] == 'C-1':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-2':
                source_text = task_template['source'].format(exp_datum['summary'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-3':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(self.user2id[exp_datum['reviewerID']], int(exp_datum['overall']), title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-4':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-5':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['summary'], user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-6':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, int(exp_datum['overall']), self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-7':
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-8':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(user_desc, self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(self.gaussian_sampling(exp_datum), exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-9':
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], self.user2id[exp_datum['reviewerID']], title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-10':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                if 'title' in self.meta_data[self.meta_dict[exp_datum['asin']]]:
                    title = self.meta_data[self.meta_dict[exp_datum['asin']]]['title']
                else:
                    title = 'unknown title'
                source_text = task_template['source'].format(exp_datum['feature'], user_desc, title, '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-11':
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), self.user2id[exp_datum['reviewerID']], self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            elif task_template['id'] == 'C-12':
                user_desc = exp_datum['reviewerName'] if 'reviewerName' in exp_datum else exp_datum['reviewerID']
                source_text = task_template['source'].format(exp_datum['feature'], int(exp_datum['overall']), user_desc, self.item2id[exp_datum['asin']], '<extra_id_0> ' * (self.image_feature_size_ratio - 1) + '<extra_id_0>')
                target_text = task_template['target'].format(exp_datum['explanation'])
                feats = np.zeros(shape=(1, self.image_feature_dim), dtype=np.float32)
                feats[0] = self._load_exp_feature_vector(exp_datum)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        input_ids = self.tokenizer.encode(source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        category_ids = [1 if token_id == 32099 else 0 for token_id in input_ids]  # 32099 为 '<extra_id_0>' 的 token id
        
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)
        
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['category_ids'] = torch.LongTensor(category_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)
        
        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text
        out_dict['task'] = task_template['task']
        if candidate_items is not None:
            out_dict['candidate_items'] = candidate_items
        
        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats
        out_dict['vis_feat_length'] = feats.shape[0]
        # --- Per-sample loss weight (protect NDCG by attenuating fake users) ---
        # If fake users are present in poisoned data, 'user_id2name' stores names like 'fake_<uid>'.
        try:
            w_fake = float(getattr(self.args, 'fake_sample_weight', 1.0))
        except Exception:
            w_fake = 1.0
        if w_fake < 1.0:
            w_fake = 1.0
        try:
            if self.mode == 'train' and w_fake != 1.0 and uid_for_weight is not None:
                name = self.user_id2name.get(uid_for_weight, '')
                if isinstance(name, str) and name.startswith('fake_'):
                    loss_weight = float(loss_weight) * w_fake
        except Exception:
            pass
        out_dict['loss_weight'] = loss_weight
        
        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁') or tokenized_text[i] == '<extra_id_0>':
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        # 添加一个 0 表示 </s>
        return whole_word_ids[:len(input_ids) - 1] + [0]
    
    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        args = self.args
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)
        V_W_L = max(entry['vis_feat_length'] for entry in batch)
        
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        category_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        vis_feats = torch.zeros(B, V_W_L, self.image_feature_dim)
        loss_weights = torch.ones(B, dtype=torch.float)
        
        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []
        candidate_items = []
        review_text = []
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            category_ids[i, :entry['input_length']] = entry['category_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            vis_feats[i, :entry['vis_feat_length']] = entry['vis_feats']
            if 'task' in entry:
                tasks.append(entry['task'])
            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])
            if 'candidate_items' in entry:
                candidate_items.append(entry['candidate_items'])
            if 'reviewText' in entry:
                review_text.append(entry['reviewText'])
            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight'] / entry['target_length'] if entry['target_length'] > 0 else entry['loss_weight']
        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks
        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text
        if candidate_items:
            batch_entry['candidate_items'] = candidate_items
        if review_text:
            batch_entry['reviewText'] = review_text
        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['category_ids'] = category_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['vis_feats'] = vis_feats
        batch_entry['loss_weights'] = loss_weights
        batch_entry['vis_token_pos'] = torch.where(category_ids == 1)
        vis_token_pos = [torch.where(row == 1)[0].tolist() for row in category_ids]
        batch_entry['vis_token_pos'] = vis_token_pos
        
        return batch_entry
    
def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False, 
               data_root='data',        # <--- 新增
               feature_root='features'  # <--- 新增
               ):
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)
    
    from all_templates import all_tasks as task_templates
    dataset = VIP5_Dataset(
        task_templates,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        mode=mode,
        split=split,
        data_root=data_root,
        feature_root=feature_root
    )
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    persistent = workers > 0
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn,
            persistent_workers=persistent)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            persistent_workers=persistent)
        
    return loader
