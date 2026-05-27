#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified DCIP-IEOS poisoning pipeline (byzantine user conversion).

This refactored version focuses on the essentials required for the
"interactive push-to-pop" attack:

1. Load anchor vectors and the popular-centre prototype.
2. Obtain cross-modal masks from the victim adapter (falling back to
   deterministic selections when unavailable).
3. Run a short alternating image/text perturbation loop that nudges the target
   representation towards the popular centre while keeping perturbations small.
4. Serialise the resulting fake user interactions together with compact
   telemetry so downstream fine-tuning can adjust the malicious strength.

The implementation deliberately omits the legacy FC/gamma scheduling logic so
that the behaviour is easier to reason about and reproduce.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import copy
import hashlib

import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .interactive import interactive_perturb_target, single_step_perturb
from .victim_adapter import VictimAdapter

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TEXT_PAIR_ROOT = os.path.join(PROJ_ROOT, "poison_text_pairs")
MODE4_IMG_SCALE = float(os.environ.get("VIP5_MODE4_IMG_SCALE", "0.2"))
MODE4_TXT_SCALE = float(os.environ.get("VIP5_MODE4_TXT_SCALE", "0.2"))
MODE4_VIS_RATIO_CAP = float(os.environ.get("VIP5_MODE4_VIS_RATIO_CAP", "0.02"))
MODE4_TXT_RATIO_CAP = float(os.environ.get("VIP5_MODE4_TXT_RATIO_CAP", "0.02"))
MODE4_KEEP_RATIO = float(os.environ.get("VIP5_MODE4_KEEP_RATIO", "0.1"))
RANDOM_ATTACK_TAIL_LEN = int(os.environ.get("VIP5_RANDOM_ATTACK_TAIL_LEN", "10"))
POPULAR_ATTACK_TAIL_LEN = int(os.environ.get("VIP5_POPULAR_ATTACK_TAIL_LEN", "10"))
POPULAR_ATTACK_TOPK = int(os.environ.get("VIP5_POPULAR_ATTACK_TOPK", "30"))
DEFENCE_TEXT_MAX_LEN = int(os.environ.get("VIP5_DEFENCE_TEXT_MAX_LEN", "128"))
DEFENCE_TEXT_HASH_DIM = int(os.environ.get("VIP5_DEFENCE_TEXT_HASH_DIM", "128"))
DEFENCE_EMBED_IMG_WEIGHT = float(os.environ.get("VIP5_DEFENCE_IMG_WEIGHT", "0.5"))
DEFENCE_EMBED_TXT_WEIGHT = float(os.environ.get("VIP5_DEFENCE_TXT_WEIGHT", "0.5"))
DEFENCE_ACT_MIN_SAMPLES = int(os.environ.get("VIP5_DEFENCE_AC_MIN_SAMPLES", "8"))
DEFENCE_ACT_SMALL_RATIO = float(os.environ.get("VIP5_DEFENCE_AC_SMALL_RATIO", "0.35"))
DEFENCE_ACT_MAX_ITER = int(os.environ.get("VIP5_DEFENCE_AC_MAX_ITER", "25"))
DEFENCE_ACT_MAX_DROP = int(os.environ.get("VIP5_DEFENCE_AC_MAX_DROP", "0"))
DEFENCE_ACT_MAX_DROP_RATIO = float(os.environ.get("VIP5_DEFENCE_AC_MAX_DROP_RATIO", "0"))
DEFENCE_SS_TOP_RATIO = float(os.environ.get("VIP5_DEFENCE_SS_TOP_RATIO", "0.01"))
DEFENCE_SS_MIN_SAMPLES = int(os.environ.get("VIP5_DEFENCE_SS_MIN_SAMPLES", "20"))
append_mode_registry: Dict[str, bool] = {}

# Global pools populated at runtime for sampling strategies.
GLOBAL_POPULAR_ITEMS: List[Any] = []
GLOBAL_MID_POP_ITEMS: List[Any] = []


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _format_float_for_name(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _sanitize_subdir(name: str) -> str:
    candidate = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    candidate = candidate.strip("/\\")
    if not candidate or candidate in {".", ".."}:
        raise ValueError(f"Invalid poison subdirectory name: {name!r}")
    return candidate


def _derive_poison_subdir(
    args: Any,
    *,
    mr: float,
    interaction_rounds: int,
    img_eps: float,
    text_ratio_max: float,
    ablation_tag: str | None = None,
    base_label: str = "dcip_ieos_fc",
) -> str:
    supplied = getattr(args, "poison_subdir", None)
    if supplied:
        return _sanitize_subdir(supplied)

    chosen_label = base_label
    if ablation_tag:
        chosen_label = f"{chosen_label}_{ablation_tag}"
    mr_part = _format_float_for_name(mr)
    img_part = _format_float_for_name(img_eps)
    txt_part = _format_float_for_name(text_ratio_max)
    return f"{chosen_label}_mr{mr_part}_ir{interaction_rounds}_img{img_part}_txt{txt_part}"


def _copy_support_files(split_dir: str, target_dir: str) -> None:
    """Mirror clean NoAttack support artefacts into the poison subdirectory."""

    support_files = [
        "review_splits.pkl",
        "candidate_cache_clean.pkl",
    ]
    os.makedirs(target_dir, exist_ok=True)
    for filename in support_files:
        src = os.path.join(split_dir, filename)
        dst = os.path.join(target_dir, filename)
        if not os.path.isfile(src) or os.path.isfile(dst):
            continue
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            logging.warning("[poison-pipeline] failed to copy %s -> %s: %s", src, dst, exc)


def _sanitize_tag(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return text.strip("_-.") or "defence"


def _write_poison_artifacts(
    out_dir: str,
    suffix: str,
    *,
    combined_sequences: list[str],
    exp_splits: dict[str, Any],
    user_id2idx: dict[str, Any],
    user_id2name: dict[str, Any],
    keywords_map: dict[str, Any],
    embedding_deltas: dict[str, Any],
    round_metrics: dict[str, Any],
    summary: dict[str, Any],
    shadow_meta: dict[str, Any],
    shadow_meta_meta: dict[str, Any],
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    seq_out = os.path.join(out_dir, f"sequential_data{suffix}.txt")
    exp_out = os.path.join(out_dir, f"exp_splits{suffix}.pkl")
    idx_out = os.path.join(out_dir, f"user_id2idx{suffix}.pkl")
    name_out = os.path.join(out_dir, f"user_id2name{suffix}.pkl")
    kw_out = os.path.join(out_dir, f"keywords{suffix}.pkl")
    delta_out = os.path.join(out_dir, f"embedding_deltas{suffix}.pkl")
    metrics_out = os.path.join(out_dir, f"round_metrics{suffix}.pkl")
    summary_out = os.path.join(out_dir, f"poison_summary{suffix}.json")
    shadow_meta_out = os.path.join(out_dir, f"shadow_meta{suffix}.json")

    with open(seq_out, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_sequences) + ("\n" if combined_sequences else ""))
    _dump_pickle(exp_out, exp_splits)
    _dump_pickle(idx_out, user_id2idx)
    _dump_pickle(name_out, user_id2name)
    _dump_pickle(kw_out, keywords_map)
    _dump_pickle(delta_out, embedding_deltas)
    _dump_pickle(metrics_out, round_metrics)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(shadow_meta_out, "w", encoding="utf-8") as f:
        json.dump({"users": shadow_meta, "meta": shadow_meta_meta}, f, ensure_ascii=False, indent=2)
    return {
        "sequential_path": seq_out,
        "exp_splits_path": exp_out,
        "user_id2idx_path": idx_out,
        "user_id2name_path": name_out,
        "keywords_path": kw_out,
        "embedding_deltas_path": delta_out,
        "round_metrics_path": metrics_out,
        "summary_path": summary_out,
        "shadow_meta_path": shadow_meta_out,
        "poison_dir": out_dir,
    }


def _cleanup_counter(counter: Counter[str]) -> Counter[str]:
    for key in list(counter.keys()):
        if counter[key] <= 0:
            del counter[key]
    return counter


def _load_histogram_baseline(path: str) -> Dict[str, Any]:
    if not path:
        raise ValueError("Histogram defence requires --defence-baseline path.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "length_hist" not in data or "item_hist" not in data:
        raise ValueError(f"Baseline file {path} missing length_hist/item_hist.")
    return data


def _default_hist_threshold(baseline: Dict[str, Any]) -> float:
    num_users = float(baseline.get("num_users") or baseline.get("config", {}).get("num_users") or 1.0)
    return max(1e-4, 1.0 / max(num_users, 1.0))


def _counter_from_hist(hist: Dict[str, Any]) -> Counter[str]:
    return Counter({str(k): float(v) for k, v in hist.items() if float(v) > 0})


def _normalise(counter: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in counter.values())
    if total <= 0:
        return {key: 0.0 for key in counter.keys()}
    return {key: float(value) / total for key, value in counter.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    value = 0.0
    keys = set(p.keys()) | set(q.keys())
    for key in keys:
        pv = float(p.get(key, 0.0))
        qv = float(q.get(key, 0.0))
        if pv <= 0.0:
            continue
        value += pv * math.log(pv / max(qv, eps))
    return float(value)


def _kl_from_counters(
    length_counts: Counter[str],
    item_counts: Counter[str],
    base_len_prob: Dict[str, float],
    base_item_prob: Dict[str, float],
) -> Dict[str, float]:
    cur_len_prob = _normalise(dict(length_counts))
    cur_item_prob = _normalise(dict(item_counts))
    return {
        "kl_length": _kl_divergence(cur_len_prob, base_len_prob),
        "kl_item": _kl_divergence(cur_item_prob, base_item_prob),
    }


def _build_user_contribs(shadow_meta: Dict[str, Any], users: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    contribs: Dict[str, Dict[str, Any]] = {}
    for user in users:
        entry = shadow_meta.get(user)
        if not entry:
            continue
        history = entry.get("history") or []
        history_list = [str(it) for it in history]
        contribs[user] = {
            "length_key": str(len(history_list)),
            "item_counts": Counter(history_list),
        }
    return contribs


def _apply_histogram_defence(
    defence_label: str,
    baseline_path: str,
    threshold: float | None,
    *,
    combined_sequences: list[str],
    exp_splits: dict[str, Any],
    user_id2idx: dict[str, Any],
    user_id2name: dict[str, Any],
    keywords_map: dict[str, Any],
    embedding_deltas: dict[str, Any],
    round_metrics: dict[str, Any],
    summary: dict[str, Any],
    shadow_meta: dict[str, Any],
    compromised_users: list[str],
    attack_variant: str,
    base_count_for_mr: int,
) -> Dict[str, Any] | None:
    baseline = _load_histogram_baseline(baseline_path)
    thresh = threshold if threshold is not None else _default_hist_threshold(baseline)
    base_len_counts = _counter_from_hist(baseline.get("length_hist", {}))
    base_item_counts = _counter_from_hist(baseline.get("item_hist", {}))
    base_len_prob = _normalise(dict(base_len_counts))
    base_item_prob = _normalise(dict(base_item_counts))

    user_contribs = _build_user_contribs(shadow_meta, compromised_users)
    ordered_users = [u for u in compromised_users if u in user_contribs]
    if not ordered_users:
        logging.warning("[defence:%s] No usable shadow users found for defence.", defence_label)
        return None

    counts_len = Counter(base_len_counts)
    counts_item = Counter(base_item_counts)
    for user in ordered_users:
        contrib = user_contribs[user]
        counts_len[contrib["length_key"]] += 1
        counts_item.update(contrib["item_counts"])
    counts_len = _cleanup_counter(counts_len)
    counts_item = _cleanup_counter(counts_item)

    metrics = _kl_from_counters(counts_len, counts_item, base_len_prob, base_item_prob)
    dropped: list[str] = []
    kept_set = set(ordered_users)

    def current_score(values: Dict[str, float]) -> float:
        return values["kl_length"] + values["kl_item"]

    while kept_set and (metrics["kl_length"] > thresh or metrics["kl_item"] > thresh):
        best_user = None
        best_metrics = None
        best_reduction = None
        for user in list(kept_set):
            contrib = user_contribs[user]
            len_tmp = counts_len.copy()
            len_key = contrib["length_key"]
            len_tmp[len_key] -= 1
            len_tmp = _cleanup_counter(len_tmp)

            item_tmp = counts_item.copy()
            item_tmp.subtract(contrib["item_counts"])
            item_tmp = _cleanup_counter(item_tmp)

            tmp_metrics = _kl_from_counters(len_tmp, item_tmp, base_len_prob, base_item_prob)
            reduction = current_score(metrics) - current_score(tmp_metrics)
            if best_reduction is None or reduction > best_reduction:
                best_reduction = reduction
                best_user = user
                best_metrics = tmp_metrics
        if best_user is None:
            break
        kept_set.remove(best_user)
        dropped.append(best_user)
        counts_len[ user_contribs[best_user]["length_key"] ] -= 1
        counts_len = _cleanup_counter(counts_len)
        counts_item.subtract(user_contribs[best_user]["item_counts"])
        counts_item = _cleanup_counter(counts_item)
        metrics = best_metrics or metrics

    kept_users = [u for u in ordered_users if u in kept_set]
    if not dropped:
        logging.info("[defence:%s] No users dropped; KL length=%.4e, KL item=%.4e", defence_label, metrics["kl_length"], metrics["kl_item"])
    else:
        logging.info(
            "[defence:%s] Dropped %d/%d users to reach KL length=%.4e KL item=%.4e",
            defence_label,
            len(dropped),
            len(ordered_users),
            metrics["kl_length"],
            metrics["kl_item"],
        )

    drop_set = set(dropped)
    filtered_sequences = [
        line for line in combined_sequences if line.split()[0] not in drop_set
    ]
    filtered_exp_splits: dict[str, Any] = {}
    for split_name, records in exp_splits.items():
        if isinstance(records, list):
            filtered_exp_splits[split_name] = [
                copy.deepcopy(entry)
                for entry in records
                if str(entry.get("reviewerID")) not in drop_set
            ]
        else:
            filtered_exp_splits[split_name] = copy.deepcopy(records)

    filtered_user_id2idx = {k: v for k, v in user_id2idx.items() if str(k) not in drop_set}
    filtered_user_id2name = {k: v for k, v in user_id2name.items() if str(k) not in drop_set}
    filtered_keywords = {k: v for k, v in keywords_map.items() if k not in drop_set}
    filtered_embedding = {k: v for k, v in embedding_deltas.items() if k not in drop_set}
    filtered_round_metrics = {k: v for k, v in round_metrics.items() if k not in drop_set}
    filtered_shadow_meta = {k: v for k, v in shadow_meta.items() if k not in drop_set}

    filtered_summary = copy.deepcopy(summary)
    filtered_summary["compromised_user_count"] = len(kept_users)
    filtered_summary["compromised_users"] = kept_users
    effective_mr = float(len(kept_users) / base_count_for_mr) if base_count_for_mr else 0.0
    filtered_summary.setdefault("config", {})["effective_mr_defence"] = effective_mr
    filtered_summary["defence"] = {
        "name": defence_label,
        "baseline_path": baseline_path,
        "kl_length": metrics["kl_length"],
        "kl_item": metrics["kl_item"],
        "threshold": thresh,
        "dropped_users": dropped,
    }

    meta_payload = {
        "attack_mode": f"{attack_variant}+{defence_label}",
        "count": len(kept_users),
        "defence": filtered_summary["defence"],
    }

    return {
        "combined_sequences": filtered_sequences,
        "exp_splits": filtered_exp_splits,
        "user_id2idx": filtered_user_id2idx,
        "user_id2name": filtered_user_id2name,
        "keywords_map": filtered_keywords,
        "embedding_deltas": filtered_embedding,
        "round_metrics": filtered_round_metrics,
        "summary": filtered_summary,
        "shadow_meta": filtered_shadow_meta,
        "shadow_meta_meta": meta_payload,
        "kept_users": kept_users,
        "dropped_users": dropped,
        "metrics": metrics,
    }


def _load_histories_from_seq(path: str, *, limit: int | None = None) -> Dict[str, List[str]]:
    histories: Dict[str, List[str]] = {}
    if not os.path.isfile(path):
        return histories
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            histories[parts[0]] = parts[1:]
            if limit and len(histories) >= limit:
                break
    return histories


def _compute_pop_stats(pop_counter: Dict[str, int]) -> Dict[str, float]:
    values = [float(v) for v in pop_counter.values() if v > 0]
    if not values:
        values = [1.0]
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
    }


def _build_ae_feature(history: List[str], pop_counter: Dict[str, int], pop_stats: Dict[str, float]) -> np.ndarray:
    length = len(history)
    if length == 0:
        return np.zeros(5, dtype=np.float32)
    unique_ratio = len(set(history)) / length
    pop_values = np.array([float(pop_counter.get(str(item), 1)) for item in history], dtype=np.float32)
    mean_pop = float(np.mean(pop_values)) if pop_values.size else 0.0
    std_pop = float(np.std(pop_values)) if pop_values.size else 0.0
    tail_ratio = float(np.mean(pop_values <= pop_stats.get("median", 1.0))) if pop_values.size else 0.0
    max_pop = float(np.max(pop_values)) if pop_values.size else 0.0
    return np.array([float(length), unique_ratio, mean_pop, std_pop, tail_ratio + max_pop * 1e-3], dtype=np.float32)


class _SimpleAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(1, min(hidden_dim, max(1, input_dim - 1)))
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encoder(x)
        return self.decoder(z)


def _train_autoencoder(features: np.ndarray, hidden_dim: int, epochs: int) -> _SimpleAE:
    input_dim = features.shape[1]
    model = _SimpleAE(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    data = torch.tensor(features, dtype=torch.float32)
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        recon = model(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _apply_ae_defence(
    defence_label: str,
    reference_seq: str | None,
    threshold: float,
    hidden_dim: int,
    epochs: int,
    *,
    combined_sequences: list[str],
    exp_splits: dict[str, Any],
    user_id2idx: dict[str, Any],
    user_id2name: dict[str, Any],
    keywords_map: dict[str, Any],
    embedding_deltas: dict[str, Any],
    round_metrics: dict[str, Any],
    summary: dict[str, Any],
    shadow_meta: dict[str, Any],
    compromised_users: list[str],
    attack_variant: str,
    base_count_for_mr: int,
    pop_counter: Dict[str, int],
    split_dir: str,
) -> Dict[str, Any] | None:
    seq_path = reference_seq or os.path.join(split_dir, "sequential_data.txt")
    ref_histories = _load_histories_from_seq(seq_path, limit=int(os.environ.get("VIP5_AE_REF_LIMIT", "10000")))
    if not ref_histories:
        logging.warning("[defence:%s] No reference histories found for AE defence.", defence_label)
        return None
    pop_counter = {str(k): int(v) for k, v in pop_counter.items() if int(v) > 0}
    if not pop_counter:
        pop_counter = {"0": 1}
    pop_stats = _compute_pop_stats(pop_counter)
    ref_features = [
        _build_ae_feature(items, pop_counter, pop_stats)
        for items in ref_histories.values()
        if items
    ]
    if not ref_features:
        logging.warning("[defence:%s] Reference features empty.", defence_label)
        return None
    ref_arr = np.stack(ref_features)
    mean = ref_arr.mean(axis=0)
    std = ref_arr.std(axis=0)
    std[std < 1e-6] = 1.0
    ref_norm = (ref_arr - mean) / std
    model = _train_autoencoder(ref_norm, hidden_dim, epochs)

    user_features: Dict[str, np.ndarray] = {}
    ordered_users: List[str] = []
    for user in compromised_users:
        entry = shadow_meta.get(user)
        if not entry:
            continue
        history = entry.get("history") or []
        if not history:
            continue
        feat = _build_ae_feature(history, pop_counter, pop_stats)
        user_features[user] = feat
        ordered_users.append(user)
    if not ordered_users:
        logging.warning("[defence:%s] No poisoning users with valid histories for AE defence.", defence_label)
        return None

    poison_arr = np.stack([user_features[u] for u in ordered_users])
    poison_norm = (poison_arr - mean) / std
    with torch.no_grad():
        tensor = torch.tensor(poison_norm, dtype=torch.float32)
        recon = model(tensor)
        errors = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
    drop_set = {u for u, err in zip(ordered_users, errors) if err > threshold}
    kept_users = [u for u in ordered_users if u not in drop_set]
    logging.info(
        "[defence:%s] AE dropped %d/%d users (threshold=%.4f, max_err=%.4e)",
        defence_label,
        len(drop_set),
        len(ordered_users),
        threshold,
        float(errors.max()) if errors.size else 0.0,
    )

    filtered_sequences = [
        line for line in combined_sequences if line.split()[0] not in drop_set
    ]
    filtered_exp_splits = {}
    for split_name, records in exp_splits.items():
        if isinstance(records, list):
            filtered_exp_splits[split_name] = [
                copy.deepcopy(entry)
                for entry in records
                if str(entry.get("reviewerID")) not in drop_set
            ]
        else:
            filtered_exp_splits[split_name] = copy.deepcopy(records)

    filtered_user_id2idx = {k: v for k, v in user_id2idx.items() if str(k) not in drop_set}
    filtered_user_id2name = {k: v for k, v in user_id2name.items() if str(k) not in drop_set}
    filtered_keywords = {k: v for k, v in keywords_map.items() if k not in drop_set}
    filtered_embedding = {k: v for k, v in embedding_deltas.items() if k not in drop_set}
    filtered_round_metrics = {k: v for k, v in round_metrics.items() if k not in drop_set}
    filtered_shadow_meta = {k: v for k, v in shadow_meta.items() if k not in drop_set}

    filtered_summary = copy.deepcopy(summary)
    filtered_summary["compromised_user_count"] = len(kept_users)
    filtered_summary["compromised_users"] = kept_users
    effective_mr = float(len(kept_users) / base_count_for_mr) if base_count_for_mr else 0.0
    filtered_summary.setdefault("config", {})["effective_mr_defence"] = effective_mr
    filtered_summary["defence"] = {
        "name": defence_label,
        "threshold": threshold,
        "dropped": list(drop_set),
        "max_error": float(errors.max()) if errors.size else 0.0,
    }

    meta_payload = {
        "attack_mode": f"{attack_variant}+{defence_label}",
        "count": len(kept_users),
        "defence": filtered_summary["defence"],
    }

    return {
        "combined_sequences": filtered_sequences,
        "exp_splits": filtered_exp_splits,
        "user_id2idx": filtered_user_id2idx,
        "user_id2name": filtered_user_id2name,
        "keywords_map": filtered_keywords,
        "embedding_deltas": filtered_embedding,
        "round_metrics": filtered_round_metrics,
        "summary": filtered_summary,
        "shadow_meta": filtered_shadow_meta,
        "shadow_meta_meta": meta_payload,
        "kept_users": kept_users,
        "dropped_users": list(drop_set),
        "metrics": {"max_error": float(errors.max()) if errors.size else 0.0},
    }


def _hash_text_vector(text: str, dim: int) -> np.ndarray:
    if dim <= 0:
        dim = 1
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text or "").split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _build_history_embedding(history: Iterable[Any], dim: int) -> np.ndarray:
    tokens = [str(item) for item in history if item is not None]
    return _hash_text_vector(" ".join(tokens), dim)


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std


def _kmeans_2(matrix: np.ndarray, *, max_iter: int, seed: int = 0) -> np.ndarray:
    num = matrix.shape[0]
    if num <= 1:
        return np.zeros(num, dtype=int)
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(num, size=2, replace=False)
    centroids = matrix[init_idx].copy()
    labels = np.zeros(num, dtype=int)
    for _ in range(max(1, max_iter)):
        dists = np.sum((matrix[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for k in range(2):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = matrix[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels


def _build_defence_embedding(
    adapter: Any,
    image_feat: Any,
    text: str,
    *,
    text_max_len: int,
    text_hash_dim: int,
    w_img: float,
    w_txt: float,
) -> np.ndarray:
    img_flat = np.asarray(image_feat, dtype=np.float32).reshape(-1)
    if img_flat.size == 0:
        img_flat = np.zeros(1, dtype=np.float32)
    target_dim = None
    if adapter is not None:
        model = getattr(adapter, "model", None)
        if model is not None:
            target_dim = getattr(getattr(model, "config", None), "d_model", None)
            if target_dim:
                target_dim = int(target_dim)

    def _fit_dim(vec: np.ndarray) -> np.ndarray:
        if target_dim and target_dim > 0:
            if vec.size > target_dim:
                vec = vec[:target_dim]
            elif vec.size < target_dim:
                pad = np.zeros(target_dim - vec.size, dtype=np.float32)
                vec = np.concatenate([vec, pad])
        return vec

    if adapter is None or torch is None or np is None:
        txt_vec = _hash_text_vector(text, text_hash_dim)
        fused = np.concatenate([img_flat, txt_vec]).astype(np.float32)
        return _fit_dim(fused)
    try:
        if not hasattr(adapter, "_ensure_model_ready") or not adapter._ensure_model_ready():
            raise RuntimeError("adapter not ready")
        model = getattr(adapter, "model", None)
        tokenizer = getattr(adapter, "tokenizer", None)
        if model is None or tokenizer is None:
            raise RuntimeError("missing model/tokenizer")
        img_raw = np.asarray(image_feat, dtype=np.float32)
        if img_raw.ndim == 1:
            img_raw = img_raw.reshape(1, 1, -1)
        elif img_raw.ndim == 2:
            img_raw = img_raw.reshape(1, img_raw.shape[0], img_raw.shape[1])
        elif img_raw.ndim != 3:
            img_raw = img_flat.reshape(1, 1, -1)

        feat_dim = getattr(getattr(model, "config", None), "feat_dim", None)
        if feat_dim:
            feat_dim = int(feat_dim)
            if img_raw.shape[-1] != feat_dim:
                if img_raw.shape[-1] > feat_dim:
                    img_raw = img_raw[..., :feat_dim]
                else:
                    pad = np.zeros((*img_raw.shape[:-1], feat_dim - img_raw.shape[-1]), dtype=np.float32)
                    img_raw = np.concatenate([img_raw, pad], axis=-1)

        vis_feats = torch.as_tensor(img_raw, dtype=torch.float32, device=adapter.device)
        with torch.no_grad():
            vis_emb = model.encoder.visual_embedding(vis_feats)
        img_vec = vis_emb.mean(dim=1).squeeze(0).detach().cpu().numpy()

        text_str = str(text or "")
        input_ids = None
        try:
            batch = tokenizer(text_str, return_tensors="pt", truncation=True, max_length=text_max_len)
            input_ids = batch.get("input_ids")
        except TypeError:
            input_ids = None
        if input_ids is not None and not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids is None:
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode(text_str)
                if tokens and isinstance(tokens[0], int):
                    ids = tokens[:text_max_len]
                else:
                    ids = list(range(min(len(tokens), text_max_len)))
                input_ids = torch.tensor([ids], dtype=torch.long)
            elif hasattr(tokenizer, "tokenize"):
                tokens = tokenizer.tokenize(text_str)
                ids = list(range(min(len(tokens), text_max_len)))
                input_ids = torch.tensor([ids], dtype=torch.long)

        if input_ids is None or input_ids.numel() == 0:
            txt_vec = np.zeros_like(img_vec)
        else:
            input_ids = input_ids.to(adapter.device)
            with torch.no_grad():
                txt_emb = model.encoder.embed_tokens(input_ids)
            txt_vec = txt_emb.mean(dim=1).squeeze(0).detach().cpu().numpy()

        if img_vec.ndim == 1 and txt_vec.ndim == 1 and img_vec.shape == txt_vec.shape:
            fused = (w_img * img_vec) + (w_txt * txt_vec)
        else:
            fused = np.concatenate([img_vec.reshape(-1), txt_vec.reshape(-1)])
        fused = fused.astype(np.float32).reshape(-1)
        return _fit_dim(fused)
    except Exception:
        txt_vec = _hash_text_vector(text, text_hash_dim)
        fused = np.concatenate([img_flat, txt_vec]).astype(np.float32)
        return _fit_dim(fused)


def _collect_poison_entries(exp_splits: dict[str, Any], compromised_users: list[str]) -> Dict[str, Dict[str, Any]]:
    compromised_set = {str(u) for u in compromised_users}
    entries: Dict[str, Dict[str, Any]] = {}
    for records in exp_splits.values():
        if not isinstance(records, list):
            continue
        for entry in records:
            user_id = str(entry.get("reviewerID", ""))
            if user_id in compromised_set and user_id not in entries:
                entries[user_id] = entry
    return entries


def _apply_activation_clustering_defence(
    defence_label: str,
    adapter: Any,
    min_samples: int,
    small_ratio: float,
    max_iter: int,
    max_drop: int | None,
    max_drop_ratio: float | None,
    *,
    combined_sequences: list[str],
    exp_splits: dict[str, Any],
    user_id2idx: dict[str, Any],
    user_id2name: dict[str, Any],
    keywords_map: dict[str, Any],
    embedding_deltas: dict[str, Any],
    round_metrics: dict[str, Any],
    summary: dict[str, Any],
    shadow_meta: dict[str, Any],
    compromised_users: list[str],
    attack_variant: str,
    base_count_for_mr: int,
) -> Dict[str, Any] | None:
    entry_map = _collect_poison_entries(exp_splits, compromised_users)
    if not entry_map:
        logging.warning("[defence:%s] No poisoned entries found for activation clustering.", defence_label)
        return None

    user_vectors: Dict[str, np.ndarray] = {}
    for user_id, entry in entry_map.items():
        vec = _build_defence_embedding(
            adapter,
            entry.get("feature", []),
            entry.get("reviewText", ""),
            text_max_len=DEFENCE_TEXT_MAX_LEN,
            text_hash_dim=DEFENCE_TEXT_HASH_DIM,
            w_img=DEFENCE_EMBED_IMG_WEIGHT,
            w_txt=DEFENCE_EMBED_TXT_WEIGHT,
        )
        if vec.size > 0:
            user_vectors[user_id] = vec

    if not user_vectors:
        logging.warning("[defence:%s] No usable embeddings for activation clustering.", defence_label)
        return None
    embedding_source = "feature_text"
    if len(user_vectors) > 1:
        mat = np.stack(list(user_vectors.values()), axis=0)
        max_var = float(np.max(np.var(mat, axis=0)))
        if max_var < 1e-8:
            history_vectors: Dict[str, np.ndarray] = {}
            for user_id in user_vectors.keys():
                history = (shadow_meta.get(user_id) or {}).get("history") or []
                if history:
                    history_vectors[user_id] = _build_history_embedding(history, DEFENCE_TEXT_HASH_DIM)
            if history_vectors:
                for user_id, vec in user_vectors.items():
                    history_vectors.setdefault(user_id, vec)
                user_vectors = history_vectors
                embedding_source = "history"
                logging.info(
                    "[defence:%s] Embeddings nearly identical; falling back to history embeddings.",
                    defence_label,
                )

    target_groups: Dict[str, List[str]] = defaultdict(list)
    for user_id, entry in entry_map.items():
        if user_id not in user_vectors:
            continue
        target = str(entry.get("asin", "") or "")
        if not target:
            continue
        target_groups[target].append(user_id)

    drop_set: set[str] = set()
    dropped_by_target: Dict[str, List[str]] = {}
    drop_scores: Dict[str, float] = {}
    for target, users in target_groups.items():
        if len(users) < min_samples:
            continue
        users = sorted(users)
        vectors = np.stack([user_vectors[u] for u in users], axis=0)
        vectors = _standardize_matrix(vectors)
        labels = _kmeans_2(vectors, max_iter=max_iter, seed=0)
        counts = np.bincount(labels, minlength=2)
        if counts.sum() == 0:
            continue
        small_label = int(np.argmin(counts))
        ratio = float(counts[small_label] / max(1, counts.sum()))
        if ratio <= small_ratio and counts[small_label] > 0:
            dropped = [u for u, lab in zip(users, labels) if lab == small_label]
            centroid = vectors[labels == small_label].mean(axis=0) if np.any(labels == small_label) else None
            for u, lab, vec in zip(users, labels, vectors):
                if lab != small_label:
                    continue
                if centroid is None:
                    score = 0.0
                else:
                    score = float(np.linalg.norm(vec - centroid))
                drop_scores[u] = score
            drop_set.update(dropped)
            dropped_by_target[target] = dropped

    if max_drop is None:
        max_drop = DEFENCE_ACT_MAX_DROP
    max_drop = int(max_drop) if max_drop is not None else 0
    if max_drop <= 0:
        max_drop = None

    if max_drop_ratio is None:
        max_drop_ratio = DEFENCE_ACT_MAX_DROP_RATIO
    max_drop_ratio = float(max_drop_ratio) if max_drop_ratio is not None else 0.0
    if max_drop_ratio <= 0.0:
        max_drop_ratio = None

    allowed_drop = len(drop_set)
    if max_drop_ratio is not None:
        allowed_drop = min(allowed_drop, max(1, int(math.ceil(len(compromised_users) * max_drop_ratio))))
    if max_drop is not None:
        allowed_drop = min(allowed_drop, max_drop)

    if allowed_drop < len(drop_set):
        candidates = sorted(drop_set, key=lambda u: (-drop_scores.get(u, 0.0), str(u)))
        drop_set = set(candidates[:allowed_drop])
        filtered_by_target: Dict[str, List[str]] = {}
        for target, users in dropped_by_target.items():
            kept = [u for u in users if u in drop_set]
            if kept:
                filtered_by_target[target] = kept
        dropped_by_target = filtered_by_target
        logging.info(
            "[defence:%s] Drop cap applied: kept %d/%d candidates (max_drop=%s, max_drop_ratio=%s)",
            defence_label,
            len(drop_set),
            len(candidates),
            str(max_drop) if max_drop is not None else "none",
            f"{max_drop_ratio:.3f}" if max_drop_ratio is not None else "none",
        )

    kept_users = [u for u in compromised_users if u not in drop_set]
    logging.info(
        "[defence:%s] Activation clustering dropped %d/%d users (min_samples=%d, small_ratio=%.3f)",
        defence_label,
        len(drop_set),
        len(compromised_users),
        min_samples,
        small_ratio,
    )

    filtered_sequences = [
        line for line in combined_sequences if line.split()[0] not in drop_set
    ]
    filtered_exp_splits = {}
    for split_name, records in exp_splits.items():
        if isinstance(records, list):
            filtered_exp_splits[split_name] = [
                copy.deepcopy(entry)
                for entry in records
                if str(entry.get("reviewerID")) not in drop_set
            ]
        else:
            filtered_exp_splits[split_name] = copy.deepcopy(records)

    filtered_user_id2idx = {k: v for k, v in user_id2idx.items() if str(k) not in drop_set}
    filtered_user_id2name = {k: v for k, v in user_id2name.items() if str(k) not in drop_set}
    filtered_keywords = {k: v for k, v in keywords_map.items() if k not in drop_set}
    filtered_embedding = {k: v for k, v in embedding_deltas.items() if k not in drop_set}
    filtered_round_metrics = {k: v for k, v in round_metrics.items() if k not in drop_set}
    filtered_shadow_meta = {k: v for k, v in shadow_meta.items() if k not in drop_set}

    filtered_summary = copy.deepcopy(summary)
    filtered_summary["compromised_user_count"] = len(kept_users)
    filtered_summary["compromised_users"] = kept_users
    effective_mr = float(len(kept_users) / base_count_for_mr) if base_count_for_mr else 0.0
    filtered_summary.setdefault("config", {})["effective_mr_defence"] = effective_mr
    filtered_summary["defence"] = {
        "name": defence_label,
        "min_samples": min_samples,
        "small_ratio": small_ratio,
        "max_drop": max_drop,
        "max_drop_ratio": max_drop_ratio,
        "embedding_source": embedding_source,
        "dropped": list(drop_set),
        "dropped_by_target": {k: len(v) for k, v in dropped_by_target.items()},
    }

    meta_payload = {
        "attack_mode": f"{attack_variant}+{defence_label}",
        "count": len(kept_users),
        "defence": filtered_summary["defence"],
    }

    return {
        "combined_sequences": filtered_sequences,
        "exp_splits": filtered_exp_splits,
        "user_id2idx": filtered_user_id2idx,
        "user_id2name": filtered_user_id2name,
        "keywords_map": filtered_keywords,
        "embedding_deltas": filtered_embedding,
        "round_metrics": filtered_round_metrics,
        "summary": filtered_summary,
        "shadow_meta": filtered_shadow_meta,
        "shadow_meta_meta": meta_payload,
        "kept_users": kept_users,
        "dropped_users": list(drop_set),
        "metrics": {"dropped_count": len(drop_set)},
    }


def _apply_spectral_signature_defence(
    defence_label: str,
    adapter: Any,
    top_ratio: float,
    min_samples: int,
    *,
    combined_sequences: list[str],
    exp_splits: dict[str, Any],
    user_id2idx: dict[str, Any],
    user_id2name: dict[str, Any],
    keywords_map: dict[str, Any],
    embedding_deltas: dict[str, Any],
    round_metrics: dict[str, Any],
    summary: dict[str, Any],
    shadow_meta: dict[str, Any],
    compromised_users: list[str],
    attack_variant: str,
    base_count_for_mr: int,
) -> Dict[str, Any] | None:
    entry_map = _collect_poison_entries(exp_splits, compromised_users)
    if not entry_map:
        logging.warning("[defence:%s] No poisoned entries found for spectral signatures.", defence_label)
        return None

    ordered_users: List[str] = []
    vectors: List[np.ndarray] = []
    for user_id in sorted(entry_map.keys()):
        entry = entry_map[user_id]
        vec = _build_defence_embedding(
            adapter,
            entry.get("feature", []),
            entry.get("reviewText", ""),
            text_max_len=DEFENCE_TEXT_MAX_LEN,
            text_hash_dim=DEFENCE_TEXT_HASH_DIM,
            w_img=DEFENCE_EMBED_IMG_WEIGHT,
            w_txt=DEFENCE_EMBED_TXT_WEIGHT,
        )
        if vec.size > 0:
            ordered_users.append(user_id)
            vectors.append(vec)

    if not vectors:
        logging.warning("[defence:%s] No usable embeddings for spectral signatures.", defence_label)
        return None

    top_ratio = max(0.0, min(0.5, float(top_ratio)))
    min_samples = max(1, int(min_samples))
    drop_set: set[str] = set()

    mat = np.stack(vectors, axis=0)
    if mat.shape[0] >= min_samples and top_ratio > 0.0:
        mat = _standardize_matrix(mat)
        try:
            _u, _s, vt = np.linalg.svd(mat, full_matrices=False)
            v1 = vt[0]
            scores = np.abs(mat @ v1)
            drop_k = max(1, int(math.ceil(len(scores) * top_ratio)))
            drop_idx = np.argsort(-scores)[:drop_k]
            drop_set = {ordered_users[i] for i in drop_idx}
        except Exception as exc:
            logging.warning("[defence:%s] Spectral SVD failed: %s", defence_label, exc)
            drop_set = set()

    kept_users = [u for u in compromised_users if u not in drop_set]
    logging.info(
        "[defence:%s] Spectral signatures dropped %d/%d users (top_ratio=%.3f, min_samples=%d)",
        defence_label,
        len(drop_set),
        len(compromised_users),
        top_ratio,
        min_samples,
    )

    filtered_sequences = [
        line for line in combined_sequences if line.split()[0] not in drop_set
    ]
    filtered_exp_splits = {}
    for split_name, records in exp_splits.items():
        if isinstance(records, list):
            filtered_exp_splits[split_name] = [
                copy.deepcopy(entry)
                for entry in records
                if str(entry.get("reviewerID")) not in drop_set
            ]
        else:
            filtered_exp_splits[split_name] = copy.deepcopy(records)

    filtered_user_id2idx = {k: v for k, v in user_id2idx.items() if str(k) not in drop_set}
    filtered_user_id2name = {k: v for k, v in user_id2name.items() if str(k) not in drop_set}
    filtered_keywords = {k: v for k, v in keywords_map.items() if k not in drop_set}
    filtered_embedding = {k: v for k, v in embedding_deltas.items() if k not in drop_set}
    filtered_round_metrics = {k: v for k, v in round_metrics.items() if k not in drop_set}
    filtered_shadow_meta = {k: v for k, v in shadow_meta.items() if k not in drop_set}

    filtered_summary = copy.deepcopy(summary)
    filtered_summary["compromised_user_count"] = len(kept_users)
    filtered_summary["compromised_users"] = kept_users
    effective_mr = float(len(kept_users) / base_count_for_mr) if base_count_for_mr else 0.0
    filtered_summary.setdefault("config", {})["effective_mr_defence"] = effective_mr
    filtered_summary["defence"] = {
        "name": defence_label,
        "top_ratio": top_ratio,
        "min_samples": min_samples,
        "dropped": list(drop_set),
    }

    meta_payload = {
        "attack_mode": f"{attack_variant}+{defence_label}",
        "count": len(kept_users),
        "defence": filtered_summary["defence"],
    }

    return {
        "combined_sequences": filtered_sequences,
        "exp_splits": filtered_exp_splits,
        "user_id2idx": filtered_user_id2idx,
        "user_id2name": filtered_user_id2name,
        "keywords_map": filtered_keywords,
        "embedding_deltas": filtered_embedding,
        "round_metrics": filtered_round_metrics,
        "summary": filtered_summary,
        "shadow_meta": filtered_shadow_meta,
        "shadow_meta_meta": meta_payload,
        "kept_users": kept_users,
        "dropped_users": list(drop_set),
        "metrics": {"dropped_count": len(drop_set)},
    }



def _append_text_pair(
    dataset: str,
    poison_subdir: str,
    target: str,
    original_text: str,
    adversarial_text: str,
    *,
    root: str | None = None,
    feature: Iterable[float] | None = None,
) -> str:
    """Append a single (original, adversarial) text pair for downstream ROUGE evaluation."""

    base_root = root or TEXT_PAIR_ROOT
    pair_dir = os.path.join(base_root, dataset, "poisoned", poison_subdir)
    os.makedirs(pair_dir, exist_ok=True)
    pair_path = os.path.join(pair_dir, "text_pairs.jsonl")
    if append_mode_registry.get(pair_path) is None:
        append_mode_registry[pair_path] = True
        try:
            os.remove(pair_path)
        except FileNotFoundError:
            pass
    record = {
        "target": target,
        "original_text": original_text,
        "adversarial_text": adversarial_text,
    }
    if feature is not None:
        record["feature"] = [float(x) for x in feature]
    pair_path = os.path.join(pair_dir, "text_pairs.jsonl")
    with open(pair_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
    return pair_path


def _mode4_quality_score(round_entry: "PerturbationRound | None") -> float:
    """Derive a simple quality score for Mode-4 single-step samples."""

    if round_entry is None:
        return 0.0
    hit_prob = float(getattr(round_entry, "hit_prob", 0.0) or 0.0)
    info_nce = float(getattr(round_entry, "info_nce", 0.0) or 0.0)
    return 0.7 * hit_prob + 0.3 * info_nce


def _is_ablation_mode1(
    *,
    interaction_rounds: int,
    mask_vis_ratio: float,
    mask_txt_ratio: float,
    img_eps: float,
    text_ratio_max: float,
    text_embed_eps: float,
    min_txt_replacements: int,
) -> bool:
    def _is_zero(value: float) -> bool:
        return abs(float(value)) <= 1e-9

    return (
        interaction_rounds <= 0
        and _is_zero(mask_vis_ratio)
        and _is_zero(mask_txt_ratio)
        and _is_zero(img_eps)
        and _is_zero(text_ratio_max)
        and _is_zero(text_embed_eps)
        and min_txt_replacements <= 0
    )


def _is_ablation_mode2(
    *,
    interaction_rounds: int,
    mask_vis_ratio: float,
    mask_txt_ratio: float,
    img_eps: float,
    text_ratio_max: float,
    text_embed_eps: float,
    min_txt_replacements: int,
) -> bool:
    def _is_zero(value: float) -> bool:
        return abs(float(value)) <= 1e-9

    return (
        interaction_rounds <= 0
        and img_eps > 0
        and _is_zero(mask_txt_ratio)
        and _is_zero(text_ratio_max)
        and _is_zero(text_embed_eps)
        and min_txt_replacements <= 0
    )


def _is_ablation_mode3(
    *,
    interaction_rounds: int,
    mask_txt_ratio: float,
    img_eps: float,
    text_ratio_max: float,
    text_embed_eps: float,
    min_txt_replacements: int,
) -> bool:
    def _is_zero(value: float) -> bool:
        return abs(float(value)) <= 1e-9

    text_enabled = (
        mask_txt_ratio > 0
        or text_ratio_max > 0
        or text_embed_eps > 0
        or min_txt_replacements > 0
    )

    return _is_zero(img_eps) and text_enabled


def _is_ablation_mode4(
    *,
    interaction_rounds: int,
    mask_txt_ratio: float,
    img_eps: float,
    text_ratio_max: float,
    text_embed_eps: float,
    min_txt_replacements: int,
) -> bool:
    """History + image + text with no interaction rounds."""

    text_enabled = (
        mask_txt_ratio > 0
        or text_ratio_max > 0
        or text_embed_eps > 0
        or min_txt_replacements > 0
    )

    return interaction_rounds <= 0 and img_eps > 0 and text_enabled


class SimpleTokenizer:
    """Whitespace tokenizer used as a lightweight fallback."""

    def encode(self, text: str) -> List[str]:
        tokens = [tok for tok in str(text).split() if tok]
        return tokens if tokens else [str(text)]

    def decode(self, tokens: Iterable[Any]) -> str:
        return " ".join(str(tok) for tok in tokens)


def _load_pickle(path: str, default: Any) -> Any:
    if not os.path.isfile(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _build_asin_map(datamap: Dict[str, Any]) -> Dict[str, str]:
    asin2id: Dict[str, str] = {}
    item2id = datamap.get("item2id", {}) or {}
    for asin, idx in item2id.items():
        asin2id[str(asin).upper()] = str(idx)
    # Allow reverse lookup (id->id)
    for idx in item2id.values():
        asin2id[str(idx).upper()] = str(idx)
    return asin2id


def _map_item_identifier(item: Any, asin2id: Dict[str, str]) -> str:
    if item is None:
        return "0"
    text = str(item)
    if text.isdigit():
        return text
    mapped = asin2id.get(text.upper())
    return mapped if mapped is not None else text


def _load_popular_item_list(dataset: str, top_k: int) -> List[str]:
    rel_path = os.path.join(
        PROJ_ROOT,
        "analysis",
        "results",
        dataset,
        f"high_pop_items_{dataset}_highcount_100.txt",
    )
    if not os.path.isfile(rel_path):
        return []
    items: List[str] = []
    with open(rel_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Item:" not in line:
                continue
            match = re.search(r"Item:\s*([A-Za-z0-9]+)", line)
            if match:
                items.append(match.group(1))
            if len(items) >= top_k:
                break
    return items


def _build_compromised_sequence(
    history: List[Any],
    target: Any,
    length: int,
    target_slot_from_end: int = 3,
) -> List[str]:
    """Return a shuffled history where the target sits at the train slot (``-3``).

    VIP5 的顺序任务会在 train 阶段把倒数第 3 个元素当作预测目标。为了保证
    策反样本只在训练中生效，我们把 ``target`` 精确插入到该位置，同时保持序列总长。
    验证/测试会直接跳过这些用户，因此无需额外处理其它位置。
    """
    target_str = str(target)
    length = max(3, int(length))

    base_pool = [str(it) for it in history if str(it) != target_str]
    if not base_pool:
        base_pool = [target_str]

    random.shuffle(base_pool)

    needed = length - 1  # slots reserved for non-target interactions
    support = base_pool[:needed]
    while len(support) < needed:
        support.append(random.choice(base_pool))

    slot = max(1, int(target_slot_from_end))
    insert_idx = max(0, length - slot)
    support.insert(insert_idx, target_str)
    sequence = support[:length]

    if sequence.count(target_str) > 1:
        sequence = [item for item in sequence if item != target_str]
        while len(sequence) < length - 1:
            sequence.append(random.choice(base_pool))
        sequence.insert(insert_idx, target_str)

    return sequence[:length]


def _place_target_in_sequence(support: List[Any], target: Any, length: int) -> List[Any]:
    """Scatter support items while fixing ``target`` at ``length-3``.

    The caller guarantees ``support`` already contains the non-target interactions
    we wish to replay for the fake user. We randomise their order (allowing
    repeats when the pool is small) and insert the true target three positions
    from the tail so downstream training (which ignores the last two items) can
    still observe it.
    """

    length = max(3, int(length))
    # Drop accidental target duplicates to keep the special slot unique, but keep
    # a fallback copy when the pool is otherwise empty.
    filtered = [item for item in support if str(item) != str(target)]
    if not filtered:
        filtered = list(support) if support else [target]

    # Ensure we have enough items to populate every non-target position.
    while len(filtered) < length - 1:
        filtered.append(random.choice(filtered))

    random.shuffle(filtered)
    others = iter(filtered[: length - 1])
    target_index = length - 3

    sequence: List[Any] = []
    for idx in range(length):
        if idx == target_index:
            sequence.append(target)
        else:
            sequence.append(next(others))
    return sequence


def _build_support_sequence(neighbors: List[Any], target: Any, length: int) -> List[Any]:
    """Create a support sequence ending with the target item.

    The behaviour can be controlled via environment variable
    ``VIP5_NEIGHBOR_STRATEGY``，便于在不同的伪装策略之间切换：

    ``original_neighbors`` (默认)
        原始实现：按邻居列表顺序截断并在需要时用同一 pool 补足。
    ``popular_mimic_v1``
        新的热门伪装方案 1：截取前 ``VIP5_NEIGHBOR_POOL`` 个（默认 30），
        随机抽取 ``length-1`` 个伪装商品，必要时允许重复。
    其它取值会自动回退到 ``original_neighbors``。
    """

    strategy = os.environ.get("VIP5_NEIGHBOR_STRATEGY", "original_neighbors").lower()
    length = max(3, length)

    if strategy == "popular_mimic_v1":
        pool_cap = int(os.environ.get("VIP5_NEIGHBOR_POOL", "30"))
        base_pool = list(neighbors) if neighbors else []
        if pool_cap > 0:
            base_pool = base_pool[:pool_cap]
        if not base_pool:
            base_pool = [target]

        sample_size = min(len(base_pool), max(0, length - 1))
        support = random.sample(base_pool, sample_size)
        while len(support) < length - 1:
            support.append(random.choice(base_pool))
        return _place_target_in_sequence(support, target, length)

    if strategy == "popular_mimic_v2":
        pool_cap = int(os.environ.get("VIP5_NEIGHBOR_POOL", "30"))
        base_pool = GLOBAL_POPULAR_ITEMS[:pool_cap] if GLOBAL_POPULAR_ITEMS else []
        if not base_pool:
            base_pool = list(neighbors) if neighbors else []
        base_pool = [item for item in base_pool if str(item) != str(target)] or [target]
        sample_size = min(len(base_pool), max(0, length - 1))
        support = random.sample(base_pool, sample_size)
        while len(support) < length - 1:
            support.append(random.choice(base_pool))
        return _place_target_in_sequence(support, target, length)

    if strategy == "medium_pop_mimic_v1":
        pool_cap = int(os.environ.get("VIP5_NEIGHBOR_POOL", "30"))
        base_pool = GLOBAL_MID_POP_ITEMS[:pool_cap] if GLOBAL_MID_POP_ITEMS else []
        if not base_pool:
            base_pool = list(neighbors) if neighbors else []
        base_pool = [item for item in base_pool if str(item) != str(target)] or [target]
        sample_size = min(len(base_pool), max(0, length - 1))
        support = random.sample(base_pool, sample_size)
        while len(support) < length - 1:
            support.append(random.choice(base_pool))
        return _place_target_in_sequence(support, target, length)

    # original 行为：沿用原始 neighbours 顺序，缺口用同一 pool 补齐
    support = list(neighbors) if neighbors else []
    support = support[: max(0, length - 1)]
    filler_pool = neighbors if neighbors else [target]
    while len(support) < length - 1:
        support.append(random.choice(filler_pool))
    return _place_target_in_sequence(support, target, length)


def _compute_summary(sequences: List[List[str]], comment_lengths: List[int]) -> Dict[str, Any]:
    if not sequences:
        return {
            "num_sequences": 0,
            "length_stats": {"min": 0, "max": 0, "mean": 0.0},
            "target_count_stats": {"min": 0, "max": 0, "mean": 0.0},
            "comment_len_stats": {"min": 0, "max": 0, "mean": 0.0},
            "teacher_presence_ratio": 0.0,
            "teacher_present_mean": 0.0,
        }
    lengths = [len(seq) for seq in sequences]
    length_min = min(lengths)
    length_max = max(lengths)
    length_mean = sum(lengths) / len(lengths)
    target_counts = [seq.count(seq[-1]) for seq in sequences]
    comment_min = min(comment_lengths) if comment_lengths else 0
    comment_max = max(comment_lengths) if comment_lengths else 0
    comment_mean = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0.0
    return {
        "num_sequences": len(sequences),
        "length_stats": {"min": float(length_min), "max": float(length_max), "mean": float(length_mean)},
        "target_count_stats": {
            "min": float(min(target_counts)),
            "max": float(max(target_counts)),
            "mean": float(sum(target_counts) / len(target_counts)),
        },
        "comment_len_stats": {
            "min": float(comment_min),
            "max": float(comment_max),
            "mean": float(comment_mean),
        },
        "teacher_presence_ratio": 0.0,
        "teacher_present_mean": 0.0,
    }


def _next_user_index(user_id2idx: Dict[Any, Any]) -> int:
    if not isinstance(user_id2idx, dict) or not user_id2idx:
        return 0
    max_idx = -1
    for value in user_id2idx.values():
        try:
            idx = int(value)  # handles numeric strings as well
        except (TypeError, ValueError):
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1



def _run_shadowcast_pipeline(args: Any) -> Dict[str, Any]:
    """Shadowcast baseline: inject clean-label image/text pairs without altering real histories."""

    logging.info("[shadowcast] launching shadowcast poisoning branch")

    dataset = getattr(args, "dataset", "unknown")
    data_root = getattr(args, "data_root", os.path.join(PROJ_ROOT, "data"))
    cache_dir = getattr(args, "cache_dir", os.path.join(os.path.dirname(__file__), "caches"))
    mr = float(getattr(args, "mr", 0.0))
    interaction_rounds = max(0, int(getattr(args, "interaction_rounds", 4)))
    mask_vis_ratio = float(getattr(args, "mask_vis_ratio", 0.15))
    mask_txt_ratio = float(getattr(args, "mask_txt_ratio", 0.18))
    img_eps = float(getattr(args, "img_eps", 0.05))
    text_ratio_max = float(getattr(args, "txt_ratio_max", 0.2))
    min_txt_replacements = int(getattr(args, "min_txt_replacements", 2))
    sim_threshold = float(getattr(args, "sim_threshold", 0.92))
    text_embed_eps = float(getattr(args, "txt_embed_eps", 0.0))
    seed = getattr(args, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    attack_label = "dcip_ieos_shadowcast"
    split_dir = os.path.join(data_root, dataset)
    poison_root = os.path.join(split_dir, "poisoned")
    os.makedirs(poison_root, exist_ok=True)

    poison_subdir = _derive_poison_subdir(
        args,
        mr=mr,
        interaction_rounds=interaction_rounds,
        img_eps=img_eps,
        text_ratio_max=text_ratio_max,
        ablation_tag=None,
        base_label=attack_label,
    )
    text_pair_root = getattr(args, "text_pair_root", None)
    poison_dir = os.path.join(poison_root, poison_subdir)
    os.makedirs(poison_dir, exist_ok=True)
    _copy_support_files(split_dir, poison_dir)
    logging.info("[shadowcast] artefacts will be written to %s", poison_dir)

    datamap_path = os.path.join(split_dir, "datamaps.json")
    datamap = _load_pickle(datamap_path, {}) if datamap_path.endswith(".pkl") else (
        json.load(open(datamap_path, "r", encoding="utf-8")) if os.path.isfile(datamap_path) else {}
    )
    asin2id = _build_asin_map(datamap)

    comp_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")
    if not os.path.isfile(comp_path):
        raise FileNotFoundError(f"Competition pool not found: {comp_path}")
    with open(comp_path, "r", encoding="utf-8") as f:
        competition_pool: List[Dict[str, Any]] = json.load(f)

    max_targets = getattr(args, "max_targets", None)
    if isinstance(max_targets, int) and max_targets > 0:
        original_size = len(competition_pool)
        competition_pool = competition_pool[:max_targets]
        logging.info(
            "[shadowcast] limiting to first %d of %d targets for this run",
            len(competition_pool),
            original_size,
        )

    from .prototypes import build_or_load_pop_center  # local import to avoid cycles

    try:
        c_pop_vec = build_or_load_pop_center(
            split_dir=split_dir,
            cache_dir=cache_dir,
            feat_root=getattr(args, "feat_root", "features"),
            feat_backbone=getattr(args, "feat_backbone", "vitb32_features"),
            pop_path=getattr(args, "pop_path", None),
            fallback_anchors=None,
        )
    except Exception:
        c_pop_vec = None

    adapter = getattr(args, "victim", None)
    if adapter is None or not isinstance(adapter, VictimAdapter):
        victim_ckpt = getattr(args, "victim_ckpt", None)
        if victim_ckpt and not os.path.isfile(victim_ckpt):
            logging.warning("[shadowcast] victim checkpoint not found: %s", victim_ckpt)
            victim_ckpt = None
        img_feat_type = getattr(args, "image_feature_type", "vitb32") or "vitb32"
        device_arg = getattr(args, "victim_device", "cpu") or "cpu"
        adapter = VictimAdapter(
            model=getattr(args, "victim_model", None),
            tokenizer=getattr(args, "victim_tokenizer", None),
            device=str(device_arg),
            ckpt_path=victim_ckpt,
            dataset=dataset,
            image_feature_type=img_feat_type,
            backbone=(getattr(args, "victim_backbone", "t5-base") or "t5-base"),
        )
    if getattr(adapter, "tokenizer", None) is None:
        adapter.tokenizer = SimpleTokenizer()

    seq_path = os.path.join(split_dir, "sequential_data.txt")
    if not os.path.isfile(seq_path):
        raise RuntimeError(f"Sequential data file missing: {seq_path}")
    with open(seq_path, "r", encoding="utf-8") as f:
        original_count = sum(1 for _ in f if _.strip())
    desired_pairs = int(math.ceil(original_count * mr)) if mr > 0 and original_count > 0 else 0
    if desired_pairs and not competition_pool:
        raise RuntimeError("Competition pool is empty; cannot generate shadowcast samples")
    if desired_pairs and desired_pairs > len(competition_pool):
        logging.warning(
            "[shadowcast] requested %d pairs but only %d targets available; recycling targets.",
            desired_pairs,
            len(competition_pool),
        )

    suffix = f"_{attack_label}_mr{mr}" if mr else f"_{attack_label}_mr0"
    legacy_files = [
        os.path.join(poison_dir, f"sequential_data{suffix}.txt"),
        os.path.join(poison_dir, f"exp_splits{suffix}.pkl"),
        os.path.join(poison_dir, f"user_id2idx{suffix}.pkl"),
        os.path.join(poison_dir, f"user_id2name{suffix}.pkl"),
    ]
    for legacy in legacy_files:
        if os.path.isfile(legacy):
            try:
                os.remove(legacy)
            except OSError:
                pass

    round_metrics: Dict[str, List[Dict[str, Any]]] = {}
    embedding_deltas: Dict[str, Any] = {}
    keywords_map: Dict[str, Dict[str, Any]] = {}
    skip_counts: Dict[str, int] = {}
    text_pairs_path: str | None = None
    poison_pair_ids: List[str] = []

    selected_entries: List[Dict[str, Any]] = []
    if desired_pairs > 0:
        pool = competition_pool[:]
        while len(selected_entries) < desired_pairs:
            random.shuffle(pool)
            selected_entries.extend(pool)
        selected_entries = selected_entries[:desired_pairs]

    for entry in selected_entries:
        target = entry.get("target")
        anchor_vec = entry.get("anchor", [])
        if target is None or not anchor_vec:
            skip_counts["missing_anchor"] = skip_counts.get("missing_anchor", 0) + 1
            continue

        target = str(target)
        anchor = list(anchor_vec)
        pop_vec = list(c_pop_vec) if c_pop_vec else list(anchor)
        keywords = entry.get("keywords", []) or []
        base_text = " ".join(str(k) for k in keywords if k) or target

        result = single_step_perturb(
            adapter,
            anchor,
            base_text,
            pop_vec,
            keywords=[str(k) for k in keywords if k],
            vis_ratio=mask_vis_ratio,
            txt_ratio=mask_txt_ratio,
            img_eps=img_eps,
            text_replace_ratio=text_ratio_max,
            min_txt_replacements=min_txt_replacements,
            text_embed_eps=text_embed_eps,
            independent_text=True,
        )

        pair_id = f"shadowcast_pair_{len(poison_pair_ids):06d}"
        poison_pair_ids.append(pair_id)
        text_pairs_path = _append_text_pair(
            dataset,
            poison_subdir,
            target,
            base_text,
            result.final_text,
            root=text_pair_root,
            feature=result.final_image,
        )

        round_metrics[pair_id] = [
            {
                "round": r.round_idx,
                "sim_after_image": float(r.sim_after_image),
                "sim_target": float(r.sim_target),
                "dist_target": float(r.dist_target),
                "img_eps_used": float(r.img_eps_used),
                "txt_ratio": float(r.txt_ratio),
                "img_budget": float(r.img_budget),
                "txt_budget": float(r.txt_budget),
                "info_nce": float(r.info_nce),
                "psnr_img": float(r.psnr_img),
                "bce_loss": float(r.bce_loss),
                "hit_prob": float(r.hit_prob),
                "txt_embed_delta_norm": float(r.txt_embed_delta_norm) if r.txt_embed_delta_norm is not None else None,
                "stop_reason": r.stop_reason,
            }
            for r in result.rounds
        ]
        embedding_deltas[pair_id] = {
            "final_sim": float(result.rounds[-1].sim_target) if result.rounds else 0.0,
            "rounds": len(result.rounds),
        }
        keywords_map[target] = {
            "tokens": [str(k) for k in keywords if k],
            "synthetic": bool(entry.get("synthetic", False)),
        }

    summary = {
        "num_sequences": 0,
        "length_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
        "target_count_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
        "comment_len_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
        "teacher_presence_ratio": 0.0,
        "teacher_present_mean": 0.0,
    }
    effective_mr = float(len(poison_pair_ids) / original_count) if original_count else 0.0
    summary["config"] = {
        "mr": mr,
        "interaction_rounds": interaction_rounds,
        "mask_vis_ratio": mask_vis_ratio,
        "mask_txt_ratio": mask_txt_ratio,
        "img_eps": img_eps,
        "text_replace_ratio": text_ratio_max,
        "min_txt_replacements": min_txt_replacements,
        "sim_threshold": sim_threshold,
        "text_embed_eps": text_embed_eps,
        "neighbor_strategy": os.environ.get("VIP5_NEIGHBOR_STRATEGY", "original_neighbors"),
        "neighbor_pool_cap": os.environ.get("VIP5_NEIGHBOR_POOL", ""),
        "effective_mr": effective_mr,
        "poison_subdir": poison_subdir,
        "attack_label": attack_label,
        "attack_variant": "shadowcast",
        "cross_mode": "decoupled",
    }
    summary["skip_reasons"] = skip_counts
    summary["poison_pair_count"] = len(poison_pair_ids)
    summary["poison_user_count"] = 0
    summary["poison_user_ids"] = []
    summary["text_pairs_path"] = text_pairs_path

    kw_out = os.path.join(poison_dir, f"keywords{suffix}.pkl")
    delta_out = os.path.join(poison_dir, f"embedding_deltas{suffix}.pkl")
    metrics_out = os.path.join(poison_dir, f"round_metrics{suffix}.pkl")
    summary_out = os.path.join(poison_dir, f"poison_summary{suffix}.json")
    shadow_meta_out = os.path.join(poison_dir, f"shadow_meta{suffix}.json")

    _dump_pickle(kw_out, keywords_map)
    _dump_pickle(delta_out, embedding_deltas)
    _dump_pickle(metrics_out, round_metrics)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    shadow_meta_payload = {
        "pairs": poison_pair_ids,
        "meta": {"attack_mode": "shadowcast", "poison_pair_count": len(poison_pair_ids)},
    }
    with open(shadow_meta_out, "w", encoding="utf-8") as f:
        json.dump(shadow_meta_payload, f, ensure_ascii=False, indent=2)

    logging.info(
        "[shadowcast] Generated %d poison pairs (requested mr=%.3f, effective mr=%.3f) for dataset '%s'",
        len(poison_pair_ids),
        mr,
        effective_mr,
        dataset,
    )

    clean_seq = os.path.join(split_dir, "sequential_data.txt")
    clean_exp = os.path.join(split_dir, "exp_splits.pkl")
    clean_idx = os.path.join(split_dir, "user_id2idx.pkl")
    clean_name = os.path.join(split_dir, "user_id2name.pkl")

    return {
        "sequential_path": clean_seq,
        "exp_splits_path": clean_exp,
        "user_id2idx_path": clean_idx,
        "user_id2name_path": clean_name,
        "keywords_path": kw_out,
        "embedding_deltas_path": delta_out,
        "round_metrics_path": metrics_out,
        "summary_path": summary_out,
        "shadow_meta_path": shadow_meta_out,
        "poison_dir": poison_dir,
        "poison_subdir": poison_subdir,
        "text_pairs_path": text_pairs_path,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: Any) -> Dict[str, Any]:
    """Execute the simplified poisoning pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    attack_variant = str(getattr(args, "attack_variant", "byzantine") or "byzantine").lower()
    if attack_variant not in {"byzantine", "shadowcast", "direct_boost", "random_attack", "popular_attack"}:
        raise ValueError(
            f"Unsupported attack_variant '{attack_variant}'. "
            "Expected 'byzantine', 'shadowcast', 'direct_boost', or 'random_attack'."
        )
    if attack_variant == "shadowcast":
        return _run_shadowcast_pipeline(args)
    is_direct_boost = attack_variant == "direct_boost"
    is_random_attack = attack_variant == "random_attack"
    is_popular_attack = attack_variant == "popular_attack"

    dataset = getattr(args, "dataset", "unknown")
    data_root = getattr(args, "data_root", os.path.join(PROJ_ROOT, "data"))
    cache_dir = getattr(args, "cache_dir", os.path.join(os.path.dirname(__file__), "caches"))
    mr = float(getattr(args, "mr", 0.0))
    sequence_length = max(2, int(getattr(args, "sequence_length", 10)))
    interaction_rounds = max(0, int(getattr(args, "interaction_rounds", 4)))
    mask_vis_ratio = float(getattr(args, "mask_vis_ratio", 0.15))
    mask_txt_ratio = float(getattr(args, "mask_txt_ratio", 0.15))
    img_eps = float(getattr(args, "img_eps", 0.05))
    # Image perturbation extended knobs (PGD support)
    img_iters = int(getattr(args, "img_iters", 3))
    psnr_min = float(getattr(args, "psnr_min", 30.0))
    img_strategy = str(getattr(args, "img_strategy", "cosine") or "cosine").lower()
    text_ratio_max = float(getattr(args, "txt_ratio_max", 0.2))
    min_txt_replacements = int(getattr(args, "min_txt_replacements", 2))
    sim_threshold = float(getattr(args, "sim_threshold", 0.92))
    text_embed_eps = float(getattr(args, "txt_embed_eps", 0.0))
    if is_direct_boost or is_random_attack or is_popular_attack:
        interaction_rounds = 0
        mask_vis_ratio = 0.0
        mask_txt_ratio = 0.0
        img_eps = 0.0
        text_ratio_max = 0.0
        min_txt_replacements = 0
        text_embed_eps = 0.0
    seed = getattr(args, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    split_dir = os.path.join(data_root, dataset)
    poison_root = os.path.join(split_dir, "poisoned")
    os.makedirs(poison_root, exist_ok=True)

    is_ablation_mode1 = _is_ablation_mode1(
        interaction_rounds=interaction_rounds,
        mask_vis_ratio=mask_vis_ratio,
        mask_txt_ratio=mask_txt_ratio,
        img_eps=img_eps,
        text_ratio_max=text_ratio_max,
        text_embed_eps=text_embed_eps,
        min_txt_replacements=min_txt_replacements,
    )
    is_ablation_mode2 = _is_ablation_mode2(
        interaction_rounds=interaction_rounds,
        mask_vis_ratio=mask_vis_ratio,
        mask_txt_ratio=mask_txt_ratio,
        img_eps=img_eps,
        text_ratio_max=text_ratio_max,
        text_embed_eps=text_embed_eps,
        min_txt_replacements=min_txt_replacements,
    )
    is_ablation_mode3 = _is_ablation_mode3(
        interaction_rounds=interaction_rounds,
        mask_txt_ratio=mask_txt_ratio,
        img_eps=img_eps,
        text_ratio_max=text_ratio_max,
        text_embed_eps=text_embed_eps,
        min_txt_replacements=min_txt_replacements,
    )
    is_ablation_mode4 = _is_ablation_mode4(
        interaction_rounds=interaction_rounds,
        mask_txt_ratio=mask_txt_ratio,
        img_eps=img_eps,
        text_ratio_max=text_ratio_max,
        text_embed_eps=text_embed_eps,
        min_txt_replacements=min_txt_replacements,
    )
    mode4_active = bool(
        is_ablation_mode4 and not is_direct_boost and not is_random_attack and not is_popular_attack
    )
    mode4_candidates: List[Dict[str, Any]] = []
    if mode4_active:
        scaled_img_eps = max(0.0, img_eps * MODE4_IMG_SCALE)
        scaled_txt_ratio_max = max(0.0, text_ratio_max * MODE4_TXT_SCALE)
        scaled_vis_ratio = min(mask_vis_ratio, MODE4_VIS_RATIO_CAP)
        scaled_txt_ratio = min(mask_txt_ratio, MODE4_TXT_RATIO_CAP)
    else:
        scaled_img_eps = img_eps
        scaled_txt_ratio_max = text_ratio_max
        scaled_vis_ratio = mask_vis_ratio
        scaled_txt_ratio = mask_txt_ratio
    ablation_tag = (
        "ablation_mode1"
        if is_ablation_mode1
        else (
            "ablation_mode2"
            if is_ablation_mode2
            else (
                "ablation_mode3"
                if is_ablation_mode3
                else ("ablation_mode4" if is_ablation_mode4 else None)
            )
        )
    )

    if is_direct_boost or is_random_attack:
        ablation_tag = None

    base_label = (
        "direct_boost"
        if is_direct_boost
        else ("random_attack" if is_random_attack else ("popular_attack" if is_popular_attack else "dcip_ieos_fc"))
    )
    img_eps_for_subdir = img_eps
    txt_ratio_for_subdir = text_ratio_max
    poison_subdir = _derive_poison_subdir(
        args,
        mr=mr,
        interaction_rounds=interaction_rounds,
        img_eps=img_eps_for_subdir,
        text_ratio_max=txt_ratio_for_subdir,
        ablation_tag=ablation_tag,
        base_label=base_label,
    )
    text_pair_root = getattr(args, "text_pair_root", None)
    poison_dir = os.path.join(poison_root, poison_subdir)
    os.makedirs(poison_dir, exist_ok=True)
    _copy_support_files(split_dir, poison_dir)
    logging.info("[poison-pipeline] artefacts will be written to %s", poison_dir)
    mode4_pair_path: str | None = None
    if mode4_active:
        base_root = text_pair_root or TEXT_PAIR_ROOT
        pair_dir = os.path.join(base_root, dataset, "poisoned", poison_subdir)
        os.makedirs(pair_dir, exist_ok=True)
        mode4_pair_path = os.path.join(pair_dir, "text_pairs.jsonl")

    if is_direct_boost:
        attack_label = "direct_boost"
    elif is_random_attack:
        attack_label = "random_attack"
    elif is_popular_attack:
        attack_label = "popular_attack"
    else:
        attack_label = (
            "dcip_ieos_fc_ablation_mode1"
            if is_ablation_mode1
            else (
                "dcip_ieos_fc_ablation_mode2"
                if is_ablation_mode2
                else (
                    "dcip_ieos_fc_ablation_mode3"
                    if is_ablation_mode3
                    else (
                        "dcip_ieos_fc_ablation_mode4" if is_ablation_mode4 else "dcip_ieos_fc"
                    )
                )
            )
        )

    datamap_path = os.path.join(split_dir, "datamaps.json")
    datamap = _load_pickle(datamap_path, {}) if datamap_path.endswith('.pkl') else (
        json.load(open(datamap_path, "r", encoding="utf-8")) if os.path.isfile(datamap_path) else {}
    )
    asin2id = _build_asin_map(datamap)

    comp_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")
    if not os.path.isfile(comp_path):
        raise FileNotFoundError(f"Competition pool not found: {comp_path}")
    with open(comp_path, "r", encoding="utf-8") as f:
        competition_pool: List[Dict[str, Any]] = json.load(f)

    max_targets = getattr(args, "max_targets", None)
    if isinstance(max_targets, int) and max_targets > 0:
        original_size = len(competition_pool)
        competition_pool = competition_pool[:max_targets]
        logging.info(
            "[poison-pipeline] limiting to first %d of %d targets for this run",
            len(competition_pool),
            original_size,
        )

    # Build global popularity lists to support alternative sampling modes.
    global GLOBAL_POPULAR_ITEMS, GLOBAL_MID_POP_ITEMS
    neighbor_counter: Counter[Any] = Counter()
    for entry in competition_pool:
        neighbor_counter.update(entry.get("neighbors", []) or [])
    sorted_neighbors = [item for item, _ in neighbor_counter.most_common()]
    top_cap = int(os.environ.get("VIP5_GLOBAL_POP_TOP", "100"))
    GLOBAL_POPULAR_ITEMS = sorted_neighbors[:top_cap]
    mid_start = int(os.environ.get("VIP5_MID_POP_START", "70"))
    mid_window = int(os.environ.get("VIP5_MID_POP_WINDOW", "30"))
    GLOBAL_MID_POP_ITEMS = sorted_neighbors[mid_start:mid_start + mid_window] if sorted_neighbors else []

    from .prototypes import build_or_load_pop_center  # local import to avoid cycles

    try:
        c_pop_vec = build_or_load_pop_center(
            split_dir=split_dir,
            cache_dir=cache_dir,
            feat_root=getattr(args, "feat_root", "features"),
            feat_backbone=getattr(args, "feat_backbone", "vitb32_features"),
            pop_path=getattr(args, "pop_path", None),
            fallback_anchors=None,
        )
    except Exception:
        c_pop_vec = None

    adapter = getattr(args, "victim", None)
    if adapter is None or not isinstance(adapter, VictimAdapter):
        victim_ckpt = getattr(args, "victim_ckpt", None)
        if victim_ckpt and not os.path.isfile(victim_ckpt):
            logging.warning("[poison-pipeline] victim checkpoint not found: %s", victim_ckpt)
            victim_ckpt = None
        img_feat_type = getattr(args, "image_feature_type", "vitb32") or "vitb32"
        device_arg = getattr(args, "victim_device", "cpu") or "cpu"
        adapter = VictimAdapter(
            model=getattr(args, "victim_model", None),
            tokenizer=getattr(args, "victim_tokenizer", None),
            device=str(device_arg),
            ckpt_path=victim_ckpt,
            dataset=dataset,
            image_feature_type=img_feat_type,
            backbone=(getattr(args, "victim_backbone", "t5-base") or "t5-base"),
        )
    if getattr(adapter, "tokenizer", None) is None:
        adapter.tokenizer = SimpleTokenizer()

    exp_path = os.path.join(split_dir, "exp_splits.pkl")
    user_idx_path = os.path.join(split_dir, "user_id2idx.pkl")
    user_name_path = os.path.join(split_dir, "user_id2name.pkl")
    seq_path = os.path.join(split_dir, "sequential_data.txt")

    exp_splits = _load_pickle(exp_path, {"train": [], "val": [], "test": []})
    user_id2idx = _load_pickle(user_idx_path, {})
    user_id2name = _load_pickle(user_name_path, {})
    if not user_id2idx:
        logging.warning(
            "[poison-pipeline] user_id2idx missing/empty at %s; rebuilding from %s",
            user_idx_path,
            seq_path,
        )
        if not os.path.isfile(seq_path):
            raise RuntimeError(f"Cannot rebuild user_id2idx: sequential data missing at {seq_path}")
        user_id2idx = {}
        with open(seq_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uid = line.split()[0]
                if uid not in user_id2idx:
                    user_id2idx[uid] = len(user_id2idx)
    if not user_id2name:
        logging.warning(
            "[poison-pipeline] user_id2name missing/empty at %s; synthesising from sequential data",
            user_name_path,
        )
        user_id2name = {uid: uid for uid in user_id2idx}

    sequences_for_stats: List[List[str]] = []
    comment_lengths: List[int] = []
    round_metrics: Dict[str, List[Dict[str, Any]]] = {}
    embedding_deltas: Dict[str, Any] = {}
    keywords_map: Dict[str, Dict[str, Any]] = {}
    skip_counts: Dict[str, int] = {}
    target_slot_from_end = 4 if (is_direct_boost or is_random_attack or is_popular_attack) else 3

    seq_path = os.path.join(split_dir, "sequential_data.txt")
    existing_users: List[str] = []
    original_sequences: List[str] = []
    if os.path.isfile(seq_path):
        with open(seq_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                original_sequences.append(line)
                existing_users.append(line.split()[0])

    shadow_seed_records: List[Tuple[str, List[str], str]] = []
    shadow_seed_path = getattr(args, "shadow_seed", None)
    if shadow_seed_path and os.path.exists(shadow_seed_path):
        try:
            with open(shadow_seed_path, "r", encoding="utf-8") as f:
                raw_seeds = json.load(f)
            if isinstance(raw_seeds, dict) and "seeds" in raw_seeds:
                raw_seeds = raw_seeds["seeds"]
            for idx, seed in enumerate(raw_seeds or []):
                history = seed.get("history", [])
                if not isinstance(history, list) or not history:
                    continue
                history = [str(x) for x in history]
                base_user = str(seed.get("user_id", f"seed_{idx}"))
                display_name = str(seed.get("display_name", base_user))
                shadow_seed_records.append((base_user, history, display_name))
            if shadow_seed_records:
                logging.info(
                    "[poison-pipeline] Loaded %d shadow seeds from %s",
                    len(shadow_seed_records),
                    shadow_seed_path,
                )
        except Exception as exc:
            logging.warning("[poison-pipeline] Failed to load shadow seed file %s: %s", shadow_seed_path, exc)
            shadow_seed_records = []

    if shadow_seed_records:
        original_records: List[Tuple[str, List[str]]] = [
            (base_user, history) for base_user, history, _ in shadow_seed_records
        ]
        item_to_sequences: defaultdict[str, List[int]] = defaultdict(list)
        for idx, (_, history, _) in enumerate(shadow_seed_records):
            for item in history:
                item_to_sequences[item].append(idx)
        user_name_lookup: Dict[str, str] = {user: display for user, _, display in shadow_seed_records}
    else:
        original_records = []
        item_to_sequences = defaultdict(list)
        for idx, seq in enumerate(original_sequences):
            parts = seq.split()
            if len(parts) <= 1:
                continue
            user = parts[0]
            items = parts[1:]
            original_records.append((user, items))
            for item in items:
                item_to_sequences[item].append(idx)
        if not original_records:
            raise RuntimeError("Unable to construct shadow users: sequential_data.txt has no usable entries (and no shadow seeds provided).")
        user_name_lookup = {}
        if user_id2name:
            for key, value in user_id2name.items():
                user_name_lookup[str(key)] = str(value)
    catalog_items: List[str] = sorted(
        {str(item) for _, items in original_records for item in items if str(item)}
    )
    if not catalog_items:
        catalog_items = sorted({str(v) for v in asin2id.values()}) or ["0"]
    popular_items = _load_popular_item_list(dataset, POPULAR_ATTACK_TOPK)
    if not popular_items:
        popular_items = catalog_items[:POPULAR_ATTACK_TOPK] or catalog_items
    available_indices = list(range(len(original_records)))
    random.shuffle(available_indices)
    unused_indices = set(available_indices)
    shadow_meta: Dict[str, Dict[str, Any]] = {}
    compromised_users: List[str] = []
    updated_sequences: Dict[int, List[str]] = {}

    original_count = len(original_records)
    desired_conversions = 0
    if mr > 0:
        base = original_count if original_count > 0 else len(competition_pool)
        desired_conversions = int(math.ceil(base * mr)) if base else 0

    if mr <= 0 or desired_conversions <= 0:
        selected_entries: List[Dict[str, Any]] = []
    else:
        if not competition_pool:
            raise RuntimeError("Competition pool is empty; cannot generate poisoned sequences")
        if desired_conversions <= len(competition_pool):
            selected_entries = random.sample(competition_pool, desired_conversions)
        else:
            selected_entries = []
            pool = competition_pool[:]
            random.shuffle(pool)
            while len(selected_entries) < desired_conversions:
                for entry in pool:
                    selected_entries.append(entry)
                    if len(selected_entries) >= desired_conversions:
                        break
                random.shuffle(pool)

    text_pairs_path: str | None = None
    for entry in selected_entries:
        if len(compromised_users) >= desired_conversions:
            break

        target = entry.get("target")
        anchor_vec = entry.get("anchor", [])
        if target is None or not anchor_vec:
            skip_counts["missing_anchor"] = skip_counts.get("missing_anchor", 0) + 1
            continue

        target = str(target)

        neighbor_set = {str(n) for n in entry.get("neighbors", []) or []}
        candidate_indices: List[int] = []
        for neighbor in neighbor_set:
            for idx in item_to_sequences.get(neighbor, []):
                if idx in unused_indices:
                    candidate_indices.append(idx)
        if candidate_indices:
            base_index = random.choice(candidate_indices)
        elif unused_indices:
            base_index = random.choice(tuple(unused_indices))
        else:
            skip_counts["insufficient_users"] = skip_counts.get("insufficient_users", 0) + 1
            break

        unused_indices.discard(base_index)
        base_user, base_items = original_records[base_index]
        base_user_str = str(base_user)
        base_name = user_name_lookup.get(base_user_str, f"user_{base_user_str}")

        history_items = [str(it) for it in base_items if str(it)]
        new_history = _build_compromised_sequence(
            history_items,
            target,
            sequence_length,
            target_slot_from_end=target_slot_from_end,
        )
        if is_random_attack:
            tail_len = max(0, RANDOM_ATTACK_TAIL_LEN)
            if tail_len > 0:
                tail_choices = random.choices(catalog_items, k=tail_len)
                new_history.extend(tail_choices)
        if is_popular_attack:
            tail_len = max(0, POPULAR_ATTACK_TAIL_LEN)
            if tail_len > 0:
                tail_choices = random.choices(popular_items, k=tail_len)
                new_history.extend(tail_choices)
        numeric_items = [_map_item_identifier(it, asin2id) for it in new_history]
        updated_sequences[base_index] = numeric_items
        compromised_users.append(base_user_str)

        anchor = list(anchor_vec)
        pop_vec = list(c_pop_vec) if c_pop_vec else list(anchor)
        keywords = entry.get("keywords", []) or []
        base_text = " ".join(str(k) for k in keywords if k) or target

        if mode4_active:
            result = single_step_perturb(
                adapter,
                anchor,
                base_text,
                pop_vec,
                keywords=[str(k) for k in keywords if k],
                vis_ratio=scaled_vis_ratio,
                txt_ratio=scaled_txt_ratio,
                img_eps=scaled_img_eps,
                text_replace_ratio=scaled_txt_ratio_max,
                min_txt_replacements=min_txt_replacements,
                text_embed_eps=text_embed_eps,
            )
        else:
            result = interactive_perturb_target(
                adapter,
                anchor,
                base_text,
                pop_vec,
                keywords=[str(k) for k in keywords if k],
                rounds=interaction_rounds,
                vis_ratio=scaled_vis_ratio,
                txt_ratio=scaled_txt_ratio,
                img_eps=scaled_img_eps,
                img_iters=img_iters,
                psnr_min=psnr_min,
                img_strategy=img_strategy,
                text_replace_ratio=scaled_txt_ratio_max,
                min_txt_replacements=min_txt_replacements,
                sim_threshold=sim_threshold,
                text_embed_eps=text_embed_eps,
                lock_vis_mask=getattr(args, "lock_vis_mask", False),
                vis_decay=getattr(args, "vis_decay", 1.0),
                stop_on_plateau=getattr(args, "stop_on_plateau", True),
            )

        sequences_for_stats.append(numeric_items)
        comment_lengths.append(len(result.final_text.split()))

        exp_entry = {
            "reviewerID": base_user_str,
            "reviewerName": base_name,
            "asin": target,
            "summary": "",
            "overall": 0.0,
            "helpful": [0, 0],
            "feature": result.final_image,
            "explanation": result.final_text,
            "reviewText": result.final_text,
        }
        exp_splits.setdefault("train", []).append(exp_entry)

        text_pairs_path = _append_text_pair(
            dataset,
            poison_subdir,
            target,
            base_text,
            result.final_text,
            root=text_pair_root,
            feature=result.final_image,
        )

        if user_id2name:
            name_sample = next(iter(user_id2name))
            if isinstance(name_sample, int) and base_user_str.isdigit():
                name_key = int(base_user_str)
            else:
                name_key = base_user_str
            user_id2name.setdefault(name_key, base_name)
        else:
            user_id2name = {base_user_str: base_name}

        round_metrics[base_user_str] = [
            {
                "round": r.round_idx,
                "sim_after_image": float(r.sim_after_image),
                "sim_target": float(r.sim_target),
                "dist_target": float(r.dist_target),
                "img_eps_used": float(r.img_eps_used),
                "txt_ratio": float(r.txt_ratio),
                "img_budget": float(r.img_budget),
                "txt_budget": float(r.txt_budget),
                "info_nce": float(r.info_nce),
                "psnr_img": float(r.psnr_img),
                "bce_loss": float(r.bce_loss),
                "hit_prob": float(r.hit_prob),
                "txt_embed_delta_norm": float(r.txt_embed_delta_norm) if r.txt_embed_delta_norm is not None else None,
                "stop_reason": r.stop_reason,
            }
            for r in result.rounds
        ]
        embedding_deltas[base_user_str] = {
            "final_sim": float(result.rounds[-1].sim_target) if result.rounds else 0.0,
            "rounds": len(result.rounds),
        }
        if mode4_active:
            final_round = result.rounds[-1] if result.rounds else None
            mode4_candidates.append(
                {
                    "user_id": base_user_str,
                    "base_index": base_index,
                    "quality": _mode4_quality_score(final_round),
                    "target": target,
                    "original_text": base_text,
                    "adversarial_text": result.final_text,
                }
            )

        keywords_map[target] = {
            "tokens": [str(k) for k in keywords if k],
            "synthetic": bool(entry.get("synthetic", False)),
        }

        shadow_meta[base_user_str] = {
            "base_user": base_user_str,
            "target": target,
            "history": new_history,
            "original_history": history_items,
            "display_name": base_name,
            "shadow_name": base_name,
            "type": "compromised",
        }

    for idx, item_list in updated_sequences.items():
        user_id = original_records[idx][0]
        user_id_str = str(user_id)
        line = " ".join([user_id_str] + [str(it) for it in item_list])
        if idx < len(original_sequences):
            original_sequences[idx] = line
        else:
            original_sequences.append(line)

    compromised_set = set(compromised_users)
    if compromised_set:
        exp_splits["val"] = [
            entry
            for entry in exp_splits.get("val", [])
            if str(entry.get("reviewerID")) not in compromised_set
        ]
        exp_splits["test"] = [
            entry
            for entry in exp_splits.get("test", [])
            if str(entry.get("reviewerID")) not in compromised_set
        ]

    filtered_mode4 = 0
    mode4_keep_ratio_applied = 1.0
    if mode4_active and mode4_candidates:
        keep_ratio = max(0.0, min(1.0, MODE4_KEEP_RATIO))
        mode4_keep_ratio_applied = keep_ratio
        if keep_ratio < 1.0:
            sorted_records = sorted(
                mode4_candidates, key=lambda rec: rec["quality"], reverse=True
            )
            keep_count = max(1, math.ceil(len(sorted_records) * keep_ratio))
            keep_users = {rec["user_id"] for rec in sorted_records[:keep_count]}
            drop_records = sorted_records[keep_count:]
            drop_users = {rec["user_id"] for rec in drop_records}
            filtered_mode4 = len(drop_users)
            if drop_users:
                drop_indices = {rec["base_index"] for rec in drop_records}
                for idx in drop_indices:
                    updated_sequences.pop(idx, None)
                compromised_users = [u for u in compromised_users if u in keep_users]
                shadow_meta = {
                    k: v for k, v in shadow_meta.items() if k not in drop_users
                }
                round_metrics = {
                    k: v for k, v in round_metrics.items() if k not in drop_users
                }
                embedding_deltas = {
                    k: v for k, v in embedding_deltas.items() if k not in drop_users
                }
                for split_name in ("train", "val", "test"):
                    entries = exp_splits.get(split_name)
                    if entries:
                        exp_splits[split_name] = [
                            entry
                            for entry in entries
                            if str(entry.get("reviewerID")) not in drop_users
                        ]
                if mode4_pair_path:
                    keep_records = [
                        rec for rec in mode4_candidates if rec["user_id"] in keep_users
                    ]
                    with open(mode4_pair_path, "w", encoding="utf-8") as f:
                        for rec in keep_records:
                            json.dump(
                                {
                                    "target": rec["target"],
                                    "original_text": rec["original_text"],
                                    "adversarial_text": rec["adversarial_text"],
                                },
                                f,
                                ensure_ascii=False,
                            )
                            f.write("\n")

    summary = _compute_summary(sequences_for_stats, comment_lengths)
    base_count_for_mr = original_count if original_count > 0 else max(1, len(shadow_seed_records))
    effective_mr = float(len(compromised_users) / base_count_for_mr) if base_count_for_mr else 0.0
    summary["config"] = {
        "mr": mr,
        "sequence_length": sequence_length,
        "interaction_rounds": interaction_rounds,
        "mask_vis_ratio": mask_vis_ratio,
        "mask_txt_ratio": mask_txt_ratio,
        "img_eps": img_eps,
        "text_replace_ratio": text_ratio_max,
        "min_txt_replacements": min_txt_replacements,
        "sim_threshold": sim_threshold,
        "text_embed_eps": text_embed_eps,
        "neighbor_strategy": os.environ.get("VIP5_NEIGHBOR_STRATEGY", "original_neighbors"),
        "neighbor_pool_cap": os.environ.get("VIP5_NEIGHBOR_POOL", ""),
        "effective_mr": effective_mr,
        "poison_subdir": poison_subdir,
        "attack_label": attack_label,
        "attack_variant": attack_variant,
        "cross_mode": getattr(args, "cross_mode", "interactive"),
        "ablation_mode": (
            "mode1"
            if is_ablation_mode1
            else (
                "mode2"
                if is_ablation_mode2
                else (
                    "mode3"
                    if is_ablation_mode3
                    else (
                        "image_text_single_step"
                        if mode4_active
                        else ("mode4" if is_ablation_mode4 else "none")
                    )
                )
            )
        ),
    }
    if mode4_active:
        summary["config"]["mode4_keep_ratio"] = float(mode4_keep_ratio_applied)
        summary["config"]["mode4_filtered"] = int(filtered_mode4)
        if filtered_mode4:
            summary["filtered_mode4_samples"] = filtered_mode4
    if is_random_attack:
        summary["config"]["random_tail_len"] = int(RANDOM_ATTACK_TAIL_LEN)
    if is_popular_attack:
        summary["config"]["popular_tail_len"] = int(POPULAR_ATTACK_TAIL_LEN)
        summary["config"]["popular_topk"] = int(POPULAR_ATTACK_TOPK)
    summary["skip_reasons"] = skip_counts
    summary["compromised_user_count"] = len(compromised_users)
    summary["compromised_users"] = compromised_users
    if desired_conversions > len(compromised_users):
        summary.setdefault("notes", {})["shortfall"] = desired_conversions - len(compromised_users)

    suffix = f"_{attack_label}_mr{mr}" if mr else f"_{attack_label}_mr0"
    combined_sequences = list(original_sequences)
    base_meta_payload = {"attack_mode": attack_variant, "count": len(compromised_users)}
    base_outputs = _write_poison_artifacts(
        poison_dir,
        suffix,
        combined_sequences=combined_sequences,
        exp_splits=exp_splits,
        user_id2idx=user_id2idx,
        user_id2name=user_id2name,
        keywords_map=keywords_map,
        embedding_deltas=embedding_deltas,
        round_metrics=round_metrics,
        summary=summary,
        shadow_meta=shadow_meta,
        shadow_meta_meta=base_meta_payload,
    )

    logging.info(
        "Compromised %d real users (requested mr=%.3f, effective mr=%.3f) out of %d candidates for dataset '%s'",
        len(compromised_users),
        mr,
        effective_mr,
        original_count,
        dataset,
    )

    defence_label = getattr(args, "defence", None)
    defence_outputs = None
    if defence_label:
        defence_result = None
        try:
            if defence_label == "hist_kl":
                baseline_path = getattr(args, "defence_baseline", None)
                if not baseline_path:
                    raise ValueError("hist_kl defence requires --defence-baseline")
                threshold = getattr(args, "defence_kl_threshold", None)
                defence_result = _apply_histogram_defence(
                    defence_label,
                    baseline_path,
                    threshold,
                    combined_sequences=combined_sequences,
                    exp_splits=exp_splits,
                    user_id2idx=user_id2idx,
                    user_id2name=user_id2name,
                    keywords_map=keywords_map,
                    embedding_deltas=embedding_deltas,
                    round_metrics=round_metrics,
                    summary=summary,
                    shadow_meta=shadow_meta,
                    compromised_users=compromised_users,
                    attack_variant=attack_variant,
                    base_count_for_mr=base_count_for_mr,
                )
            elif defence_label == "ae_filter":
                reference_seq = getattr(args, "defence_ae_reference", None) or getattr(args, "defence_baseline", None)
                threshold = float(getattr(args, "defence_ae_threshold", 0.05))
                hidden_dim = int(getattr(args, "defence_ae_hidden_dim", 3))
                epochs = int(getattr(args, "defence_ae_epochs", 200))
                defence_result = _apply_ae_defence(
                    defence_label,
                    reference_seq,
                    threshold,
                    hidden_dim,
                    epochs,
                    combined_sequences=combined_sequences,
                    exp_splits=exp_splits,
                    user_id2idx=user_id2idx,
                    user_id2name=user_id2name,
                    keywords_map=keywords_map,
                    embedding_deltas=embedding_deltas,
                    round_metrics=round_metrics,
                    summary=summary,
                    shadow_meta=shadow_meta,
                    compromised_users=compromised_users,
                    attack_variant=attack_variant,
                    base_count_for_mr=base_count_for_mr,
                    pop_counter=item_count,
                    split_dir=split_dir,
                )
            elif defence_label == "act_cluster":
                min_samples = getattr(args, "defence_ac_min_samples", None)
                small_ratio = getattr(args, "defence_ac_small_ratio", None)
                max_iter = getattr(args, "defence_ac_max_iter", None)
                max_drop = getattr(args, "defence_ac_max_drop", None)
                max_drop_ratio = getattr(args, "defence_ac_max_drop_ratio", None)
                defence_result = _apply_activation_clustering_defence(
                    defence_label,
                    adapter,
                    int(min_samples) if min_samples is not None else DEFENCE_ACT_MIN_SAMPLES,
                    float(small_ratio) if small_ratio is not None else DEFENCE_ACT_SMALL_RATIO,
                    int(max_iter) if max_iter is not None else DEFENCE_ACT_MAX_ITER,
                    int(max_drop) if max_drop is not None else DEFENCE_ACT_MAX_DROP,
                    float(max_drop_ratio) if max_drop_ratio is not None else DEFENCE_ACT_MAX_DROP_RATIO,
                    combined_sequences=combined_sequences,
                    exp_splits=exp_splits,
                    user_id2idx=user_id2idx,
                    user_id2name=user_id2name,
                    keywords_map=keywords_map,
                    embedding_deltas=embedding_deltas,
                    round_metrics=round_metrics,
                    summary=summary,
                    shadow_meta=shadow_meta,
                    compromised_users=compromised_users,
                    attack_variant=attack_variant,
                    base_count_for_mr=base_count_for_mr,
                )
            elif defence_label == "spectral_sig":
                top_ratio = getattr(args, "defence_ss_top_ratio", None)
                min_samples = getattr(args, "defence_ss_min_samples", None)
                defence_result = _apply_spectral_signature_defence(
                    defence_label,
                    adapter,
                    float(top_ratio) if top_ratio is not None else DEFENCE_SS_TOP_RATIO,
                    int(min_samples) if min_samples is not None else DEFENCE_SS_MIN_SAMPLES,
                    combined_sequences=combined_sequences,
                    exp_splits=exp_splits,
                    user_id2idx=user_id2idx,
                    user_id2name=user_id2name,
                    keywords_map=keywords_map,
                    embedding_deltas=embedding_deltas,
                    round_metrics=round_metrics,
                    summary=summary,
                    shadow_meta=shadow_meta,
                    compromised_users=compromised_users,
                    attack_variant=attack_variant,
                    base_count_for_mr=base_count_for_mr,
                )
            else:
                logging.warning("[defence:%s] Unknown defence label", defence_label)
        except Exception as exc:
            logging.error("[defence:%s] Failed to apply defence: %s", defence_label, exc)
            defence_result = None
        if defence_result:
            defence_tag = _sanitize_tag(defence_label)
            defence_dir = os.path.join(
                os.path.dirname(poison_dir),
                f"{os.path.basename(poison_dir)}_{defence_tag}",
            )
            defence_suffix = f"{suffix}_{defence_tag}"
            defence_outputs = _write_poison_artifacts(
                defence_dir,
                defence_suffix,
                combined_sequences=defence_result["combined_sequences"],
                exp_splits=defence_result["exp_splits"],
                user_id2idx=defence_result["user_id2idx"],
                user_id2name=defence_result["user_id2name"],
                keywords_map=defence_result["keywords_map"],
                embedding_deltas=defence_result["embedding_deltas"],
                round_metrics=defence_result["round_metrics"],
                summary=defence_result["summary"],
                shadow_meta=defence_result["shadow_meta"],
                shadow_meta_meta=defence_result["shadow_meta_meta"],
            )
            defence_outputs["poison_subdir"] = f"{poison_subdir}_{defence_tag}"
            logging.info(
                "[defence:%s] Filtered artefacts written to %s",
                defence_label,
                defence_outputs["poison_dir"],
            )
        else:
            logging.warning("[defence:%s] Defence configuration returned no artefacts.", defence_label)

    result = {
        **base_outputs,
        "poison_subdir": poison_subdir,
        "text_pairs_path": text_pairs_path,
    }
    if defence_outputs:
        result["defence_outputs"] = defence_outputs
        if text_pairs_path and os.path.isfile(text_pairs_path):
            defence_pair_dir = os.path.join(
                TEXT_PAIR_ROOT,
                dataset,
                "poisoned",
                defence_outputs["poison_subdir"],
            )
            os.makedirs(defence_pair_dir, exist_ok=True)
            defence_pair_path = os.path.join(defence_pair_dir, "text_pairs.jsonl")
            try:
                shutil.copy2(text_pairs_path, defence_pair_path)
                defence_outputs["text_pairs_path"] = defence_pair_path
                logging.info(
                    "[defence:%s] Copied text_pairs -> %s",
                    defence_label,
                    defence_pair_path,
                )
            except Exception as exc:
                logging.warning(
                    "[defence:%s] Failed to copy text_pairs: %s", defence_label, exc
                )
    return result


__all__ = ["run_pipeline"]
