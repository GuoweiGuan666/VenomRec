#!/usr/bin/env python
"""Standalone evaluation script for VIP5.

This replaces the old notebook workflow and runs entirely from the CLI.  It
loads a fine-tuned checkpoint, builds the required dataset loader, and
reports the requested metrics for direct / sequential / explanation tasks.

Example usage::

    python scripts/evaluate_direct.py \
        --ckpt snap/clothing/0924_182005/DcipIeosFcAttack_0.1_clothing-vitb32-2-8-12/BEST_HIT.pth \
        --task direct \
        --prompts B-5 \
        --batch-size 16 \
        --num-workers 4 \
        --device cuda:0

To evaluate multiple prompts, pass a comma-separated list, e.g. ``--prompts
B-5,B-8``.
"""

from __future__ import annotations

import argparse
import json
import os

# Ensure Transformers falls back to the slow tokenizer implementation so the
# optional `tokenizers` dependency is not required when running evaluations.
os.environ.setdefault("USE_TOKENIZERS", "0")

import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
try:  # ensure optional dependency is available
    import tokenizers  # type: ignore  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "The 'tokenizers' package is required for evaluation. Activate the training environment "
        "(vip5_env) or install tokenizers via `pip install tokenizers`."
    ) from exc

import numpy as np
import torch
try:
    from adapters import AdapterConfig  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    try:
        from transformers.adapters import AdapterConfig  # type: ignore
    except ImportError:
        class AdapterConfig:  # minimal stub for evaluation only
            def __init__(self):
                self.tasks = None
                self.d_model = None
                self.use_single_adapter = None
                self.reduction_factor = None
                self.track_z = None
                self.non_linearity = "relu"
from tqdm import tqdm

# Ensure project modules are importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
for path in (PROJECT_ROOT, SRC_DIR, NOTEBOOK_DIR):
    if str(path) not in os.sys.path:
        os.sys.path.insert(0, str(path))

from src.tokenization import P5Tokenizer
from src.model import VIP5Tuning
from src.utils import load_state_dict, parse_item_id, load_target_items, load_target_items_from_path
from src.data import get_loader
from evaluate.metrics4rec import evaluate_all
from evaluate.utils import bleu_score, rouge_score
from transformers import T5Config, T5Tokenizer


@dataclass
class EvalConfig:
    ckpt: Path
    task: str
    prompts: Tuple[str, ...]
    batch_size: int
    num_workers: int
    device: str
    beam_size: int
    max_length: int
    return_sequences: int
    output_dir: Path | None
    attack_mode: str | None
    mr: float | None
    split: str | None
    backbone: str
    debug_samples: int
    poison_subdir: str | None


class DotDict(dict):
    """Dictionary with attribute access convenience."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def parse_checkpoint_metadata(ckpt_path: Path) -> Dict[str, str]:
    """Extract attack/dataset metadata from checkpoint directory name.

    Expected pattern: ``<Attack>_<mr>_<dataset>-<feat>-<size>-<reduction>-<epoch>``
    Example: ``DcipIeosFcAttack_0.1_clothing-vitb32-2-8-12``
    """

    folder_name = ckpt_path.parent.name
    try:
        attack, mr_str, rest = folder_name.split("_", 2)
        dataset, feat, size_ratio, reduction, epoch = rest.split("-")
    except ValueError as exc:
        raise ValueError(
            f"Unable to parse checkpoint folder '{folder_name}'. "
            "Expected pattern: <Attack>_<mr>_<dataset>-<feat>-<size>-<reduction>-<epoch>"
        ) from exc

    return {
        "attack_mode": attack,
        "mr": mr_str,
        "split": dataset,
        "image_feature_type": feat,
        "image_feature_size_ratio": size_ratio,
        "reduction_factor": reduction,
        "epoch": epoch,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VIP5 direct/sequential/explanation tasks.")
    parser.add_argument("--ckpt", required=True, help="Path to fine-tuned checkpoint (.pth)")
    parser.add_argument(
        "--task",
        choices=["direct", "sequential", "explanation"],
        required=True,
        help="Evaluation task type.",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Comma-separated prompt templates, e.g. 'B-5,B-8'.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--return-sequences", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store evaluation logs (defaults to log/<split>/<date>/evaluation_logs)",
    )
    parser.add_argument("--attack-mode", default=None)
    parser.add_argument("--mr", type=float, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--backbone", default="t5-small")
    parser.add_argument("--debug-samples", type=int, default=0,
                        help="Print up to N raw predictions that failed to parse.")
    parser.add_argument(
        "--poison-subdir",
        default=None,
        help="Relative subdirectory under data/<split>/poisoned/ containing poisoned artefacts.",
    )
    parser.add_argument(
        "--eval-er",
        action="store_true",
        help="Also compute Exposure Rate (ER) by re-running evaluation with targeted items injected into candidates.",
    )
    parser.add_argument(
        "--targets-path",
        default=None,
        help="Optional path to target item list (analysis format or numeric IDs) used for ER evaluation.",
    )
    return parser


# ---------------------------------------------------------------------------
# Argument & config initialisation
# ---------------------------------------------------------------------------


def build_args(cfg: EvalConfig) -> DotDict:
    meta = parse_checkpoint_metadata(cfg.ckpt)

    args = DotDict()
    args.load = str(cfg.ckpt)

    # Metadata
    args.attack_mode = cfg.attack_mode or meta["attack_mode"]
    args.mr = cfg.mr if cfg.mr is not None else float(meta["mr"])
    args.split = cfg.split or meta["split"]
    args.train = args.valid = args.test = args.split
    args.image_feature_type = meta["image_feature_type"]
    args.image_feature_size_ratio = int(meta["image_feature_size_ratio"])
    args.reduction_factor = int(meta["reduction_factor"])
    args.epoch = int(meta["epoch"])
    args.poison_subdir = cfg.poison_subdir

    # Model/training hyper-parameters (aligned with training defaults)
    args.use_adapter = True
    args.use_single_adapter = True
    args.use_vis_layer_norm = True
    args.add_adapter_cross_attn = True
    args.use_lm_head_adapter = True

    args.tokenizer = "p5"
    args.backbone = cfg.backbone
    args.max_text_length = 1024
    args.gen_max_length = 64
    args.do_lower_case = False
    args.dropout = 0.1
    args.weight_decay = 0.01
    args.adam_eps = 1e-6
    args.gradient_accumulation_steps = 1
    args.losses = "sequential,direct,explanation"

    args.seed = 2022
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.whole_word_embed = True
    args.category_embed = True
    args.world_size = torch.cuda.device_count() or 1

    args.batch_size = cfg.batch_size
    args.num_workers = cfg.num_workers
    args.distributed = False
    args.multiGPU = args.world_size > 1
    args.fp16 = True
    args.clip_grad_norm = 5.0
    args.lr = 1e-3
    args.warmup_ratio = 0.1
    args.optim = "adamw"
    args.comment = ""
    args.local_rank = 0

    # Target item lists (parse from analysis directory)
    args.data_target = {}
    for ds in ["beauty", "clothing", "sports", "toys", "microlens"]:
        ids = load_target_items(ds, str(PROJECT_ROOT))
        if ids:
            args.data_target[ds] = ids

    return args


# ---------------------------------------------------------------------------
# Model / tokenizer initialisation
# ---------------------------------------------------------------------------


def create_config(args: DotDict) -> T5Config:
    config = T5Config.from_pretrained(args.backbone)
    for k, v in args.items():
        setattr(config, k, v)
    config.non_linearity = "relu"

    feat_dim_map = {
        "vitb32": 512,
        "vitb16": 512,
        "vitl14": 768,
        "rn50": 1024,
        "rn101": 512,
    }
    config.feat_dim = feat_dim_map[args.image_feature_type]
    config.n_vis_tokens = args.image_feature_size_ratio

    adapter_cfg = AdapterConfig()
    adapter_cfg.tasks = args.losses.split(",")
    adapter_cfg.d_model = config.d_model
    adapter_cfg.use_single_adapter = args.use_single_adapter
    adapter_cfg.reduction_factor = args.reduction_factor
    adapter_cfg.track_z = False
    config.adapter_config = adapter_cfg
    return config


def create_tokenizer(args: DotDict):
    if args.tokenizer.lower() == "p5":
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case,
        )
    else:
        tokenizer = T5Tokenizer.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case,
        )
    print(f"Tokenizer built: {type(tokenizer).__name__}")
    return tokenizer


def create_model(args: DotDict, config: T5Config, device: torch.device):
    # Maintain backward compatibility with original VIP5 implementation
    VIP5Tuning.model = property(lambda self: self)
    model = VIP5Tuning.from_pretrained(args.backbone, config=config)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def prepare_output_directory(cfg: EvalConfig, args: DotDict) -> Path:
    if cfg.output_dir:
        out_dir = cfg.output_dir
    else:
        date_tag = Path(args.load).parents[1].name if args.load else datetime.now().strftime("%m%d")
        out_dir = PROJECT_ROOT / "log" / args.split / date_tag / "evaluation_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def collect_generation(model: VIP5Tuning, loader, device: torch.device, cfg: EvalConfig):
    model.eval()
    all_info = []
    for _, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating"):
        with torch.no_grad():
            # move tensor fields to device, keep lists/strings untouched
            device_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    device_batch[key] = value.to(device)
                elif isinstance(value, list) and value and torch.is_tensor(value[0]):
                    device_batch[key] = [v.to(device) for v in value]
                else:
                    device_batch[key] = value

            _ = model.generate_step(device_batch)
            beam_outputs = model.generate(
                input_ids=device_batch["input_ids"],
                whole_word_ids=device_batch["whole_word_ids"],
                category_ids=device_batch["category_ids"],
                vis_feats=device_batch["vis_feats"],
                task=batch["task"][0],
                max_length=cfg.max_length,
                num_beams=cfg.beam_size,
                no_repeat_ngram_size=0,
                num_return_sequences=cfg.return_sequences,
                early_stopping=True,
            )
            generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        for j, target in enumerate(batch["target_text"]):
            start = j * cfg.return_sequences
            end = start + cfg.return_sequences
            all_info.append({
                "target_item": target,
                "gen_item_list": generated_sents[start:end],
            })
    return all_info


def build_score_dict(
    all_info: List[Dict[str, List[str]]],
    debug_samples: int = 0,
    override_target: str | None = None,
) -> Tuple[Dict[int, List[int]], Dict[int, Dict[int, float]], Dict[str, int], List[str]]:
    gt: Dict[int, List[int]] = {}
    ui_scores: Dict[int, Dict[int, float]] = {}
    parse_fail_targets = 0
    parse_fail_preds = 0
    debug_logs: List[str] = []
    valid_idx = 0
    override_id = None
    if override_target is not None:
        override_id = parse_item_id(str(override_target))
        if override_id is None:
            raise ValueError(f"Unable to parse override target '{override_target}'")

    for info in all_info:
        if override_id is not None:
            target = override_id
        else:
            target = parse_item_id(info["target_item"])
            if target is None:
                parse_fail_targets += 1
                if debug_samples and len(debug_logs) < debug_samples:
                    debug_logs.append(f"target='{info['target_item']}'")
                continue
        preds: Dict[int, float] = {}
        for rank, item in enumerate(info["gen_item_list"], start=1):
            parsed = parse_item_id(item)
            if parsed is None:
                parse_fail_preds += 1
                if debug_samples and len(debug_logs) < debug_samples:
                    debug_logs.append(f"pred='{item}'")
                continue
            if parsed not in preds:
                preds[parsed] = -float(rank)
        if not preds:
            continue
        gt[valid_idx] = [target]
        ui_scores[valid_idx] = preds
        valid_idx += 1

    stats = {
        "target_failures": parse_fail_targets,
        "pred_failures": parse_fail_preds,
        "processed": valid_idx,
    }
    return gt, ui_scores, stats, debug_logs


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = build_arg_parser()
    cli_args = parser.parse_args()

    default_prompt = {
        "direct": "B-5",
        "sequential": "A-3",
        "explanation": "C-3",
    }
    prompt_arg = (
        tuple(p.strip() for p in cli_args.prompts.split(",") if p.strip())
        if cli_args.prompts
        else (default_prompt[cli_args.task],)
    )

    cfg = EvalConfig(
        ckpt=Path(cli_args.ckpt).resolve(),
        task=cli_args.task,
        prompts=prompt_arg,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        device=cli_args.device,
        beam_size=cli_args.beam_size,
        max_length=cli_args.max_length,
        return_sequences=cli_args.return_sequences,
        output_dir=cli_args.output_dir,
        attack_mode=cli_args.attack_mode,
        mr=cli_args.mr,
        split=cli_args.split,
        backbone=cli_args.backbone,
        debug_samples=cli_args.debug_samples,
        poison_subdir=cli_args.poison_subdir,
    )

    device = torch.device(cfg.device)
    args = build_args(cfg)
    if cli_args.targets_path:
        override_targets = load_target_items_from_path(cli_args.targets_path)
        if override_targets:
            args.data_target[args.split] = override_targets
    args.eval_target_item = None

    torch.cuda.set_device(device) if device.type == "cuda" else None

    config = create_config(args)
    tokenizer = create_tokenizer(args)
    model = create_model(args, config, device)
    if isinstance(tokenizer, P5Tokenizer):
        model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer

    print(f"Loading checkpoint: {args.load}")
    state_dict = load_state_dict(args.load, "cpu")
    model.load_state_dict(state_dict, strict=False)

    out_dir = prepare_output_directory(cfg, args)

    results: Dict[str, Any] = {}
    targeted = [str(t) for t in args.data_target.get(args.split, [])]

    hr_results: Dict[str, Any] = {}

    for prompt in cfg.prompts:
        task_list = {cfg.task: [prompt]}
        sample_numbers = {"sequential": (1, 1), "direct": (1, 1), "explanation": 1}
        args.eval_target_item = None
        loader = get_loader(
            args,
            task_list,
            sample_numbers,
            split=args.test,
            mode="test",
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
            data_root="data",
            feature_root="features",
        )
        print(f"Task {cfg.task}, prompt {prompt}: {len(loader)} batches")

        if cfg.task == "explanation":
            tokens_predict: List[str] = []
            tokens_gt: List[str] = []
            for _, batch in tqdm(enumerate(loader), total=len(loader), desc="Generating"):
                with torch.no_grad():
                    preds = model.generate_step(batch)
                    tokens_predict.extend(preds)
                    tokens_gt.extend(batch["target_text"])

            bleu1 = bleu_score(tokens_gt, tokens_predict, n_gram=1, smooth=False)
            bleu4 = bleu_score(tokens_gt, tokens_predict, n_gram=4, smooth=False)
            rouge = rouge_score(tokens_gt, tokens_predict)
            print(f"BLEU-1 {bleu1:7.4f}, BLEU-4 {bleu4:7.4f}")
            for key, value in rouge.items():
                print(f"{key} {value:7.4f}")

            results[prompt] = {
                "bleu1": bleu1,
                "bleu4": bleu4,
                "rouge": rouge,
            }
        else:
            all_info = collect_generation(model, loader, device, cfg)
            gt, ui_scores, stats, debug_logs = build_score_dict(all_info, cfg.debug_samples)

            if stats["target_failures"] or stats["pred_failures"]:
                print(
                    f"[debug] parse failures: targets={stats['target_failures']}, "
                    f"preds={stats['pred_failures']} (processed={stats['processed']})"
                )
                if debug_logs:
                    for sample in debug_logs:
                        print(f"  raw {sample}")

            metrics = {}
            for k in (1, 5, 10, 20):
                msg, res = evaluate_all(ui_scores, gt, targeted, k)
                metrics[f"top{k}"] = {"summary": msg, **res}
                print(f"\nMetrics @{k}\n{msg}")
            hr_results[prompt] = metrics

    results["hr"] = hr_results

    if cli_args.eval_er and targeted:
        er_results: Dict[str, Dict[str, Any]] = {}
        for tgt in targeted:
            args.eval_target_item = tgt
            for prompt in cfg.prompts:
                task_list = {cfg.task: [prompt]}
                sample_numbers = {"sequential": (1, 1), "direct": (1, 1), "explanation": 1}
                loader = get_loader(
                    args,
                    task_list,
                    sample_numbers,
                    split=args.test,
                    mode="test",
                    batch_size=args.batch_size,
                    workers=args.num_workers,
                    distributed=args.distributed,
                    data_root="data",
                    feature_root="features",
                )
                print(f"[ER] Task {cfg.task}, prompt {prompt}, target {tgt}: {len(loader)} batches")
                all_info = collect_generation(model, loader, device, cfg)
                gt, ui_scores, stats, debug_logs = build_score_dict(all_info, cfg.debug_samples, override_target=tgt)

                if stats["target_failures"] or stats["pred_failures"]:
                    print(
                        f"[ER][debug] parse failures: targets={stats['target_failures']}, "
                        f"preds={stats['pred_failures']} (processed={stats['processed']})"
                    )
                    if debug_logs:
                        for sample in debug_logs:
                            print(f"  raw {sample}")

                metrics = {}
                er_target_id = parse_item_id(str(tgt))
                if er_target_id is None:
                    raise ValueError(f"Unable to parse targeted item '{tgt}' for ER computation.")
                er_target_ids = [er_target_id]
                for k in (1, 5, 10, 20):
                    msg, res = evaluate_all(ui_scores, gt, er_target_ids, k)
                    metrics[f"top{k}"] = {"summary": msg, **res}
                    print(f"\n[ER] Metrics @{k} (target={tgt})\n{msg}")
                er_results.setdefault(tgt, {})[prompt] = metrics
        args.eval_target_item = None
        results["er"] = er_results
        # Average ER across targets for each prompt (per top-k).
        er_avg: Dict[str, Any] = {}
        for prompt in cfg.prompts:
            prompt_avg: Dict[str, Any] = {}
            for k in (1, 5, 10, 20):
                vals: List[float] = []
                for tgt in targeted:
                    entry = er_results.get(tgt, {}).get(prompt, {})
                    metric = entry.get(f"top{k}", {})
                    if "er" in metric:
                        vals.append(metric["er"])
                if vals:
                    prompt_avg[f"top{k}"] = {
                        "er": float(sum(vals) / len(vals)),
                        "n": len(vals),
                    }
            if prompt_avg:
                er_avg[prompt] = prompt_avg
        if er_avg:
            results["er_avg"] = er_avg
            for prompt, metrics in er_avg.items():
                for k, entry in metrics.items():
                    print(f"[ER][avg] {prompt} {k}: {entry['er']:.6f} (n={entry['n']})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "checkpoint": args.load,
        "task": cfg.task,
        "prompts": cfg.prompts,
        "attack_mode": args.attack_mode,
        "mr": args.mr,
        "split": args.split,
        "device": cfg.device,
        "timestamp": timestamp,
    }
    output = {"meta": meta, "results": results}

    out_path = out_dir / f"{cfg.task}_eval_{cfg.prompts[0]}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
