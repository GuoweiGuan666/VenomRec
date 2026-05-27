#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified command line entry point for interactive DCIP-IEOS poisoning."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
from typing import Any, List

from .poison_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "caches")
DEFAULT_VICTIM_CKPT = os.path.join(
    PROJ_ROOT,
    "snap",
    "beauty",
    "0805",
    "NoAttack_0.0_beauty-vitb32-2-8-20",
    "BEST_EVAL_LOSS.pth",
)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _parse_targets_file(path: str, limit: int | None = None) -> List[Any]:
    """Parse targets file into a list of identifiers.

    Supports lines like:
    - "Item: B0000C52L6 (ID: 53), Count: 5"
    - "ID: 53"
    - "53"
    """

    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Targets file not found: {path}")
    id_pattern = re.compile(r"ID:\s*(\d+)")
    asin_pattern = re.compile(r"Item:\s*([A-Z0-9]+)", re.IGNORECASE)
    targets: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            match = id_pattern.search(text)
            if match:
                try:
                    targets.append(int(match.group(1)))
                except ValueError:
                    continue
            else:
                asin_match = asin_pattern.search(text)
                if asin_match:
                    targets.append(asin_match.group(1).upper())
                elif text.isdigit():
                    try:
                        targets.append(int(text))
                    except ValueError:
                        continue
            if limit is not None and len(targets) >= limit:
                break
    if not targets:
        raise ValueError(f"No targets parsed from {path}")
    return targets


def _default_pop_path(dataset: str) -> str:
    return os.path.join(
        PROJ_ROOT,
        "analysis",
        "results",
        dataset,
        f"high_pop_items_{dataset}_highcount_100.txt",
    )


def _default_targets_path(dataset: str) -> str:
    return os.path.join(
        PROJ_ROOT,
        "analysis",
        "results",
        dataset,
        f"low_pop_items_{dataset}_lowcount_1.txt",
    )


def _ensure_competition_pool(args: argparse.Namespace) -> None:
    """Build per-run competition pool for multi-target poisoning."""

    targets_path = getattr(args, "targets_path", None)
    if not targets_path:
        default_targets = _default_targets_path(args.dataset)
        if os.path.isfile(default_targets):
            targets_path = default_targets
            args.targets_path = default_targets
        else:
            return

    max_targets = getattr(args, "max_targets", None)
    targets = _parse_targets_file(targets_path, max_targets)

    cache_dir = getattr(args, "cache_dir", DEFAULT_CACHE_DIR)
    if os.path.abspath(cache_dir) == os.path.abspath(DEFAULT_CACHE_DIR):
        tag = os.path.splitext(os.path.basename(targets_path))[0]
        digest = hashlib.md5("\n".join(map(str, targets)).encode("utf-8")).hexdigest()[:8]
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, f"mt_{args.dataset}_{tag}_{digest}")
        args.cache_dir = cache_dir

    if getattr(args, "poison_subdir", None) is None:
        digest = hashlib.md5("\n".join(map(str, targets)).encode("utf-8")).hexdigest()[:8]
        args.poison_subdir = f"dcip_ieos_fc_mt{len(targets)}_{digest}"

    pop_path = getattr(args, "pop_path", None)
    if not pop_path:
        pop_path = _default_pop_path(args.dataset)
        args.pop_path = pop_path

    if not os.path.isfile(pop_path):
        raise FileNotFoundError(f"High-popularity file not found: {pop_path}")

    os.makedirs(cache_dir, exist_ok=True)
    comp_path = os.path.join(cache_dir, f"competition_pool_{args.dataset}.json")
    if os.path.isfile(comp_path):
        logging.info("[run_dcip_ieos] using existing competition pool: %s", comp_path)
        return

    from . import pool_miner

    pool_info = pool_miner.build_competition_pool(
        dataset=args.dataset,
        pop_path=pop_path,
        targets=targets,
        model=None,
        cache_dir=None,
        c_size=8,
        keyword_top=50,
        min_keywords=5,
        feat_root=getattr(args, "feat_root", "features"),
        feat_backbone=getattr(args, "feat_backbone", "vitb32_features"),
        img_dim=512,
    )

    legacy: List[dict] = []
    pool_dict = pool_info.get("pool", {})
    for asin, info in pool_dict.items():
        legacy.append(
            {
                "target": asin,
                "neighbors": info.get("competitors", []),
                "anchor": info.get("anchor", []),
                "keywords": info.get("keywords", []),
                "synthetic": bool(info.get("synthetic", False)),
            }
        )

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False, indent=2)
    logging.info(
        "[run_dcip_ieos] wrote competition pool with %d targets -> %s",
        len(legacy),
        comp_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Simplified DCIP-IEOS poisoning runner")
    parser.add_argument("dataset", help="Dataset split (e.g. clothing)")
    parser.add_argument("mr", type=float, help="Malicious ratio (e.g. 0.1)")
    parser.add_argument("gpu_list", help="Comma separated GPU ids (compat placeholder)")
    parser.add_argument("image_feature_type", default="vitb32", help="Vision backbone name")
    parser.add_argument("image_feature_size_ratio", type=int, default=2)
    parser.add_argument("reduction_factor", type=int, default=8)
    parser.add_argument("epochs", type=int, default=12)

    parser.add_argument("--data-root", default=os.path.join(PROJ_ROOT, "data"))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--feat-root", default="features")
    parser.add_argument("--feat-backbone", default="vitb32_features")
    parser.add_argument("--pop-path", default=None)
    parser.add_argument("--targets-path", default=None)

    parser.add_argument("--sequence-length", type=int, default=10, help="Poisoned sequence length")
    parser.add_argument("--interaction-rounds", type=int, default=4, help="Number of alternating perturbation rounds")
    # Mask ratios (support legacy aliases from shell script)
    parser.add_argument(
        "--mask-vis-ratio", "--mask-top-q", dest="mask_vis_ratio", type=float, default=0.15,
        help="Top ratio for visual mask (alias: --mask-top-q)")
    parser.add_argument(
        "--mask-txt-ratio", "--mask-top-p", dest="mask_txt_ratio", type=float, default=0.15,
        help="Top ratio for text mask (alias: --mask-top-p)")

    # Image perturb budgets (support legacy alias --eps)
    parser.add_argument(
        "--img-eps", "--eps", dest="img_eps", type=float, default=0.05,
        help="Max image perturbation (L_inf) (alias: --eps)")
    parser.add_argument(
        "--iters", dest="img_iters", type=int, default=3,
        help="PGD iterations for image perturbation (when strategy=pgd)")
    parser.add_argument(
        "--psnr_min", dest="psnr_min", type=float, default=30.0,
        help="Minimum PSNR to keep visual quality during PGD")
    parser.add_argument(
        "--img-strategy", dest="img_strategy", choices=["cosine", "pgd"], default="cosine",
        help="Image perturbation strategy: cosine (single-step) or pgd (multi-step)")
    parser.add_argument("--txt-ratio-max", type=float, default=0.2, help="Max text replacement ratio per round")
    parser.add_argument("--min-txt-replacements", type=int, default=2, help="Min replacements per round")
    parser.add_argument("--sim-threshold", type=float, default=0.92, help="Similarity threshold to stop iterations")
    parser.add_argument("--txt-embed-eps", type=float, default=0.01, help="Max embedding perturbation magnitude")
    parser.add_argument(
        "--cross-mode",
        choices=["interactive", "decoupled"],
        default="interactive",
        help="Whether to recompute text masks after image updates ('interactive') or keep the initial masks ('decoupled').",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lock-vis-mask", action="store_true", help="Keep first visual mask across rounds")
    parser.add_argument("--vis-decay", type=float, default=1.0, help="Per-round decay factor applied to visual mask ratio")
    parser.add_argument(
        "--stop-on-plateau",
        dest="stop_on_plateau",
        action="store_true",
        help="Stop iteration when cosine similarity stops improving (default)",
    )
    parser.add_argument(
        "--no-stop-on-plateau",
        dest="stop_on_plateau",
        action="store_false",
        help="Continue iterating even if cosine similarity plateaus",
    )
    parser.set_defaults(stop_on_plateau=True)

    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="If set, only poison the first N targets from the competition pool (for smoke tests)",
    )

    parser.add_argument(
        "--victim-ckpt",
        default=DEFAULT_VICTIM_CKPT,
        help="Path to pretrained VIP5 checkpoint used as victim",
    )
    parser.add_argument(
        "--victim-backbone",
        default="t5-base",
        help="Backbone name for victim model/tokenizer (e.g. t5-base)",
    )
    parser.add_argument(
        "--victim-device",
        default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        help="Device to load victim model on (e.g. cuda:0 or cpu)",
    )

    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--log-rounds", type=_to_bool, default=False, help="Log per-target round metrics summary")
    parser.add_argument(
        "--shadow-seed",
        default=None,
        help="Path to JSON file of query-derived user seeds (for grey-box shadow user construction).",
    )
    parser.add_argument(
        "--poison-subdir",
        dest="poison_subdir",
        default=None,
        help=(
            "Optional subdirectory name under data/<dataset>/poisoned/ used to store the generated artefacts. "
            "When omitted, a name is derived from attack parameters (e.g. dcip_ieos_fc_mr0.001_ir10_img0.05_txt0.2)."
        ),
    )
    parser.add_argument(
        "--text-pair-root",
        dest="text_pair_root",
        default=None,
        help=(
            "Optional base directory for storing text_pairs.jsonl files. "
            "Defaults to <repo>/poison_text_pairs when not provided."
        ),
    )
    parser.add_argument(
        "--attack-variant",
        choices=["byzantine"],
        default="byzantine",
        help="Poisoning variant. The public release exposes the main VenomRec/DCIP-IEOS setting only.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _ensure_competition_pool(args)

    logging.info("[run_dcip_ieos] launching poisoning: dataset=%s mr=%.3f", args.dataset, args.mr)
    result = run_pipeline(args)
    logging.info("[run_dcip_ieos] artefacts written to: %s", json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
