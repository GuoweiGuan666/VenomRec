#!/usr/bin/env python3
"""
Export competition_pool_<dataset>.json in the legacy list format.

Modern `pool_miner.build_competition_pool` writes a rich dictionary
structure (with raw items, metadata, etc.).  Older parts of the pipeline
still expect the simplified list of dictionaries::

    [
        {
            "target": "...",
            "neighbors": [...],
            "anchor": [...],
            "keywords": [...],
            "synthetic": bool
        },
        ...
    ]

This helper recreates that file for any dataset (e.g., sports/toys) by
invoking `pool_miner.build_competition_pool` and converting the output to
the legacy schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List

from attack.ours.dcip_ieos import pool_miner


def _parse_item_list(path: str, limit: int | None = None) -> List[str]:
    """Parse `analysis/results/...` popularity files.

    Lines look like ``Item: B0000C52L6 (ID: 53), Count: 5``.  We preserve ASINs
    because the downstream builder can map them back via datamaps.
    """

    asin_pattern = re.compile(r"Item:\s*([A-Z0-9]+)", re.IGNORECASE)
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = asin_pattern.search(line)
            if not m:
                continue
            items.append(m.group(1).upper())
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise ValueError(f"failed to parse any items from {path!r}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert competition pool to legacy format")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., clothing/sports)")
    parser.add_argument("--pop-path", required=True, help="Path to high-popularity items txt")
    parser.add_argument("--targets-path", required=True, help="Path to low-popularity targets txt")
    parser.add_argument("--output-dir", default="attack/ours/dcip_ieos/caches", help="Output directory")
    parser.add_argument("--target-limit", type=int, default=None, help="Optional cap on number of targets")
    parser.add_argument("--c-size", type=int, default=8, help="Number of neighbours per target (legacy=8)")
    parser.add_argument("--keywords-top", type=int, default=50, help="Max keywords per target")
    parser.add_argument(
        "--min-keywords",
        type=int,
        default=None,
        help="Minimum number of keywords per target (defaults to --keywords-top)",
    )
    parser.add_argument("--feat-root", default="features", help="Root directory that stores feature .npy files")
    parser.add_argument("--feat-backbone", default="vitb32_features", help="Subfolder under feat-root for the backbone")
    parser.add_argument("--img-dim", type=int, default=512, help="Fallback feature dim when a .npy is missing")
    parser.add_argument(
        "--allow-missing-image",
        action="store_true",
        default=True,
        help="If set, missing image feats become zeros instead of raising",
    )
    parser.add_argument("--no-allow-missing-image", dest="allow_missing_image", action="store_false")
    parser.add_argument(
        "--allow-missing-text",
        action="store_true",
        default=True,
        help="If set, missing text falls back to meta/review before raising",
    )
    parser.add_argument("--no-allow-missing-text", dest="allow_missing_text", action="store_false")
    parser.add_argument(
        "--kmeans-k",
        type=int,
        default=8,
        help="Number of clusters for neighbour mining (ignored when --no-kmeans)",
    )
    parser.add_argument(
        "--no-kmeans",
        action="store_true",
        help="Disable KMeans clustering so neighbours come from the entire high-pop set",
    )
    args = parser.parse_args()

    targets = _parse_item_list(args.targets_path, args.target_limit)
    min_keywords = args.min_keywords if args.min_keywords is not None else args.keywords_top
    kmeans_k = None if args.no_kmeans else args.kmeans_k

    pool_info = pool_miner.build_competition_pool(
        dataset=args.dataset,
        pop_path=args.pop_path,
        targets=targets,
        model=None,
        cache_dir=None,
        c_size=args.c_size,
        keyword_top=args.keywords_top,
        min_keywords=min_keywords,
        kmeans_k=kmeans_k,
        feat_root=args.feat_root,
        feat_backbone=args.feat_backbone,
        img_dim=args.img_dim,
        allow_missing_image=args.allow_missing_image,
        allow_missing_text=args.allow_missing_text,
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

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"competition_pool_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote {len(legacy)} targets -> {out_path}")


if __name__ == "__main__":
    main()
