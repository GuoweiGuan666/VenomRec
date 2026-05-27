#!/usr/bin/env python
"""Precompute candidate item lists for VIP5 direct tasks.

Offline caching avoids sampling the candidate pool on every batch and greatly
reduces DataLoader overhead once cached.

Example
-------
python scripts/precompute_candidates.py --dataset clothing --attack-mode DcipIeosFcAttack --mr 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh]


def camel_to_snake(name: str) -> str:
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    if name.endswith("_attack"):
        name = name[:-7]
    if name == "no":  # handle "NoAttack"
        name = "noattack"
    alias = {
        "direct_boosting": "direct_boost",
        "random_attack": "random_attack",
        "random": "random_attack",
        "popular_attack": "popular_attack",
        "popular": "popular_attack",
        "popular_item_mimicking": "popular_mimicking",
        "shadow_cast": "shadowcast",
        "shadowcast": "shadowcast",
        "dcip_ieos": "dcip_ieos",
        "dcip_ieos_fc": "dcip_ieos_fc",
        "dcip_ieos_fc_ablation_mode1": "dcip_ieos_fc_ablation_mode1",
    }
    return alias.get(name, name)


def sample_candidates(
    user_seq: list[int],
    catalog: np.ndarray,
    candidate_num: int,
    *,
    rng: np.random.Generator,
    allow_duplicates: bool = True,
) -> list[str]:
    user_seq_set = {str(u) for u in user_seq}
    candidates: list[str] = []
    attempts = 0
    max_attempts = 10

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
                return candidates[:candidate_num]
            while len(candidates) < candidate_num:
                candidates.append(str(rng.choice(pool)))

    return candidates[:candidate_num]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute direct-task candidate cache")
    parser.add_argument("--dataset", default="clothing", help="Dataset split name")
    parser.add_argument("--attack-mode", default="NoAttack", help="Attack mode name (CamelCase or snake_case)")
    parser.add_argument("--mr", type=float, default=0.0, help="Malicious ratio")
    parser.add_argument("--candidate-num", type=int, default=128, help="Number of cached candidates per user")
    parser.add_argument("--data-root", default="data", help="Base data directory")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed for reproducibility")
    parser.add_argument("--poison-subdir", default=None, help="Optional subdirectory inside data/<dataset>/poisoned containing poisoned artefacts")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    atk_snake = camel_to_snake(args.attack_mode)
    mr_str = str(float(args.mr))

    poison_subdir = args.poison_subdir.strip("/\\") if args.poison_subdir else None

    shadowcast_mode = atk_snake == "dcip_ieos_shadowcast"

    if poison_subdir:
        base_dir = Path(args.data_root) / args.dataset / "poisoned" / poison_subdir
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Poison subdirectory not found: {base_dir}")
        if shadowcast_mode:
            seq_path = Path(args.data_root) / args.dataset / "sequential_data.txt"
            suffix_token = f"{atk_snake}_mr{mr_str}"
            cache_name = f"candidate_cache_{suffix_token}.pkl"
            cache_path = base_dir / cache_name
        else:
            seq_candidate = base_dir / f"sequential_data_{atk_snake}_mr{mr_str}.txt"
            if seq_candidate.exists():
                seq_path = seq_candidate
                suffix_token = f"{atk_snake}_mr{mr_str}"
            else:
                matches = sorted(base_dir.glob("sequential_data_*.txt"))
                if not matches:
                    raise FileNotFoundError(f"Sequential data not found in {base_dir}")
                seq_path = matches[0]
                suffix_token = seq_path.stem[len("sequential_data"):].lstrip("_")
            cache_suffix = suffix_token or None
            cache_name = f"candidate_cache_{cache_suffix}.pkl" if cache_suffix else "candidate_cache_clean.pkl"
            cache_path = base_dir / cache_name
    elif atk_snake not in ("none", "noattack") and args.mr > 0:
        base_dir = Path(args.data_root) / args.dataset / "poisoned"
        seq_path = base_dir / f"sequential_data_{atk_snake}_mr{mr_str}.txt"
        cache_path = base_dir / f"candidate_cache_{atk_snake}_mr{mr_str}.pkl"
    else:
        base_dir = Path(args.data_root) / args.dataset
        seq_path = base_dir / "sequential_data.txt"
        cache_path = base_dir / "candidate_cache_clean.pkl"

    if not seq_path.exists():
        raise FileNotFoundError(f"Sequential data not found: {seq_path}")

    print(f"[INFO] Loading sequential data from {seq_path}")
    sequential = read_lines(seq_path)

    user_items: Dict[str, list[int]] = {}
    item_count: Dict[int, int] = {}
    for line in sequential:
        if not line:
            continue
        parts = line.split()
        user = parts[0]
        items = [int(x) for x in parts[1:]]
        user_items[user] = items
        for it in items:
            item_count[it] = item_count.get(it, 0) + 1

    catalog = list(item_count.keys())
    catalog_np = np.array(catalog, dtype=np.int64)
    if not catalog:
        raise RuntimeError("Empty catalog; cannot precompute candidates")

    print(f"[INFO] Sampling candidates for {len(user_items)} users (candidate_num={args.candidate_num})")
    cache: Dict[str, List[str]] = {}
    user_keys = list(user_items.keys())
    for idx, user in enumerate(tqdm(user_keys, total=len(user_keys), desc="users", mininterval=0.5, miniters=1)):
        seq = user_items[user]
        cache[user] = sample_candidates(
            seq,
            catalog_np,
            args.candidate_num,
            rng=rng,
        )
        if (idx + 1) % 100 == 0:
            print(f"[INFO] processed {idx + 1}/{len(user_keys)} users")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)

    print(f"[DONE] Candidate cache written to {cache_path}")


if __name__ == "__main__":
    main()
