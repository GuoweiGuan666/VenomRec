#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation helpers for VIP5 + DCIP-IEOS.

This module provides small, framework-agnostic utilities that can be imported
directly from a notebook to compute HR/NDCG/MRR and ER, with consistent
definitions and reproducibility controls. It avoids model or dataset
assumptions: you feed ranked lists (or scores) and ground-truth items.

Typical usage in a notebook:

    from src.eval_utils import (
        seed_everything, metrics_from_ranked, exposure_rate,
        rerank_discount_compensate, load_popularity_counts
    )

    # topk_items: List[List[item_id]] per user
    # gt_items:   List[item_id] per user
    out = metrics_from_ranked(topk_items, gt_items, ks=(5,10))
    er  = exposure_rate(topk_items, targets=target_ids, k=10, mode='user')

    # Optional re-rank (discount + compensate)
    pop = load_popularity_counts('data/clothing/sequential_data.txt')
    new_scores = rerank_discount_compensate(scores, item_ids, pop,
                                            alpha=0.8, eta=0.05)

All functions are pure-Python and NumPy-optional (fall back to lists).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math
import random

try:  # optional
    import numpy as np
except Exception:
    np = None  # type: ignore


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int | None) -> None:
    """Fix Python and NumPy randomness when provided."""
    if seed is None:
        return
    random.seed(seed)
    try:
        if np is not None:
            np.random.seed(seed)  # type: ignore[attr-defined]
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Ranking metrics (HR/NDCG/MRR)
# -----------------------------------------------------------------------------
def _dcg_at_k(rank: int) -> float:
    # rank is 1-based
    return 1.0 / math.log2(rank + 1.0)


def metrics_from_ranked(
    topk_item_ids: Sequence[Sequence[Any]],
    gt_item_ids: Sequence[Any],
    *,
    ks: Iterable[int] = (5, 10),
) -> Dict[str, float]:
    """Compute HR@k, NDCG@k, MRR@k from ranked lists.

    Parameters
    ----------
    topk_item_ids: per-user ranked item ids (length >= max(k))
    gt_item_ids:   per-user ground-truth item id
    ks:            the list of k to evaluate (e.g., (5,10))
    """
    ks = tuple(sorted({int(k) for k in ks}))
    n = min(len(topk_item_ids), len(gt_item_ids))

    # Initialize accumulators
    HR = {k: 0.0 for k in ks}
    NDCG = {k: 0.0 for k in ks}
    MRR = {k: 0.0 for k in ks}

    for i in range(n):
        ranked = list(topk_item_ids[i])
        gt = gt_item_ids[i]
        # Find first position of gt (1-based)
        try:
            pos = ranked.index(gt) + 1
        except ValueError:
            pos = None
        for k in ks:
            if pos is not None and pos <= k:
                HR[k] += 1.0
                NDCG[k] += _dcg_at_k(pos)  # IDCG@k = 1 for single-relevant
                # MRR uses reciprocal of first relevant rank
                MRR[k] += 1.0 / float(pos)

    out: Dict[str, float] = {}
    denom = float(n) if n > 0 else 1.0
    for k in ks:
        out[f"HR@{k}"] = HR[k] / denom
        out[f"NDCG@{k}"] = NDCG[k] / denom
        out[f"MRR@{k}"] = MRR[k] / denom
    return out


# -----------------------------------------------------------------------------
# Exposure rate (ER)
# -----------------------------------------------------------------------------
def exposure_rate(
    topk_item_ids: Sequence[Sequence[Any]],
    *,
    targets: Iterable[Any] | None,
    k: int = 10,
    mode: str = "user",
) -> float:
    """Compute exposure rate for a set of target items.

    Two modes:
    - mode='user': fraction of users whose top-k contains any target item
    - mode='position': fraction of top-k positions occupied by targets
    """
    targets_set = set(str(t) for t in (targets or []))
    if not targets_set:
        return 0.0
    n_users = len(topk_item_ids)
    if n_users == 0:
        return 0.0

    if mode == "position":
        cnt = 0
        total = 0
        for ranked in topk_item_ids:
            cut = list(ranked)[:k]
            total += len(cut)
            cnt += sum(1 for it in cut if str(it) in targets_set)
        return float(cnt) / float(total or 1)
    else:  # user coverage
        hit = 0
        for ranked in topk_item_ids:
            cut = list(ranked)[:k]
            if any(str(it) in targets_set for it in cut):
                hit += 1
        return float(hit) / float(n_users)


# -----------------------------------------------------------------------------
# Popularity and re-ranking helpers
# -----------------------------------------------------------------------------
def load_popularity_counts(seq_path: str) -> Dict[str, int]:
    """Load item popularity counts from a whitespace-separated sequential file.

    Expected format per line: `<user_id> <item_id> <item_id> ...`.
    """
    pop: Dict[str, int] = {}
    try:
        with open(seq_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                for it in parts[1:]:
                    pop[it] = pop.get(it, 0) + 1
    except Exception:
        pass
    return pop


def rerank_discount_compensate(
    scores: Mapping[Any, float] | Sequence[float],
    item_ids: Sequence[Any] | None,
    pop_map: Mapping[Any, int],
    *,
    alpha: float = 0.8,
    eta: float = 0.05,
) -> Tuple[List[Any], List[float]]:
    """Apply (a) popularity discount and (b) long-tail compensation.

    - scores: per-item score array or dict
    - item_ids: optional list to define the order; if None and scores is a dict,
      item_ids is inferred as scores.keys()
    - pop_map: mapping item_id -> user interaction count
    Returns a pair (sorted_item_ids, sorted_scores) after re-ranking.
    """
    if isinstance(scores, dict):
        if item_ids is None:
            item_ids = list(scores.keys())
        base = {str(k): float(v) for k, v in scores.items()}
    else:
        assert item_ids is not None, "item_ids is required when scores is a sequence"
        base = {str(it): float(sc) for it, sc in zip(item_ids, scores)}

    def _discount(it: str, s: float) -> float:
        n = float(pop_map.get(it, 0))
        return s / (1.0 + alpha / math.sqrt(max(1.0, n)))

    def _compensate(it: str) -> float:
        # Simple inverse-log popularity: lower pop → higher expo score
        n = float(pop_map.get(it, 0))
        return 1.0 / (1.0 + math.log1p(max(0.0, n)))

    reranked = []
    for it, sc in base.items():
        sc1 = _discount(it, sc)
        sc2 = sc1 + eta * _compensate(it)
        reranked.append((it, sc2))

    reranked.sort(key=lambda x: x[1], reverse=True)
    items = [it for it, _ in reranked]
    vals = [v for _, v in reranked]
    return items, vals


__all__ = [
    "seed_everything",
    "metrics_from_ranked",
    "exposure_rate",
    "load_popularity_counts",
    "rerank_discount_compensate",
]

