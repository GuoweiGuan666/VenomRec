"""Loss utilities for interactive poisoning."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    vec_a = np.asarray(list(a), dtype=float)
    vec_b = np.asarray(list(b), dtype=float)
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def info_nce_loss(
    target: Iterable[float],
    positive: Iterable[float],
    negatives: Iterable[Iterable[float]] | None = None,
    temperature: float = 0.1,
) -> float:
    """Compute a simple InfoNCE loss for alignment monitoring."""

    negs = list(negatives or [])
    sims = [cosine_similarity(target, positive)]
    sims.extend(cosine_similarity(target, neg) for neg in negs)
    logits = np.asarray(sims, dtype=float) / max(temperature, 1e-6)
    logits = logits - logits.max()  # numerical stability
    exp_scores = np.exp(logits)
    numerator = exp_scores[0]
    denominator = exp_scores.sum()
    if denominator <= 1e-12:
        return float("inf")
    loss = -math.log(numerator / denominator)
    return float(loss)


__all__ = ["cosine_similarity", "info_nce_loss"]
