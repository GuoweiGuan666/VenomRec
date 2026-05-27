"""Compat shim re-exporting perturbation helpers after the refactor."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Sequence

from .image_perturbation import ImagePerturber, masked_pgd_image
from .text_perturbation import (
    DOMAIN_SYNONYMS,
    TextPerturber,
    guided_text_paraphrase,
    is_replaceable_token,
    normalize_token,
)


def bridge_sequences(
    seq: Sequence[Dict[str, Any]],
    target_item: Any,
    pool_items: Sequence[Any],
    p_insert: float,
    p_replace: float,
    stats_ref: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(seq, list):
        seq = list(seq)
    seq = [dict(it) for it in seq]

    length_target = int(stats_ref.get("length", len(seq)))
    mean_dt = stats_ref.get("mean_dt")
    if mean_dt is None and len(seq) >= 2:
        mean_dt = (seq[-1].get("timestamp", 0) - seq[0].get("timestamp", 0)) / max(len(seq) - 1, 1)
    if mean_dt is None:
        mean_dt = 1.0

    changed = 0
    for item in seq:
        if random.random() < float(p_replace):
            item["item"] = random.choice(pool_items)
            changed += 1

    while len(seq) < length_target:
        idx = random.randint(0, len(seq)) if seq else 0
        val = target_item if random.random() < float(p_insert) else random.choice(pool_items)
        ts = seq[0]["timestamp"] if seq else 0
        ts += idx * mean_dt
        seq.insert(idx, {"item": val, "timestamp": ts})
        changed += 1

    if len(seq) > length_target:
        seq = seq[:length_target]

    start_ts = seq[0]["timestamp"] if seq else 0
    for i, item in enumerate(seq):
        item["timestamp"] = start_ts + i * mean_dt

    coverage = changed / max(len(seq), 1)
    logging.info("bridge_sequences coverage %.2f%%", coverage * 100)
    return seq


__all__ = [
    "ImagePerturber",
    "TextPerturber",
    "masked_pgd_image",
    "bridge_sequences",
    "DOMAIN_SYNONYMS",
    "guided_text_paraphrase",
    "normalize_token",
    "is_replaceable_token",
]
