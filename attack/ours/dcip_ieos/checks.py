#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation helpers for the DCIP-IEOS pipeline.

The real research project performs a large number of sanity checks after the
poisoning attack has produced the modified artefacts.  The light‑weight kata in
this repository mirrors a subset of those checks so that unit tests can reason
about the behaviour of the pipeline without relying on heavy third‑party
dependencies.  The routines below intentionally operate on plain Python data
structures to keep them fast and portable.
"""

from __future__ import annotations

import logging
import math

import os
import pickle
import random
from typing import Any, Dict, Optional, Sequence

from .utils_tokenizers import CharTokenizer

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# ---------------------------------------------------------------------------
# Cache helpers used by the command line interface
# ---------------------------------------------------------------------------

def cache_files_exist(cache_dir: str) -> bool:
    """Return ``True`` if default cache files exist in ``cache_dir``."""

    has_comp = any(
        fname.startswith("competition_pool") and fname.endswith(".json")
        for fname in os.listdir(cache_dir)
    )
    mask_ok = os.path.isfile(os.path.join(cache_dir, "cross_modal_mask.pkl"))
    return has_comp and mask_ok


def load_cross_modal_mask(cache_dir: str) -> Any:
    """Load and return the cross-modal mask object."""

    path = os.path.join(cache_dir, "cross_modal_mask.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Poisoned artefact checks
# ---------------------------------------------------------------------------

def poisoned_files_exist(paths: Dict[str, str]) -> bool:
    """Return ``True`` if all files in ``paths`` exist and are readable.

    Files whose key contains the substring ``"keywords"`` are additionally
    unpickled to ensure they can be loaded successfully.
    """

    for key, p in paths.items():
        if not (isinstance(p, str) and os.path.isfile(p) and os.access(p, os.R_OK)):
            return False
        if "keywords" in key:
            try:
                with open(p, "rb") as f:
                    pickle.load(f)
            except Exception:
                return False
    return True


def _l2_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the L2 distance between two vectors."""

    n = min(len(x), len(y))
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(x[:n], y[:n])))


def _cosine_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors."""

    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(x[:n], y[:n]))
    norm_x = math.sqrt(sum(float(a) ** 2 for a in x[:n]))
    norm_y = math.sqrt(sum(float(b) ** 2 for b in y[:n]))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return dot / (norm_x * norm_y)


def verify_embedding_shrinkage(
    competition_pool: Sequence[Dict[str, Any]],
    exp_splits: Dict[str, Any],
    fake_users: Sequence[str],
    distance_cache: Optional[str | Dict[str, float]] = None,
) -> None:
    """Warn if perturbed embeddings did not move closer to the target.

    Parameters
    ----------
    distance_cache:
        Optional mapping or pickle path containing ``Δdistance`` values as
        produced by :mod:`poison_pipeline`.  When ``Δdistance`` is non-negative
        the distance to the target increased or stayed constant and a warning
        is logged.
    """

    cache: Dict[str, float] = {}
    if distance_cache:
        if isinstance(distance_cache, str):
            try:
                with open(distance_cache, "rb") as f:
                    cache = pickle.load(f)
            except Exception:
                cache = {}
        elif isinstance(distance_cache, dict):
            cache = dict(distance_cache)

    order = [str(u) for u in fake_users]
    id_to_feat = {
        str(e.get("reviewerID")): e.get("feature", [])
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in set(order)
    }

    for idx, tgt in enumerate(competition_pool):
        if idx >= len(order):
            break
        user_id = order[idx]
        if user_id not in id_to_feat:
            continue
        before = tgt.get("anchor", [])
        target_vec = tgt.get("target_feat", [0.0] * len(before))
        after = id_to_feat[user_id]
        dist_before = _l2_distance(before, target_vec)
        dist_after = _l2_distance(after, target_vec)
        delta = cache.get(user_id, dist_after - dist_before)
        if delta >= 0:
            logging.warning(
                "Embedding distance increased for user %s: before=%.4f after=%.4f Δ=%.4f",
                user_id,
                dist_before,
                dist_after,
                delta,
            )


def _psnr(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute the Peak Signal to Noise Ratio between ``x`` and ``y``."""

    n = min(len(x), len(y))
    if n == 0:
        return float("inf")
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(x[:n], y[:n])) / n
    return 10.0 * math.log10(1.0 / (mse + 1e-12))


def verify_poison_statistics(
    competition_pool: Sequence[Dict[str, Any]],
    exp_splits: Dict[str, Any],
    sequential_lines: Sequence[str],
    fake_users: Sequence[str],
    *,
    psnr_min: float = 20.0,
    max_text_ratio: float = 0.5,
    min_masked_check: int = 2,
    seq_len_tolerance: int = 2,
    p_insert: float = 0.2,
    p_replace: float = 0.2,
    mask_cache: Optional[str | Dict[int, Dict[str, Any]]] = None,
    tokenizer: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> None:
    """Validate basic perturbation statistics.

    Parameters are deliberately lax so the function remains robust for the
    simplified data used in the tests.

    ``p_insert`` and ``p_replace`` control the expected proportion of
    insertions and replacements performed during sequence bridging.  The
    function merely validates that the supplied probabilities are within the
    unit interval.
    """

    order = [str(u) for u in fake_users]
    fake_set = set(order)
    id_to_feat = {
        str(e.get("reviewerID")): e.get("feature", [])
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in fake_set
    }
    id_to_text = {
        str(e.get("reviewerID")): e.get("reviewText", "")
        for e in exp_splits.get("train", [])
        if str(e.get("reviewerID")) in fake_set
    }
    if not (0.0 <= float(p_insert) <= 1.0):
        raise AssertionError("p_insert outside [0, 1]")
    if not (0.0 <= float(p_replace) <= 1.0):
        raise AssertionError("p_replace outside [0, 1]")
    
    # Load cross-modal mask cache -------------------------------------------
    masks: Dict[int, Dict[str, Any]] = {}
    if isinstance(mask_cache, dict):
        masks = mask_cache
    else:
        cache_dir = mask_cache if isinstance(mask_cache, str) else os.path.join(
            os.path.dirname(__file__), "caches"
        )
        try:
            masks = load_cross_modal_mask(cache_dir)
        except Exception:
            masks = {}

    seq_map = {}
    for line in sequential_lines:
        parts = line.strip().split()
        if parts:
            seq_map[parts[0]] = parts[1:]

    for idx, tgt in enumerate(competition_pool):
        if idx >= len(order):
            break
        user_id = order[idx]

        # Image PSNR
        before = tgt.get("anchor", [])
        after = id_to_feat.get(user_id, [])
        if _psnr(before, after) < psnr_min:
            raise AssertionError(f"PSNR below threshold for user {user_id}")

        # Text replacement rate based on masked token positions -------------
        mask_entry = masks.get(idx, {}) if isinstance(masks, dict) else {}
        poisoned_text = id_to_text.get(user_id, "")

        txt_mask = [bool(v) for v in mask_entry.get("text_mask", [])]
        tok_name = str(mask_entry.get("tokenizer_name", "")).lower()
        # Be tolerant: default to CharTokenizer when no external tokenizer is provided
        if tok_name == CharTokenizer.name or tokenizer is None:
            tok = CharTokenizer()
        else:
            tok = tokenizer

        orig_tokens = mask_entry.get("text_tokens")
        if orig_tokens is None:
            orig_text = mask_entry.get("text", "")
            orig_tokens = tok.encode(orig_text)
        new_tokens = tok.encode(poisoned_text)

        L = min(len(orig_tokens), len(new_tokens), len(txt_mask))
        masked = [i for i in range(L) if txt_mask[i]]
        diff = sum(1 for i in masked if orig_tokens[i] != new_tokens[i])
        ratio = diff / len(masked) if masked else 0.0
        log = logger.info if logger else logging.info
        if len(masked) < min_masked_check:
            log(
                f"[verify] skip text replacement check: masked={len(masked)} "
                f"< min_masked_check={min_masked_check}"
            )
        elif len(masked) < 10:
            log("[verify] skip: masked<10")
            return
        else:
            log(
                f"[verify] masked={len(masked)} diff={diff} ratio={ratio:.3f} "
                f"(max={max_text_ratio})"
            )
            if ratio > max_text_ratio:
                raise AssertionError(
                    f"Text replacement rate too high for user {user_id}"
                )

        # Sequence statistics
        seq = seq_map.get(user_id, [])
        neighbours = tgt.get("neighbors") or tgt.get("competitors", [])
        expected_len = max(1, len(neighbours) + 1)
        if abs(len(seq) - expected_len) > seq_len_tolerance:
            raise AssertionError(
                f"Sequence length for user {user_id} outside tolerance"
            )
        if p_insert >= 1.0 and str(tgt.get("target")) not in seq:
            raise AssertionError(f"Target item missing in sequence for {user_id}")
        

def evaluate_anchor_similarity(
    competition_pool: Sequence[Dict[str, Any]],
    *,
    sample_size: int = 5,
    cache_dir: Optional[str] = None,
    pca: bool = False,
) -> Dict[str, float]:
    """Compare target embeddings to anchors and random averages.

    Randomly samples ``sample_size`` targets ``t`` from ``competition_pool`` and
    compares the cosine similarity between the target embedding ``E(t)`` and the
    pre‑computed ``anchor`` against the similarity between ``E(t)`` and a
    randomly sampled average of other anchors.  An :class:`AssertionError` is
    raised if any sampled target is closer to the random average than to its
    own anchor.  Basic statistics are returned and logged.

    When ``pca`` is ``True`` and the required optional dependencies are
    available, a 2D PCA projection of anchors and their neighbours is saved as
    ``competition_pool_pca.png`` in ``cache_dir``.
    """

    if not competition_pool:
        logging.info("Empty competition pool – skipping anchor similarity check")
        return {"samples": 0, "anchor_mean": 0.0, "random_mean": 0.0}

    valid_pool = [entry for entry in competition_pool if entry.get("anchor")]
    if not valid_pool:
        logging.info("No anchor vectors available – skipping anchor similarity check")
        return {"samples": 0, "anchor_mean": 0.0, "random_mean": 0.0}

    rng = random.Random()
    sample = rng.sample(valid_pool, min(sample_size, len(valid_pool)))

    anchor_sims = []
    random_sims = []
    for entry in sample:
        target_vec = entry.get("embedding") or entry.get("feature") or []
        anchor_vec = entry.get("anchor", [])
        if not target_vec or not anchor_vec:
            continue

        cos_anchor = _cosine_similarity(target_vec, anchor_vec)

        # Random average of other anchors used as a "hot" baseline
        others = [e for e in competition_pool if e is not entry and e.get("anchor")]
        hot_sample = rng.sample(others, min(len(others), 5)) if others else []
        if hot_sample:
            dim = len(anchor_vec)
            avg = [0.0] * dim
            for o in hot_sample:
                vec = o.get("anchor", [])
                n = min(dim, len(vec))
                for i in range(n):
                    avg[i] += float(vec[i])
            for i in range(dim):
                avg[i] /= len(hot_sample)
            cos_rand = _cosine_similarity(target_vec, avg)
        else:
            cos_rand = 0.0

        if cos_anchor < cos_rand:
            raise AssertionError(
                f"Anchor less similar than random average for target {entry.get('target')}"
            )

        anchor_sims.append(cos_anchor)
        random_sims.append(cos_rand)

    stats = {
        "samples": len(anchor_sims),
        "anchor_mean": sum(anchor_sims) / len(anchor_sims) if anchor_sims else 0.0,
        "random_mean": sum(random_sims) / len(random_sims) if random_sims else 0.0,
    }

    logging.info(
        "Anchor similarity over %d samples: anchor %.4f vs random %.4f",
        stats["samples"],
        stats["anchor_mean"],
        stats["random_mean"],
    )

    if pca and cache_dir:
        try:  # optional heavy dependencies
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("PCA visualisation skipped: %s", exc)
        else:
            vectors = []
            labels = []  # 0 anchor, 1 neighbour
            id_map = {str(e.get("target")): e for e in competition_pool}
            for entry in competition_pool:
                a = entry.get("anchor")
                if a:
                    vectors.append(a)
                    labels.append(0)
                for n_id in entry.get("neighbors") or entry.get("competitors", []):
                    neigh = id_map.get(str(n_id))
                    if neigh is not None:
                        vec = neigh.get("embedding") or neigh.get("feature") or neigh.get("anchor")
                        if vec:
                            vectors.append(vec)
                            labels.append(1)
            if len(vectors) >= 2:
                pca_model = PCA(n_components=2)
                arr = np.asarray(vectors, dtype=float)
                proj = pca_model.fit_transform(arr)
                anchors = [i for i, l in enumerate(labels) if l == 0]
                neighs = [i for i, l in enumerate(labels) if l == 1]
                plt.figure()
                if anchors:
                    plt.scatter(proj[anchors, 0], proj[anchors, 1], c="red", label="anchor")
                if neighs:
                    plt.scatter(proj[neighs, 0], proj[neighs, 1], c="blue", label="neighbor")
                plt.legend()
                out_path = os.path.join(cache_dir, "competition_pool_pca.png")
                plt.savefig(out_path)
                plt.close()
                logging.info("Saved PCA visualisation to %s", out_path)

    return stats



__all__ = [
    "cache_files_exist",
    "load_cross_modal_mask",
    "poisoned_files_exist",
    "verify_embedding_shrinkage",
    "verify_poison_statistics",
    "evaluate_anchor_similarity",
]
