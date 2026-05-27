#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fidelity‑Constrained (teacher‑free) multi‑objective loss utilities.

This module provides lightweight helpers used inside the DCIP×IEOS poisoning
pipeline to compute a composite objective J that balances:

- Alignment towards the target representation (drives ER)
- Anchor fidelity (soft constraint against drifting too far from the anchor)
- Text naturalness proxy (soft penalty on replacement ratio and length drift)
- Distribution alignment (sequence length and item popularity histograms)

All functions are numeric and self‑contained to keep the pipeline portable.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any, Optional
import math
import os

# ---------------------------------------------------------------------------
# Basic numeric helpers
# ---------------------------------------------------------------------------

def _to_float_list(x: Iterable[float] | None) -> List[float]:
    if x is None:
        return []
    try:
        return [float(v) for v in x]
    except Exception:
        return []


def _l2_norm(x: Iterable[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in x))


def _l2_normalize(x: Iterable[float]) -> List[float]:
    v = _to_float_list(x)
    n = _l2_norm(v)
    if n <= 1e-12:
        return [0.0 for _ in v]
    inv = 1.0 / n
    return [t * inv for t in v]


def _squared_l2(a: Iterable[float], b: Iterable[float]) -> float:
    aa = _to_float_list(a)
    bb = _to_float_list(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return 0.0
    return sum((aa[i] - bb[i]) * (aa[i] - bb[i]) for i in range(n))


# ---------------------------------------------------------------------------
# Fidelity‑Constrained losses (teacher‑free)
# ---------------------------------------------------------------------------

def fc_compute_L_align(feat_curr: Iterable[float], feat_target: Iterable[float]) -> float:
    """Alignment term: ||norm(curr) − norm(target)||^2."""
    a = _l2_normalize(feat_curr)
    b = _l2_normalize(feat_target)
    return _squared_l2(a, b)


def fc_compute_L_anchor(
    feat_curr: Iterable[float],
    feat_anchor: Iterable[float],
    *,
    eps_ratio: float = 0.10,
    eps_min: float = 1e-3,
    feat_target: Optional[Iterable[float]] = None,
) -> float:
    """Anchor fidelity (hinge): max(0, ||norm(curr)−norm(anchor)|| − eps)^2.

    eps is derived from the anchor–target distance if ``feat_target`` is
    provided, otherwise from ``eps_min`` alone.
    """
    a = _l2_normalize(feat_curr)
    c = _l2_normalize(feat_anchor)
    dist = math.sqrt(_squared_l2(a, c))
    if feat_target is not None:
        t = _l2_normalize(feat_target)
        base = math.sqrt(_squared_l2(t, c))
        eps = max(eps_min, eps_ratio * base)
    else:
        eps = eps_min
    margin = max(0.0, dist - eps)
    return margin * margin


def fc_compute_L_text(
    replace_ratio_step: float,
    ratio_max: float,
    delta_len_norm: float,
    *,
    lambda_len: float = 0.1,
    ratio_nonkw: float = 0.0,
    lambda_nonkw: float = 0.0,
) -> float:
    """Text naturalness proxy combining replacement ratio and length drift."""
    tau = max(1e-8, float(ratio_max) / 2.0)
    rr = max(0.0, float(replace_ratio_step))
    dlen = max(0.0, float(delta_len_norm))
    rnk = max(0.0, float(ratio_nonkw))
    return (rr / tau) ** 2 + lambda_len * (dlen ** 2) + lambda_nonkw * rnk


def fc_compute_J(weights: Dict[str, float], parts: Dict[str, float]) -> float:
    """Compose the final J from individual loss parts using provided weights.

    Expected keys in ``parts``: 'L_align', 'L_anchor', 'L_text', 'L_stat'.
    Missing keys default to 0.0 to keep the function robust.
    """
    def g(name: str) -> float:
        return float(parts.get(name, 0.0))

    return (
        float(weights.get('w_align', 1.0)) * g('L_align')
        + float(weights.get('w_anchor', 0.0)) * g('L_anchor')
        + float(weights.get('w_text', 0.0)) * g('L_text')
        + float(weights.get('w_stat', 0.0)) * g('L_stat')
    )


# ---------------------------------------------------------------------------
# Distribution alignment (length + popularity histograms)
# ---------------------------------------------------------------------------

class FidelityStatsKeeper:
    """Maintain real/fake histograms and compute length/pop KL penalties.

    - Real histograms are built from the clean ``sequential_data.txt``.
    - Fake histograms are updated incrementally with newly generated sequences.
    - Popularity is derived from per‑item frequencies and binned into ``pop_bins``.
    """

    def __init__(self, real_seq_path: str, pop_bins: int = 10, smooth: float = 1e-8) -> None:
        self.pop_bins = max(2, int(pop_bins))
        self.smooth = float(smooth)
        self.real_len_hist: List[float] = []
        self.real_pop_hist: List[float] = []
        self.fake_len_hist: List[float] = [0.0 for _ in range(self.pop_bins)]
        self.fake_pop_hist: List[float] = [0.0 for _ in range(self.pop_bins)]
        self._pop_thresholds: List[int] = []  # item frequency thresholds per bin
        self._item_freq: Dict[str, int] = {}
        try:
            self._build_real_hists(real_seq_path)
        except Exception:
            # Keep empty/fallback hists – L_stat will be 0.0
            self.real_len_hist = [1.0] + [0.0 for _ in range(self.pop_bins - 1)]
            self.real_pop_hist = [1.0] + [0.0 for _ in range(self.pop_bins - 1)]

    # ---- public API -----------------------------------------------------
    def update_fake(self, seq_lengths: List[int], item_ids: List[str]) -> None:
        for L in seq_lengths:
            bin_idx = self._len_to_bin(L)
            self.fake_len_hist[bin_idx] += 1.0
        for it in item_ids:
            b = self._item_to_pop_bin(str(it))
            self.fake_pop_hist[b] += 1.0

    def compute_L_stat(self) -> Tuple[float, float, float]:
        """Return (kl_len, kl_pop, L_stat=kl_len+kl_pop)."""
        kl_len = self._kl(self.fake_len_hist, self.real_len_hist)
        kl_pop = self._kl(self.fake_pop_hist, self.real_pop_hist)
        return kl_len, kl_pop, (kl_len + kl_pop)

    def compute_L_stat_with_candidate(self, seq_len: int, item_ids: List[str]) -> Tuple[float, float, float]:
        tmp_len = list(self.fake_len_hist)
        tmp_pop = list(self.fake_pop_hist)
        tmp_len[self._len_to_bin(seq_len)] += 1.0
        for it in item_ids:
            tmp_pop[self._item_to_pop_bin(str(it))] += 1.0
        kl_len = self._kl(tmp_len, self.real_len_hist)
        kl_pop = self._kl(tmp_pop, self.real_pop_hist)
        return kl_len, kl_pop, (kl_len + kl_pop)

    # ---- internals ------------------------------------------------------
    def _build_real_hists(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        lengths: List[int] = []
        # count item frequencies
        item_freq: Dict[str, int] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) <= 1:
                    continue
                items = parts[1:]
                lengths.append(len(items))
                for it in items:
                    item_freq[it] = item_freq.get(it, 0) + 1
        self._item_freq = item_freq
        # build popularity thresholds (equal‑frequency bins)
        if item_freq:
            counts = sorted(item_freq.values())
            n = len(counts)
            thresholds = []
            for b in range(1, self.pop_bins):
                idx = int(math.ceil(b * n / self.pop_bins)) - 1
                idx = max(0, min(idx, n - 1))
                thresholds.append(counts[idx])
            self._pop_thresholds = thresholds
        else:
            self._pop_thresholds = [0 for _ in range(self.pop_bins - 1)]
        # length histogram (equal‑width bins based on percentiles)
        if lengths:
            max_len = max(lengths)
        else:
            max_len = 1
        # simple equal‑width bins in [1, max_len]
        self.real_len_hist = [0.0 for _ in range(self.pop_bins)]
        for L in lengths:
            b = self._len_to_bin(L, max_len=max_len)
            self.real_len_hist[b] += 1.0
        s = sum(self.real_len_hist) or 1.0
        self.real_len_hist = [v / s for v in self.real_len_hist]
        # popularity histogram: over all items
        self.real_pop_hist = [0.0 for _ in range(self.pop_bins)]
        for it, c in item_freq.items():
            b = self._count_to_pop_bin(c)
            self.real_pop_hist[b] += 1.0
        s = sum(self.real_pop_hist) or 1.0
        self.real_pop_hist = [v / s for v in self.real_pop_hist]

    def _len_to_bin(self, L: int, *, max_len: Optional[int] = None) -> int:
        L = int(max(1, L))
        if max_len is None:
            # derive from real histogram span: assume bins equally split up to max observed
            # When not available, fallback to pop_bins as a proxy range
            max_len = self.pop_bins
        w = max(1, int(math.ceil(max_len / self.pop_bins)))
        idx = (L - 1) // w
        return max(0, min(self.pop_bins - 1, idx))

    def _count_to_pop_bin(self, c: int) -> int:
        # map absolute frequency to bin using thresholds
        for i, thr in enumerate(self._pop_thresholds):
            if c <= thr:
                return i
        return self.pop_bins - 1

    def _item_to_pop_bin(self, it: str) -> int:
        c = int(self._item_freq.get(it, 0))
        return self._count_to_pop_bin(c)

    def _kl(self, p_counts: List[float], q_probs: List[float]) -> float:
        # convert counts to probabilities with smoothing
        p = [max(self.smooth, v) for v in p_counts]
        sp = sum(p)
        p = [v / sp for v in p]
        q = [max(self.smooth, v) for v in q_probs]
        sq = sum(q)
        q = [v / sq for v in q]
        kl = 0.0
        for pi, qi in zip(p, q):
            kl += pi * math.log(pi / qi)
        return float(kl)

