#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting utilities for DCIP-IEOS attack analysis.

All functions are best-effort: if optional deps (matplotlib, numpy, sklearn)
are missing, the function will no-op and log a short message instead of
crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple


def _try_imports():
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        np = None  # type: ignore
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        plt = None  # type: ignore
    return np, plt


def _ensure_out(out_dir: str) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass


def _load_sequences(path: str) -> List[List[str]]:
    seqs: List[List[str]] = []
    if not (path and os.path.isfile(path)):
        return seqs
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    seqs.append(parts[1:])
    except Exception:
        return []
    return seqs


def plot_pre_post_distributions(pre_seq: str, post_seq: str, out_dir: str, *, title_prefix: str = "") -> List[str]:
    """Overlay distribution plots before/after poisoning.

    Produces 2 figures under ``out_dir``:
      - ``seq_len_pre_post.png`` – sequence length distribution
      - ``item_pop_pre_post.png`` – item popularity distribution
    Returns the list of file paths created (may be empty on failure).
    """

    np, plt = _try_imports()
    if plt is None:
        logging.info("[viz] matplotlib not available; skip pre/post distribution plots")
        return []

    _ensure_out(out_dir)
    paths: List[str] = []

    pre = _load_sequences(pre_seq)
    post = _load_sequences(post_seq)
    if not pre or not post:
        logging.info("[viz] missing sequences for pre/post plots: pre=%d post=%d", len(pre), len(post))
    # 1) Sequence length distribution
    try:
        pre_lens = [len(s) for s in pre]
        post_lens = [len(s) for s in post]
        if not pre_lens or not post_lens:
            raise ValueError("empty lengths")
        fig = plt.figure()
        kwargs = dict(alpha=0.5, bins=30, density=True)
        plt.hist(pre_lens, label="pre", color="#1f77b4", **kwargs)
        plt.hist(post_lens, label="post", color="#2ca02c", **kwargs)
        plt.xlabel("sequence length")
        plt.ylabel("density")
        ttl = (title_prefix + " ").strip() + "seq length: pre vs post"
        plt.title(ttl)
        plt.legend()
        out1 = os.path.join(out_dir, "seq_len_pre_post.png")
        fig.tight_layout()
        fig.savefig(out1)
        plt.close(fig)
        paths.append(out1)
    except Exception as exc:
        logging.info("[viz] skip seq length plot: %s", exc)

    # 2) Item popularity distribution (via counts)
    try:
        def _count_items(seqs: List[List[str]]) -> Dict[str, int]:
            cnt: Dict[str, int] = {}
            for s in seqs:
                for it in s:
                    cnt[it] = cnt.get(it, 0) + 1
            return cnt
        pre_cnt = _count_items(pre)
        post_cnt = _count_items(post)
        if not pre_cnt or not post_cnt:
            raise ValueError("empty item counts")
        # To make shapes comparable, align on union of items and fill 0s
        all_items = set(pre_cnt) | set(post_cnt)
        pre_vals = [pre_cnt.get(i, 0) for i in all_items]
        post_vals = [post_cnt.get(i, 0) for i in all_items]
        if np is not None:
            pre_vals = list(np.asarray(pre_vals, dtype=float))  # type: ignore
            post_vals = list(np.asarray(post_vals, dtype=float))  # type: ignore
        fig = plt.figure()
        kwargs = dict(alpha=0.5, bins=50, density=True)
        plt.hist(pre_vals, label="pre", color="#1f77b4", **kwargs)
        plt.hist(post_vals, label="post", color="#2ca02c", **kwargs)
        plt.xlabel("item count")
        plt.ylabel("density")
        ttl = (title_prefix + " ").strip() + "item popularity: pre vs post"
        plt.title(ttl)
        plt.legend()
        out2 = os.path.join(out_dir, "item_pop_pre_post.png")
        fig.tight_layout()
        fig.savefig(out2)
        plt.close(fig)
        paths.append(out2)
    except Exception as exc:
        logging.info("[viz] skip item popularity plot: %s", exc)

    return paths


def plot_target_pop_trend(metrics_pkl: str, out_dir: str) -> str | None:
    """Plot mean ± std trend of similarities to anchor/target across rounds.

    Loads ``round_metrics*.pkl`` produced by the pipeline and aggregates
    per-round ``sim_anchor`` and ``sim_target`` across targets. Saves
    ``trend_target_vs_pop.png`` in ``out_dir``.
    """

    np, plt = _try_imports()
    if plt is None:
        logging.info("[viz] matplotlib not available; skip trend plot")
        return None

    _ensure_out(out_dir)

    try:
        with open(metrics_pkl, "rb") as f:
            log: Dict[str, Any] = pickle.load(f)
    except Exception as exc:
        logging.info("[viz] failed to load metrics %s: %s", metrics_pkl, exc)
        return None

    # Collect per-round arrays
    per_round_anchor: Dict[int, List[float]] = {}
    per_round_target: Dict[int, List[float]] = {}
    for v in log.values():
        rounds = v.get("rounds", [])
        for rec in rounds:
            r = int(rec.get("round", 0))
            if "sim_anchor" in rec:
                per_round_anchor.setdefault(r, []).append(float(rec.get("sim_anchor", 0.0)))
            if "sim_target" in rec:
                per_round_target.setdefault(r, []).append(float(rec.get("sim_target", 0.0)))

    if not per_round_anchor and not per_round_target:
        logging.info("[viz] metrics missing sim_* keys; skip trend plot")
        return None

    max_r = 0
    if per_round_anchor:
        max_r = max(max_r, max(per_round_anchor))
    if per_round_target:
        max_r = max(max_r, max(per_round_target))

    xs = list(range(0, max_r + 1))

    def _agg(vals: Dict[int, List[float]]) -> Tuple[List[float], List[float]]:
        means: List[float] = []
        stds: List[float] = []
        for i in xs:
            arr = vals.get(i, [])
            if not arr:
                means.append(0.0)
                stds.append(0.0)
            else:
                if np is not None:
                    a = np.asarray(arr, dtype=float)
                    means.append(float(a.mean()))
                    stds.append(float(a.std()))
                else:
                    m = sum(arr) / len(arr)
                    v = sum((x - m) ** 2 for x in arr) / max(1, len(arr) - 1)
                    means.append(m)
                    stds.append(v ** 0.5)
        return means, stds

    m_anchor, s_anchor = _agg(per_round_anchor)
    m_target, s_target = _agg(per_round_target)

    fig = plt.figure()
    # Anchor trend
    plt.plot(xs, m_anchor, color="#ff7f0e", label="sim(target→anchor)")
    plt.fill_between(xs, [m - s for m, s in zip(m_anchor, s_anchor)], [m + s for m, s in zip(m_anchor, s_anchor)], color="#ff7f0e", alpha=0.2)
    # Target trend (optional)
    if any(m_target):
        plt.plot(xs, m_target, color="#1f77b4", label="sim(target→target_feat)")
        plt.fill_between(xs, [m - s for m, s in zip(m_target, s_target)], [m + s for m, s in zip(m_target, s_target)], color="#1f77b4", alpha=0.15)
    plt.xlabel("epoch (inner round)")
    plt.ylabel("cosine similarity")
    plt.title("Target vs Popular (anchor) similarity across epochs")
    plt.legend()
    outp = os.path.join(out_dir, "trend_target_vs_pop.png")
    fig.tight_layout()
    fig.savefig(outp)
    plt.close(fig)
    return outp


def plot_embedding_pca_pre_post(comp_pool_json: str, exp_splits_pkl: str, out_dir: str) -> str | None:
    """2D PCA of target anchors (pre) vs fake-user features (post).

    Saves ``embedding_pca_pre_post.png`` in ``out_dir``. This is optional and
    will be skipped if required deps are missing or data is insufficient.
    """

    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.info("[viz] skip PCA plot: matplotlib/numpy unavailable: %s", exc)
        return None

    try:
        with open(comp_pool_json, "r", encoding="utf-8") as f:
            comp = json.load(f)
    except Exception as exc:
        logging.info("[viz] skip PCA plot: cannot read comp pool: %s", exc)
        return None
    try:
        with open(exp_splits_pkl, "rb") as f:
            exp = pickle.load(f)
    except Exception as exc:
        logging.info("[viz] skip PCA plot: cannot read exp_splits: %s", exc)
        return None

    # Collect anchors and fake-user features in the same order
    anchors: List[List[float]] = []
    fake_feats: List[List[float]] = []
    fake_uids: List[str] = []

    # In the toy pipeline, fake users are appended; try to order by target index
    # We infer fake user ids by scanning exp_splits train entries that have reviewerName starting with 'fake_'
    train = list(exp.get("train", [])) if isinstance(exp, dict) else []
    for entry in comp:
        a = entry.get("anchor", [])
        if a:
            anchors.append([float(v) for v in a])
    for e in train:
        name = str(e.get("reviewerName", ""))
        if name.startswith("fake_"):
            fake_uids.append(str(e.get("reviewerID")))
            feat = e.get("feature", [])
            if feat:
                fake_feats.append([float(v) for v in feat])

    if len(anchors) < 2 or len(fake_feats) < 2:
        logging.info("[viz] insufficient data for PCA plot")
        return None

    try:
        from sklearn.decomposition import PCA  # type: ignore
        use_pca = True
    except Exception:
        use_pca = False

    A = np.asarray(anchors, dtype=float)
    B = np.asarray(fake_feats, dtype=float)
    dim = min(A.shape[1], B.shape[1])
    A = A[:, :dim]
    B = B[:, :dim]
    X = np.vstack([A, B])
    if X.shape[1] < 2:
        logging.info("[viz] insufficient dims for 2D PCA plot")
        return None
    Z = None
    if use_pca:
        try:
            Z = PCA(n_components=2).fit_transform(X)
        except Exception:
            Z = X[:, :2]
    else:
        Z = X[:, :2]

    nA = A.shape[0]
    ZA = Z[:nA]
    ZB = Z[nA:]
    _ensure_out(out_dir)
    fig = plt.figure()
    plt.scatter(ZA[:, 0], ZA[:, 1], c="#d62728", label="pre: anchors", alpha=0.7)
    plt.scatter(ZB[:, 0], ZB[:, 1], c="#2ca02c", label="post: fake features", alpha=0.6)
    plt.title("Embedding distribution: pre (anchor) vs post (fake features)")
    plt.legend()
    outp = os.path.join(out_dir, "embedding_pca_pre_post.png")
    fig.tight_layout()
    fig.savefig(outp)
    plt.close(fig)
    return outp


__all__ = [
    "plot_pre_post_distributions",
    "plot_target_pop_trend",
    "plot_embedding_pca_pre_post",
]

