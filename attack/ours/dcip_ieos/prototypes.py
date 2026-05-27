#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build or load simple popular-center prototypes for IEOS representation loss.

This lightweight helper avoids adding heavy victim dependencies in M1.
It attempts to build a popular item center vector from available metadata.

Priority:
1) If a cached prototype exists, load it.
2) If a high-pop items list exists, average their numeric item embeddings when available.
3) Fallback to averaging anchors from the competition pool at runtime.

The output center is L2-normalized for cosine-friendly geometry.
"""
from __future__ import annotations

import json
import os
import pickle
import math
from typing import Any, Dict, List, Optional


def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _l2_normalize(v: List[float]) -> List[float]:
    n = _l2_norm(v)
    if n <= 1e-12:
        return [0.0 for _ in v]
    inv = 1.0 / n
    return [float(x) * inv for x in v]


def _load_pickle(path: str, default=None):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return default


def _dump_pickle(path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception:
        pass


def _load_high_pop_list(pop_path: str) -> List[str]:
    items: List[str] = []
    try:
        with open(pop_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s:
                    items.append(s)
    except Exception:
        items = []
    return items


def build_or_load_pop_center(split_dir: str,
                             cache_dir: str,
                             feat_root: str,
                             feat_backbone: str,
                             pop_path: Optional[str] = None,
                             fallback_anchors: Optional[List[List[float]]] = None) -> List[float]:
    """Return a normalized popular center vector for IEOS.

    Parameters
    ----------
    split_dir: str
        Path to data/<split> directory.
    cache_dir: str
        Where to store cached prototype vectors.
    pop_path: str
        Optional path to high-pop items list (ASIN or item ids).
    fallback_anchors: List[List[float]]
        Optional list of anchor vectors to average if no pop list is available.
    """
    split_name = os.path.basename(split_dir.rstrip(os.sep))
    cache_path = os.path.join(cache_dir, 'prototypes', split_name, 'c_pop.pkl')
    proto = _load_pickle(cache_path, None)
    if isinstance(proto, list) and proto and isinstance(proto[0], float):
        return _l2_normalize(proto)

    # Attempt true build from POP list + features
    try:
        # discover features dir and datamaps
        # features layout: features/<feat_backbone>/<split>/
        feat_dir = os.path.join(feat_root, feat_backbone, split_name)
        datamaps_path = os.path.join('data', split_name, 'datamaps.json')
        item2id = None
        if os.path.isfile(datamaps_path):
            try:
                with open(datamaps_path, 'r', encoding='utf-8') as f:
                    dm = json.load(f)
                for key in ('item2id', 'asin2id', 'asin_to_id'):
                    if isinstance(dm.get(key), dict):
                        item2id = dm[key]
                        break
            except Exception:
                item2id = None

        # load popular item ids (robust parser)
        raw_lines = _load_high_pop_list(pop_path) if pop_path else []
        pop_items: List[str] = []
        if raw_lines:
            import re
            pat = re.compile(r"Item:\s*([A-Z0-9]+)\s*\(ID:\s*([0-9]+)\)")
            for s in raw_lines:
                s = str(s).strip()
                if not s:
                    continue
                m = pat.search(s)
                if m:
                    asin = m.group(1)
                    pop_items.append(asin)
                else:
                    # tolerate plain ASIN/ID lines
                    tok = s.split()[0]
                    pop_items.append(tok)
        vecs: List[List[float]] = []
        import numpy as np  # local import to avoid global dependency

        def _paths_for(it: str) -> List[str]:
            cand = [os.path.join(feat_dir, f"{it}.npy")]
            if item2id is not None and it in item2id:
                cand.append(os.path.join(feat_dir, f"{item2id[it]}.npy"))
            return cand

        for it in pop_items:
            found = False
            for p in _paths_for(it):
                if os.path.isfile(p):
                    try:
                        v = np.load(p).astype('float32').reshape(-1)
                        vecs.append(v.tolist())
                        found = True
                        break
                    except Exception:
                        continue
            # tolerate misses silently

        if vecs:
            # mean and normalize
            dim = len(vecs[0])
            acc = [0.0] * dim
            for v in vecs:
                if len(v) != dim:
                    continue
                for i, x in enumerate(v):
                    acc[i] += float(x)
            cnt = len(vecs)
            vec = [x / max(1, cnt) for x in acc]
            vec = _l2_normalize(vec)
            _dump_pickle(cache_path, vec)
            return vec
    except Exception:
        # fall back silently below
        pass

    # Fallback: average anchors if provided
    vec: List[float] = []
    if fallback_anchors and len(fallback_anchors) > 0:
        dim = len(fallback_anchors[0])
        acc = [0.0] * dim
        cnt = 0
        for a in fallback_anchors:
            if len(a) == dim:
                for i, v in enumerate(a):
                    acc[i] += float(v)
                cnt += 1
        if cnt > 0:
            vec = [x / cnt for x in acc]

    if not vec:
        # extreme fallback: unit vector of appropriate size (if we can guess)
        if fallback_anchors and len(fallback_anchors) > 0:
            dim = len(fallback_anchors[0])
            vec = [0.0] * dim
            if dim > 0:
                vec[0] = 1.0
        else:
            vec = [1.0]

    vec = _l2_normalize(vec)
    _dump_pickle(cache_path, vec)
    return vec


__all__ = ["build_or_load_pop_center"]
