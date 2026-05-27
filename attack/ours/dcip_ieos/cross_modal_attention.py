"""Utilities to extract and cache cross-modal attention maps.

These helpers provide a thin layer around ``VictimAdapter`` that keeps
cross-attention matrices on disk, making it cheap to inspect or visualise the
regions that drive the interactive poisoning pipeline.  Attention arrays are
stored as compressed ``.npz`` files containing:

* ``matrix``: full V×T cross-attention matrix.
* ``visual_scores``: per-visual token importance (mean pool over text tokens).
* ``text_scores``: per-text token importance (mean pool over visual tokens).
* ``tokens``: token strings aligned with ``text_scores``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Iterable, List, Tuple

import numpy as np


@dataclass
class CrossModalAttention:
    """Container returned by :func:`compute_cross_modal_attention`."""

    matrix: np.ndarray
    visual_scores: np.ndarray
    text_scores: np.ndarray
    tokens: List[str]

    def top_visual(self, k: int = 5) -> List[Tuple[int, float]]:
        idx = np.argsort(self.visual_scores)[::-1][: max(1, k)]
        return [(int(i), float(self.visual_scores[i])) for i in idx]

    def top_text(self, k: int = 5) -> List[Tuple[str, float]]:
        idx = np.argsort(self.text_scores)[::-1][: max(1, k)]
        return [(self.tokens[i] if i < len(self.tokens) else str(i), float(self.text_scores[i])) for i in idx]


def _ensure_2d(attn: np.ndarray) -> np.ndarray:
    if attn.ndim == 2:
        return attn
    if attn.ndim == 3:
        return attn.reshape(attn.shape[0], attn.shape[2])
    if attn.ndim == 4:
        return attn.reshape(attn.shape[1], attn.shape[2])
    raise ValueError(f"Unsupported attention shape: {attn.shape}")


def _tokenize(adapter: Any, text: str) -> List[str]:
    tok = getattr(adapter, "tokenizer", None)
    if tok is None:
        return text.split()
    try:
        if hasattr(tok, "tokenize") and callable(tok.tokenize):
            tokens = tok.tokenize(text)
            if isinstance(tokens, list) and tokens:
                return [str(t) for t in tokens]
        if hasattr(tok, "encode") and hasattr(tok, "convert_ids_to_tokens"):
            encoded = tok.encode(text, add_special_tokens=True)
            tokens = tok.convert_ids_to_tokens(encoded)
            return [str(t) for t in tokens]
        batch = tok(text, return_tensors="pt") if callable(tok) else None
        input_ids = None
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
        if input_ids is not None and hasattr(tok, "convert_ids_to_tokens"):
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if isinstance(input_ids, list) and input_ids:
                flat = input_ids[0] if isinstance(input_ids[0], list) else input_ids
                return [str(t) for t in tok.convert_ids_to_tokens(flat)]
    except Exception:  # pragma: no cover - best effort for diverse tokenizers
        pass
    tokens = text.split()
    return tokens if tokens else [text]


def compute_cross_modal_attention(
    adapter: Any,
    image_feat: Iterable[float],
    text: str,
) -> CrossModalAttention:
    """Query the adapter and return a ``CrossModalAttention`` structure."""

    attn_dict = adapter(image=image_feat, text=text, output_attentions=True)
    cross = np.asarray(attn_dict.get("cross_attentions", []), dtype=float)
    if cross.size == 0:
        raise RuntimeError("VictimAdapter did not return cross-attention values")

    cross = _ensure_2d(cross)
    visual_scores = cross.mean(axis=1)
    text_scores = cross.mean(axis=0)
    tokens = _tokenize(adapter, text)

    # Align token scores with available tokens (truncate or pad zeros)
    if text_scores.size != len(tokens):
        if text_scores.size < len(tokens):
            tokens = tokens[: text_scores.size]
        else:
            extra = [f"tok_{i}" for i in range(text_scores.size - len(tokens))]
            tokens = tokens + extra

    return CrossModalAttention(
        matrix=cross,
        visual_scores=visual_scores,
        text_scores=text_scores,
        tokens=tokens,
    )


def _cache_path(cache_dir: str, dataset: str, key: str) -> str:
    safe_dataset = dataset.replace(os.sep, "_")
    safe_key = key.replace(os.sep, "_")
    return os.path.join(cache_dir, f"cross_attn_{safe_dataset}_{safe_key}.npz")


def load_cached_attention(cache_dir: str, dataset: str, key: str) -> CrossModalAttention | None:
    path = _cache_path(cache_dir, dataset, key)
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)
    matrix = np.asarray(data["matrix"], dtype=float)
    visual_scores = np.asarray(data["visual_scores"], dtype=float)
    text_scores = np.asarray(data["text_scores"], dtype=float)
    tokens = list(data["tokens"].tolist())
    return CrossModalAttention(matrix=matrix, visual_scores=visual_scores, text_scores=text_scores, tokens=tokens)


def save_attention(cache_dir: str, dataset: str, key: str, attn: CrossModalAttention) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, dataset, key)
    np.savez_compressed(
        path,
        matrix=attn.matrix,
        visual_scores=attn.visual_scores,
        text_scores=attn.text_scores,
        tokens=np.asarray(attn.tokens, dtype=object),
    )
    return path


def get_or_compute_attention(
    adapter: Any,
    image_feat: Iterable[float],
    text: str,
    *,
    cache_dir: str,
    dataset: str,
    cache_key: str,
    force: bool = False,
) -> Tuple[CrossModalAttention, str | None]:
    """Load attention from cache or compute and persist it."""

    cached = None if force else load_cached_attention(cache_dir, dataset, cache_key)
    if cached is not None:
        return cached, _cache_path(cache_dir, dataset, cache_key)

    try:
        attn = compute_cross_modal_attention(adapter, image_feat, text)
    except Exception as exc:
        logging.warning("[cross-attn] compute failed for key=%s: %s", cache_key, str(exc))
        raise
    path = save_attention(cache_dir, dataset, cache_key, attn)
    return attn, path


__all__ = [
    "CrossModalAttention",
    "compute_cross_modal_attention",
    "get_or_compute_attention",
    "load_cached_attention",
    "save_attention",
]
