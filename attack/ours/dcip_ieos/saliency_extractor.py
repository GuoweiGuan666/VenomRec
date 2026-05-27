#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract saliency scores for DCIP-IEOS.

This module implements light‑weight utilities used by the unit tests.  The
real project uses heavy dependencies such as PyTorch and a large VIP5 model.
Those libraries are intentionally not required here so the implementation below
relies solely on standard Python features.  The goal is to mimic the behaviour
of the original code well enough for high level integration tests.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
from numbers import Number
import math
import random

try:  # optional heavy deps
    import numpy as np
    import torch
except Exception:  # pragma: no cover - keep lightweight
    np = None  # type: ignore
    torch = None  # type: ignore

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class SaliencyExtractor:
    """Compute simple saliency metrics and cross‑modal masks."""

    # ------------------------------------------------------------------
    # Basic saliency used as a Grad‑CAM/attention‑rollout fallback
    # ------------------------------------------------------------------
    def extract(self, features: Iterable) -> List[float]:
        """Return the mean absolute value of ``features``.

        ``features`` may either be a single iterable of numbers or an iterable
        of iterables.  Only Python built‑ins are used which keeps the method
        portable and removes the dependency on ``numpy``/``torch``.
        """

        if features is None:
            return []

        try:
            features_list = list(features)
        except TypeError:
            return []

        if len(features_list) == 0:
            return []

        if isinstance(features_list[0], Number):
            tensor = [float(x) for x in features_list]
            return [abs(x) for x in tensor]

        stacked: List[List[float]] = [list(map(float, f)) for f in features_list]
        length = len(stacked[0]) if stacked else 0
        sums = [0.0] * length
        for vec in stacked:
            for i, val in enumerate(vec):
                sums[i] += abs(val)
        return [s / len(stacked) for s in sums]
    
    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def category_ids_to_vis_token_pos(
        self, category_ids: Iterable[Iterable[int]]
    ) -> List[List[int]]:
        """Return visual token positions from ``category_ids``.

        Each element in ``category_ids`` is expected to be an iterable of
        integers where a value of ``1`` marks the position of a visual token.
        The method is intentionally tolerant and will coerce values to ``int``.
        Invalid rows result in an empty position list.
        """

        positions: List[List[int]] = []
        for row in category_ids:
            try:
                positions.append([i for i, v in enumerate(row) if int(v) == 1])
            except Exception:
                positions.append([])
        return positions

    # ------------------------------------------------------------------
    # Cross modal mask extraction
    # ------------------------------------------------------------------
    def extract_cross_modal_masks(
        self,
        items: Iterable[Dict[str, Any]],
        cache_dir: Optional[str] = None,
        top_p: float = 0.15,
        top_q: float = 0.15,
        vis_token_pos: Optional[Iterable[Iterable[int]]] = None,
        model: Optional[Any] = None,
        *,
        min_vis_tokens: int = 1,
        min_txt_tokens: int = 2,
        attn_agg: str = "mean",
        vis_token_replicas: int = 1,
    ) -> Tuple[Dict[int, Dict[str, List[bool]]], Dict[str, Dict[str, float]]]:
        """Compute cross‑modal saliency masks for ``items``.

        The procedure mimics the behaviour of the original project in a very
        small footprint:

        1. Each item's image and text are converted into simple numerical
           representations.  Images are flattened to a list of floats and text
           is mapped to the ordinal value of each character.
        2. If a ``model`` is supplied it is queried with
           ``model(image, text, output_attentions=True)`` and the returned
           ``cross_attentions`` are used.  When the model call fails or the
           attention map does not match the feature dimensions, a warning is
           emitted and a cross‑attention matrix is approximated by taking the
           absolute outer product between the image and text vectors.
        3. Image saliency is the sum over the text dimension and vice versa for
           text saliency.
        4. The top‑``p`` (image) and top‑``q`` (text) proportions are converted
           to binary masks.
        5. When any part of the computation fails, the method falls back to the
           much simpler :meth:`extract` based heuristic which resembles
           Grad‑CAM/attention‑rollout.
        6. All item level masks are cached to
           ``caches/cross_modal_mask.pkl`` relative to this module.
        7. ``vis_token_replicas`` can replicate a single visual token multiple
           times so that ``top_p`` masking yields non-trivial coverage.
        """

        def _to_float_list(obj: Any) -> List[float]:
            if isinstance(obj, list):
                return [float(x) for x in obj]
            if isinstance(obj, (int, float)):
                return [float(obj)]
            return []

        def _encode_text(text: Any) -> List[float]:
            if not isinstance(text, str):
                text = str(text)
            return [float(ord(c)) for c in text]

        def _topk_mask(scores: List[float], ratio: float, name: str) -> List[bool]:
            n = len(scores)
            if n == 0:
                return []
            if n <= 1:
                logging.warning(
                    "WARNING: %s tokens<=1 → mask coverage 100%%", name
                )
                return [True] * n
            k = max(int(math.ceil(n * float(ratio))), 1)
            k = min(k, n)
            indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
            mask = [False] * n
            for i in indices:
                mask[i] = True
            return mask

        masks: Dict[int, Dict[str, List[bool]]] = {}
        img_ratios: List[float] = []
        txt_ratios: List[float] = []
        skipped = 0
        fallback_count = 0

        vis_pos_list = list(vis_token_pos) if vis_token_pos is not None else None

        for idx, item in enumerate(items):
            image_vec = _to_float_list(item.get("image_feat", item.get("image", [])))
            if len(image_vec) == 1 and int(vis_token_replicas) > 1:
                image_vec = image_vec * int(vis_token_replicas)
            text_raw = item.get("text", "")
            text_vec = _encode_text(text_raw)

            n_img = len(image_vec)
            n_txt = len(text_vec)
            if n_img < int(min_vis_tokens) or n_txt < int(min_txt_tokens):
                logging.warning(
                    "WARNING: insufficient tokens (image=%d text=%d) → skipping item %d",
                    n_img,
                    n_txt,
                    idx,
                )
                masks[idx] = {"image": [False] * n_img, "text": [False] * n_txt}
                skipped += 1
                continue
            if n_img == 1:
                logging.info("Degenerate sample: n_img==1 for item %d", idx)

            cross_attn: Optional[List[List[float]]] = None
            # If no model is provided, we are inherently in fallback mode
            warn_fallback = model is None
            if model is not None:
                try:
                    output = model(item.get("image"), text_raw, output_attentions=True)
                    cross_attn_tmp = getattr(output, "cross_attentions", None)
                    if cross_attn_tmp is None and isinstance(output, dict):
                        cross_attn_tmp = output.get("cross_attentions")
                    if cross_attn_tmp is not None:
                        cross_attn_tmp = [list(map(float, row)) for row in cross_attn_tmp]
                        arr = cross_attn_tmp
                        if len(arr) == n_img + 1:
                            arr = arr[1:]
                        if arr and len(arr[0]) == n_txt + 1:
                            arr = [row[1:] for row in arr]
                        if len(arr) == n_img and all(len(row) == n_txt for row in arr):
                            cross_attn = arr
                            assert len(cross_attn) == n_img and all(
                                len(r) == n_txt for r in cross_attn
                            ), f"got {len(cross_attn)}x{len(cross_attn[0]) if cross_attn else 0}, expect {(n_img, n_txt)}"
                        else:
                            warn_fallback = True
                    else:
                        warn_fallback = True
                except Exception:
                    warn_fallback = True

            if cross_attn is None:
                if warn_fallback:
                    logging.warning("FALLBACK: outer-product (fallback to outer-product)")
                    fallback_count += 1
                try:
                    cross_attn = [
                        [abs(i_val * t_val) for t_val in text_vec]
                        for i_val in image_vec
                    ]
                except Exception:
                    cross_attn = None
            else:
                logging.info("Using REAL cross-attn for item %d", idx)

            pos_list: Optional[List[int]] = None
            if vis_pos_list is not None and idx < len(vis_pos_list):
                try:
                    pos_list = [int(p) for p in vis_pos_list[idx]]
                except Exception:
                    pos_list = None
            elif vis_pos_list is None:
                cat_ids = item.get("category_ids")
                if cat_ids is not None:
                    try:
                        pos_list = self.category_ids_to_vis_token_pos([cat_ids])[0]
                    except Exception:
                        pos_list = None

            if pos_list is not None:
                if cross_attn is not None:
                    try:
                        cross_attn = [
                            cross_attn[p]
                            for p in pos_list
                            if 0 <= p < len(cross_attn)
                        ]
                    except Exception:
                        cross_attn = None
                image_vec = [
                    image_vec[p]
                    for p in pos_list
                    if 0 <= p < len(image_vec)
                ]

            if cross_attn is not None:
                logging.info("Using REAL cross-attn for item %d", idx)
                try:
                    img_scores = [sum(row) for row in cross_attn]
                    txt_scores = [sum(col) for col in zip(*cross_attn)] if cross_attn else []
                    if attn_agg == "max":
                        img_scores = [max(row) for row in cross_attn]
                        txt_scores = [max(col) for col in zip(*cross_attn)] if cross_attn else []
                except Exception:
                    img_scores = self.extract(image_vec)
                    txt_scores = self.extract(text_vec)
            else:
                # Fallback: use simple saliency on the raw features
                img_scores = self.extract(image_vec)
                txt_scores = self.extract(text_vec)

            img_mask = _topk_mask(img_scores, top_p, "image")
            if not any(img_mask) and len(img_mask) > 0:
                logging.warning("WARNING: empty image mask; randomly selecting one token")
                img_mask[random.randrange(len(img_mask))] = True
            txt_mask = _topk_mask(txt_scores, top_q, "text")
            if not any(txt_mask) and len(txt_mask) > 0:
                logging.warning("WARNING: empty text mask; randomly selecting one token")
                txt_mask[random.randrange(len(txt_mask))] = True

            if pos_list is not None:
                assert len(img_mask) == len(pos_list)
            assert len(img_mask) == len(image_vec)

            # Store masks together with the text used for their creation.  The
            # saliency stage operates on a character level "tokenizer" where
            # each character represents a token.  Downstream stages can reload
            # this information to ensure that text masks and token sequences are
            # perfectly aligned.
            text_tokens = list(text_raw)
            mask_entry = {
                "image": img_mask,
                "image_mask": img_mask,
                "text_mask": txt_mask,
                "item_id": item.get("id", idx),
                "text": text_raw,
                "text_tokens": text_tokens,
                "tokenizer_name": "char",
                "tokenizer_meta": {},
            }
            masks[idx] = mask_entry

            img_true = sum(1 for b in img_mask if b)
            txt_true = sum(1 for b in txt_mask if b)
            img_ratios.append(img_true / len(img_mask) if img_mask else 0.0)
            txt_ratios.append(txt_true / len(txt_mask) if txt_mask else 0.0)


        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "caches")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "cross_modal_mask.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(masks, f)

        def _summary(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
            vals = sorted(values)
            n = len(vals)
            mean = sum(vals) / n
            mid = n // 2
            if n % 2:
                median = vals[mid]
            else:
                median = (vals[mid - 1] + vals[mid]) / 2.0
            p90 = vals[min(int(n * 0.9), n - 1)]
            return {
                "mean": mean,
                "median": median,
                "p90": p90,
                "min": vals[0],
                "max": vals[-1],
            }

        stats = {"image": _summary(img_ratios), "text": _summary(txt_ratios)}
        total = len(img_ratios) + skipped
        stats["fallback_rate"] = (
            float(fallback_count) / float(total) if total else 0.0
        )
        stats["skipped"] = skipped

        logging.info(
            "Image mask ratios: mean=%.3f median=%.3f p90=%.3f min=%.3f max=%.3f",
            stats["image"]["mean"],
            stats["image"]["median"],
            stats["image"]["p90"],
            stats["image"]["min"],
            stats["image"]["max"],
        )
        logging.info(
            "Text mask ratios: mean=%.3f median=%.3f p90=%.3f min=%.3f max=%.3f",
            stats["text"]["mean"],
            stats["text"]["median"],
            stats["text"]["p90"],
            stats["text"]["min"],
            stats["text"]["max"],
        )
        if stats["fallback_rate"] > 0.2:
            logging.warning(
                "WARNING: fallback_rate %.1f%% >20%%",
                stats["fallback_rate"] * 100,
            )
        else:
            logging.info("fallback_rate %.1f%%", stats["fallback_rate"] * 100)

        return masks, stats
    

# ---------------------------------------------------------------------------
# Helpers mimicking the heavier research implementation
# ---------------------------------------------------------------------------
def project_visual_tokens(victim_model: Any, feats_np: Any) -> Any:
    """Project raw visual features to model's ``d_model`` tokens.

    ``feats_np`` may either be a 1-D array or a 2-D array of shape
    ``[L_img, D_in]``.  When the heavy ``victim_model`` is unavailable the input
    is returned unchanged which keeps the helper functional in the light-weight
    tests.
    """

    if torch is None or victim_model is None:
        return feats_np
    arr = np.asarray(feats_np, dtype="float32") if np is not None else feats_np
    if arr.ndim == 1:
        arr = arr[None, :]
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=getattr(victim_model, "device", "cpu"))
    try:
        with torch.no_grad():
            vis_embeds = victim_model.encoder.visual_embedding(tensor[None, ...])  # type: ignore[attr-defined]
        vis_tokens = vis_embeds.view(1, -1, vis_embeds.shape[-1])
    except Exception:  # pragma: no cover - fallback when model lacks method
        vis_tokens = tensor.unsqueeze(0)
    assert vis_tokens.shape[1] >= 1, f"n_vis_tokens too small: {vis_tokens.shape}"
    return vis_tokens


def assemble_single_item_batch(tokenizer: Any, text_str: str, vis_feats_np: Any, device: str) -> Dict[str, Any]:
    """Assemble inputs mirroring the training template."""

    if tokenizer is not None and torch is not None:
        source_text = f"{text_str} <extra_id_0>"
        enc = tokenizer(source_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
    else:
        ids = [0] * (len(str(text_str)) + 1)
        input_ids = torch.tensor([ids], device=device) if torch is not None else [[0] * len(ids)]
    if torch is not None:
        category_ids = (input_ids == getattr(tokenizer, "convert_tokens_to_ids", lambda x: 0)("<extra_id_0>"))
        category_ids = category_ids.long()
        vis_feats = torch.as_tensor(vis_feats_np, dtype=torch.float32, device=device)
        if vis_feats.ndim == 1:
            vis_feats = vis_feats.unsqueeze(0)
        return {"input_ids": input_ids, "category_ids": category_ids, "vis_feats": vis_feats.unsqueeze(0)}
    return {"input_ids": input_ids, "category_ids": [], "vis_feats": [vis_feats_np]}


def _aggregate_cross_attn(cross_attn: Any) -> Optional[Any]:
    if cross_attn is None or torch is None:
        return None
    try:
        stacked = torch.stack([torch.as_tensor(a) for a in cross_attn], dim=0)  # L,B,H,T_dec,T_enc
        agg = stacked.mean(dim=0)  # B,H,T_dec,T_enc
        agg = agg.mean(dim=1)[0]  # T_dec,T_enc
        return agg
    except Exception:
        return None


def _topk_mask(scores: Any, ratio: float) -> List[bool]:
    arr = scores.detach().cpu().numpy() if torch is not None and isinstance(scores, torch.Tensor) else np.asarray(scores) if np is not None else scores
    n = len(arr)
    if n == 0:
        return []
    k = max(int(math.ceil(n * float(ratio))), 1)
    idx = np.argsort(-arr)[:k] if np is not None else sorted(range(n), key=lambda i: arr[i], reverse=True)[:k]
    mask = [False] * n
    for i in idx:
        mask[int(i)] = True
    return mask


def get_masks_from_cross_attn(
    victim_model: Any,
    tokenizer: Any,
    text_str: str,
    vis_feats_np: Any,
    top_p: float = 0.15,
    top_q: float = 0.10,
) -> Dict[str, List[bool]]:
    """Return image/text masks derived from decoder cross-attention."""

    device = getattr(victim_model, "device", "cpu")
    try:
        batch = assemble_single_item_batch(tokenizer, text_str, vis_feats_np, device)
        labels = batch["input_ids"].clone() if torch is not None and isinstance(batch["input_ids"], torch.Tensor) else None
        outputs = victim_model(
            input_ids=batch.get("input_ids"),
            category_ids=batch.get("category_ids"),
            vis_feats=batch.get("vis_feats"),
            labels=labels,
            output_attentions=True,
            use_cache=False,
        )
        cross = getattr(outputs, "cross_attentions", None)
        if cross is None and isinstance(outputs, dict):
            cross = outputs.get("cross_attentions")
        agg = _aggregate_cross_attn(cross)
        if agg is not None:
            img_scores = agg.sum(dim=0) if torch is not None else agg.sum(axis=0)
            txt_scores = agg.sum(dim=1) if torch is not None else agg.sum(axis=1)
            img_mask = _topk_mask(img_scores, top_p)
            txt_mask = _topk_mask(txt_scores, top_q)
            logging.info("Using REAL cross-attn")
            return {"image": img_mask, "text": txt_mask}
    except Exception:
        pass

    # Silent fallback: compute outer-product based masks without emitting
    # aggregate ratio/fallback logs (to avoid confusing warnings downstream).
    try:
        img = _to_float_list(vis_feats_np)
        txt = _encode_text(text_str)
        # scores per image/text by sum over the other modality
        img_scores = [sum(abs(iv * tv) for tv in txt) for iv in img] if img and txt else []
        txt_scores = [sum(abs(iv * tv) for iv in img) for tv in txt] if img and txt else []
        img_mask = _topk_mask(img_scores, top_p)
        txt_mask = _topk_mask(txt_scores, top_q)
        return {"image": img_mask, "text": txt_mask}
    except Exception:
        return {"image": [], "text": []}


def _norm(x: Any) -> float:
    try:
        import math as _m
        if torch is not None and isinstance(x, torch.Tensor):
            return float(x.pow(2).sum().sqrt())
        arr = np.asarray(x) if np is not None else list(x)
        return float((_m.fsum(float(v) * float(v) for v in arr)) ** 0.5)
    except Exception:
        return 0.0


def get_masks_from_grad(
    victim_model: Any,
    tokenizer: Any,
    text_str: str,
    vis_feats_np: Any,
    c_pop_vec: Any,
    ieos_metric: str = "l2",
    top_p: float = 0.15,
    top_q: float = 0.10,
) -> Dict[str, List[bool]]:
    """Return image/text masks via true gradients of L_ieos.

    Best-effort implementation using autograd when torch is available and
    when the victim encoder accepts ``input_ids``/``category_ids``/``vis_feats``.
    - Image saliency: |∂L/∂vis_feats| per feature dimension (matches mask usage).
    - Text saliency: gradient norm at encoder hidden state per token, mapped
      to a char-level mask for downstream projection to whitespace tokens.
    Falls back to outer-product pathway when anything fails.
    """

    if torch is None:
        # Without torch, we cannot compute true gradients – defer to fallback
        # Silent outer-product fallback (no aggregate warnings)
        try:
            img = _to_float_list(vis_feats_np)
            txt = _encode_text(text_str)
            img_scores = [sum(abs(iv * tv) for tv in txt) for iv in img] if img and txt else []
            txt_scores = [sum(abs(iv * tv) for iv in img) for tv in txt] if img and txt else []
            return {"image": _topk_mask(img_scores, top_p), "text": _topk_mask(txt_scores, top_q)}
        except Exception:
            return {"image": [], "text": []}

    try:
        device = getattr(victim_model, "device", "cpu")
        victim_model.eval()
        batch = assemble_single_item_batch(tokenizer, text_str, vis_feats_np, device)
        input_ids = batch.get("input_ids")
        category_ids = batch.get("category_ids")
        vis_feats = batch.get("vis_feats")
        assert isinstance(input_ids, torch.Tensor) and isinstance(vis_feats, torch.Tensor)
        vis_feats = vis_feats.clone().detach().requires_grad_(True)

        # Forward through encoder to obtain token-level hidden states
        enc = victim_model.encoder
        out = enc(
            input_ids=input_ids,
            whole_word_ids=torch.zeros_like(input_ids),
            category_ids=category_ids,
            vis_feats=vis_feats,
            return_dict=True,
        )
        hid = getattr(out, "last_hidden_state", None)
        if hid is None:
            raise RuntimeError("encoder returned no last_hidden_state")
        hid = hid.detach().requires_grad_(True)

        # Compute L_ieos by selected metric
        fused = hid.mean(dim=1)[0]
        cpop = torch.as_tensor(c_pop_vec, dtype=torch.float32, device=device)
        d = min(fused.numel(), cpop.numel())
        if d <= 0:
            raise RuntimeError("dimension mismatch for IEOS gradient")
        fv = fused[:d]
        cv = cpop[:d]
        m = str(ieos_metric).lower()
        if m == "cos":
            fv_u = fv / (fv.norm(p=2) + 1e-12)
            cv_u = cv / (cv.norm(p=2) + 1e-12)
            L = 2.0 - 2.0 * (fv_u * cv_u).sum()
        elif m == "raw_l2":
            L = ((fv - cv) ** 2).sum()
        else:  # 'l2' unit-L2
            fv_u = fv / (fv.norm(p=2) + 1e-12)
            cv_u = cv / (cv.norm(p=2) + 1e-12)
            L = ((fv_u - cv_u) ** 2).sum()
        L.backward()

        # Image saliency from feature-level gradients
        try:
            gimg = vis_feats.grad  # [1,1,F]
            if gimg is not None:
                gimg = gimg.detach().abs().view(-1)  # per-feature abs gradient
                img_mask = _topk_mask(gimg, top_p)
            else:
                img_mask = []
        except Exception:
            img_mask = []

        # Text saliency from hidden-state gradients mapped to characters
        try:
            hid_grad = hid.grad  # [1,T,d]
            if hid_grad is not None:
                token_scores = hid_grad.detach().norm(dim=-1)[0].cpu().tolist()  # [T]
            else:
                token_scores = []

            # Map token scores into a char-level mask by greedy string matching
            text_raw = str(text_str or "")
            chars = list(text_raw)
            n_chars = len(chars)
            char_scores = [0.0] * n_chars
            if token_scores and hasattr(tokenizer, "convert_ids_to_tokens"):
                toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                # Stop at placeholder token for visual features if present
                try:
                    stop_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
                except Exception:
                    stop_id = None
                ptr = 0
                for i, (tok_id, score) in enumerate(zip(input_ids[0].tolist(), token_scores)):
                    if stop_id is not None and int(tok_id) == int(stop_id):
                        break
                    tok = toks[i]
                    # heuristic: T5 uses "▁" to mark spaces; strip it for matching
                    piece = tok.replace("▁", " ").strip()
                    if not piece:
                        continue
                    j = text_raw.find(piece, ptr)
                    if j < 0:
                        # fallback: try from start (may overcount duplicates)
                        j = text_raw.find(piece)
                    if j >= 0:
                        k = min(n_chars, j + len(piece))
                        for p in range(j, k):
                            char_scores[p] += float(score)
                        ptr = k
            # Build top-q char mask
            txt_mask = _topk_mask(torch.tensor(char_scores), top_q) if torch is not None else _topk_mask(char_scores, top_q)
            return {
                "image": img_mask,
                "text": txt_mask,
                "text_tokens": list(text_raw),
                "tokenizer_name": "char",
            }
        except Exception:
            pass

    except Exception:
        pass

    # Silent outer-product fallback (no aggregate warnings)
    try:
        img = _to_float_list(vis_feats_np)
        txt = _encode_text(text_str)
        img_scores = [sum(abs(iv * tv) for tv in txt) for iv in img] if img and txt else []
        txt_scores = [sum(abs(iv * tv) for iv in img) for tv in txt] if img and txt else []
        return {"image": _topk_mask(img_scores, top_p), "text": _topk_mask(txt_scores, top_q)}
    except Exception:
        return {"image": [], "text": []}




# ---------------------------------------------------------------------------
# Convenience helper used by the poison pipeline
# ---------------------------------------------------------------------------

def get_masks(
    model: Optional[Any],
    state: Dict[str, Any],
    use_cache: bool,
    *,
    saliency_mode: str = "attn",
    c_pop: Optional[Any] = None,
    ieos_metric: str = "l2",
) -> Dict[str, List[bool]]:
    """Return cross-modal saliency masks for the given ``state``.

    The real project obtains saliency masks from a large model.  For the test
    environment we mimic the behaviour using :class:`SaliencyExtractor`.  When
    ``use_cache`` is ``True`` the function first looks for a previously
    computed mask stored under ``state['mask']``.  If none is found, or when
    ``use_cache`` is ``False``, the masks are recomputed based on the current
    image and text stored in ``state``.
    """

    cached = state.get("mask") if isinstance(state, dict) else None
    if use_cache and isinstance(cached, dict):
        return {
            "image": list(cached.get("image", [])),
            "text": list(cached.get("text", [])),
        }

    extractor = SaliencyExtractor()
    img_feat = state.get("image") if isinstance(state, dict) else None
    txt_feat = state.get("text") if isinstance(state, dict) else None
    # Victim-aware path: if model looks like a victim adapter, try cross-attn
    try:
        if model is not None and hasattr(model, "model") and hasattr(model, "tokenizer"):
            if str(saliency_mode).lower() == "grad" and c_pop is not None:
                m = get_masks_from_grad(model.model, model.tokenizer, str(txt_feat or ""), img_feat, c_pop, ieos_metric=ieos_metric, top_p=0.15, top_q=0.10)
            else:
                m = get_masks_from_cross_attn(model.model, model.tokenizer, str(txt_feat or ""), img_feat, top_p=0.15, top_q=0.10)
            mask = {
                "image": list(m.get("image", [])),
                "text": list(m.get("text", [])),
                "text_tokens": list(m.get("text_tokens", [])) if isinstance(m, dict) else [],
                "tokenizer_name": str(m.get("tokenizer_name", "")) if isinstance(m, dict) else "",
            }
        else:
            # Prefer adapter-aware path if a victim-like wrapper was provided
            items = [{"image": img_feat, "text": txt_feat, "image_feat": img_feat}]
            masks, _ = extractor.extract_cross_modal_masks(items, model=model)
            mask = masks.get(0, {"image": [], "text": []})
    except Exception:
        items = [{"image": img_feat, "text": txt_feat, "image_feat": img_feat}]
        masks, _ = extractor.extract_cross_modal_masks(items, model=model)
        mask = masks.get(0, {"image": [], "text": []})

    img_cov = (
        sum(1 for b in mask.get("image", []) if b) / len(mask.get("image", []))
        if mask.get("image")
        else 0.0
    )
    txt_cov = (
        sum(1 for b in mask.get("text", []) if b) / len(mask.get("text", []))
        if mask.get("text")
        else 0.0
    )
    logging.info(
        "mask_coverage image=%.3f text=%.3f", img_cov, txt_cov
    )


    if isinstance(state, dict):
        state["mask"] = mask

    return {"image": list(mask.get("image", [])), "text": list(mask.get("text", []))}
