"""Interactive cross-modal perturbation utilities (refactor in progress).

This module implements a simplified push-to-pop pipeline built around the
following stages:

1. Load anchor / popular centre representations.
2. Extract cross-modal masks from the victim adapter (with deterministic
   fallbacks when unavailable).
3. Perform a small number of alternating image / text perturbation rounds while
   monitoring the cosine similarity between the current target embedding and
   the popular centre.
4. Return a compact record describing the perturbation trajectory.

The goal is to replace the legacy DCIP-IEOS logic with a minimal yet effective
implementation that better matches the “interactive push-to-pop” description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import logging
import math

import numpy as np

from .dynamic_perturbation_optimizer import AdaptiveBudgetController
from .image_perturbation import ImagePerturber
from .losses import info_nce_loss
from .text_perturbation import TextPerturber, normalize_token


@dataclass
class PerturbationRound:
    """Record collected after each interaction round."""

    round_idx: int
    sim_after_image: float
    sim_target: float
    dist_target: float
    img_eps_used: float
    txt_ratio: float
    img_budget: float
    txt_budget: float
    info_nce: float
    psnr_img: float
    bce_loss: float
    hit_prob: float
    txt_embed_delta_norm: float | None = None
    stop_reason: str | None = None


@dataclass
class PerturbationResult:
    """Summary returned for every poisoned target."""

    target_id: str
    rounds: List[PerturbationRound]
    final_image: List[float]
    final_text: str


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _topk_mask(values: np.ndarray, ratio: float) -> np.ndarray:
    """Return boolean mask selecting the top ratio of entries."""

    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    n = max(1, int(math.ceil(values.size * max(0.0, min(1.0, ratio)))))
    idx = np.argpartition(values.reshape(-1), -n)[-n:]
    mask = np.zeros_like(values, dtype=bool)
    mask.reshape(-1)[idx] = True
    return mask


def get_visual_text_masks(
    adapter: Any,
    image_feat: np.ndarray,
    text: str,
    *,
    vis_ratio: float = 0.15,
    txt_ratio: float = 0.15,
    token_count: int | None = None,
    pop_vec: Iterable[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract cross-modal masks with deterministic fallbacks.

    Parameters
    ----------
    adapter:
        VictimAdapter exposing ``__call__`` returning a cross-attention map.
    image_feat:
        Image feature vector (flattened patch tokens).
    text:
        Current text string.
    vis_ratio / txt_ratio:
        Proportion of visual/text tokens to keep in the mask.
    """

    image_arr = np.asarray(image_feat, dtype=float).reshape(-1)
    vis_tokens = int(image_arr.size)
    txt_tokens = token_count if token_count and token_count > 0 else max(len(text.split()), 1)

    try:
        saliency = adapter.compute_cross_modal_saliency(image_feat, text)
        vis_scores = np.asarray(saliency.get("visual_scores", []), dtype=float)
        txt_scores = np.asarray(saliency.get("text_scores", []), dtype=float)
        if vis_scores.size == 0:
            raise RuntimeError("empty visual saliency")
        if vis_scores.size != vis_tokens:
            repeats = int(math.ceil(vis_tokens / max(vis_scores.size, 1)))
            vis_scores = np.repeat(vis_scores, repeats)[:vis_tokens]
        if txt_scores.size == 0:
            txt_scores = np.ones(txt_tokens, dtype=float)
        txt_tokens = txt_scores.size
        vis_mask = _topk_mask(vis_scores, vis_ratio)
        txt_mask = _topk_mask(txt_scores, txt_ratio)
        return vis_mask, txt_mask
    except Exception as exc:
        logging.debug("[mask] saliency extraction failed (%s); using fallback", str(exc))

    vis_mask = np.zeros((vis_tokens,), dtype=bool)
    txt_mask = np.zeros((txt_tokens,), dtype=bool)
    if vis_tokens:
        vis_mask[:max(1, int(math.ceil(vis_tokens * vis_ratio)))] = True
    if txt_tokens:
        txt_mask[:max(1, int(math.ceil(txt_tokens * txt_ratio)))] = True
    return vis_mask, txt_mask


# ---------------------------------------------------------------------------
# Token helpers shared across perturbation modes
# ---------------------------------------------------------------------------

def _count_tokens(adapter: Any, text: str) -> int:
    tok = getattr(adapter, "tokenizer", None)
    if tok is None:
        return max(len(text.split()), 1)
    try:
        if hasattr(tok, "encode") and callable(tok.encode):
            encoded = tok.encode(text)
            if isinstance(encoded, list):
                return len(encoded) or 1
        batch = tok(text, return_tensors=None)
        ids = batch.get("input_ids") if isinstance(batch, dict) else batch
        if isinstance(ids, list):
            if ids and isinstance(ids[0], list):
                return len(ids[0]) or 1
            return len(ids) or 1
    except Exception:
        pass
    return max(len(text.split()), 1)


# ---------------------------------------------------------------------------
# Interactive perturbation
# ---------------------------------------------------------------------------

def compute_cosine(a: Iterable[float], b: Iterable[float]) -> float:
    vec_a = np.asarray(list(a), dtype=float)
    vec_b = np.asarray(list(b), dtype=float)
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def interactive_perturb_target(
    adapter: Any,
    anchor_vec: Iterable[float],
    base_text: str,
    c_pop: Iterable[float],
    *,
    cross_mode: str = "interactive",
    keywords: Iterable[str] | None = None,
    rounds: int = 4,
    vis_ratio: float = 0.15,
    txt_ratio: float = 0.15,
    img_eps: float = 0.05,
    img_iters: int | None = None,
    psnr_min: float | None = None,
    img_strategy: str | None = None,
    text_replace_ratio: float = 0.20,
    min_txt_replacements: int = 2,
    sim_threshold: float = 0.92,
    text_embed_eps: float = 0.0,
    lock_vis_mask: bool = False,
    vis_decay: float = 1.0,
    stop_on_plateau: bool = True,
) -> PerturbationResult:
    """Run the alternating perturbation loop for a single target."""

    anchor_arr = np.asarray(list(anchor_vec), dtype=float)
    image_feat = anchor_arr.copy()
    anchor_arr = image_feat.copy()
    current_text = base_text
    pop_vec = np.asarray(list(c_pop), dtype=float)

    rounds_log: List[PerturbationRound] = []
    # Configure image perturber (default strategy is cosine single-step)
    _iters = 3 if img_iters is None else int(img_iters)
    _psnr = 30.0 if psnr_min is None else float(psnr_min)
    _strategy = (img_strategy or "cosine").strip().lower()
    image_perturber = ImagePerturber(eps=img_eps, iters=_iters, psnr_min=_psnr, strategy=_strategy)
    text_perturber = TextPerturber(ratio=text_replace_ratio)
    controller = AdaptiveBudgetController(
        img_eps=img_eps,
        txt_ratio=text_replace_ratio,
        embed_eps=text_embed_eps,
    )

    mode = (cross_mode or "interactive").strip().lower()
    if mode not in {"interactive", "decoupled"}:
        raise ValueError(f"Unsupported cross_mode '{cross_mode}'. Expected 'interactive' or 'decoupled'.")

    best_sim = -1.0
    best_state = (image_feat.copy(), current_text)

    def _run_single_step(image_feat_arr: np.ndarray, text: str) -> PerturbationResult:
        rounds_log: List[PerturbationRound] = []
        img_budget, txt_budget, embed_budget = controller.current()
        image_perturber.eps = img_budget
        text_perturber.ratio = txt_budget

        token_count = _count_tokens(adapter, text)
        vis_mask, txt_mask = get_visual_text_masks(
            adapter,
            image_feat_arr,
            text,
            vis_ratio=vis_ratio,
            txt_ratio=txt_ratio,
            token_count=token_count,
            pop_vec=pop_vec,
        )

        psnr_img = float("inf")
        eps_used = 0.0
        if img_budget > 0:
            new_img, psnr_img, eps_used, _img_meta = image_perturber.perturb(
                image_feat_arr,
                mask=vis_mask,
                target_feat=pop_vec,
            )
            image_feat_arr = np.asarray(new_img, dtype=float)

        sim_after_image = compute_cosine(image_feat_arr, pop_vec)

        text_active = (
            txt_budget > 0
            or min_txt_replacements > 0
            or text_embed_eps > 0
        )
        txt_mask_after = txt_mask
        if text_active:
            token_count = _count_tokens(adapter, text)
            _, txt_mask_after = get_visual_text_masks(
                adapter,
                image_feat_arr,
                text,
                vis_ratio=vis_ratio,
                txt_ratio=txt_ratio,
                token_count=token_count,
                pop_vec=pop_vec,
            )

        txt_mask_arr = np.asarray(txt_mask_after).reshape(-1)
        txt_mask_bool = [bool(m) for m in txt_mask_arr]
        token_count = len(txt_mask_bool)
        token_embeddings = None
        if text_embed_eps > 0.0 and hasattr(adapter, "encode_text_tokens"):
            try:
                enc = adapter.encode_text_tokens(text)
                if isinstance(enc, dict):
                    token_embeddings = enc.get("tokens")
            except Exception:
                token_embeddings = None
        if token_embeddings is not None:
            emb_arr = np.asarray(token_embeddings)
            if emb_arr.ndim == 2 and emb_arr.shape[0] == token_count:
                token_embeddings = emb_arr.tolist()
            else:
                token_embeddings = None

        replace_ratio = 0.0
        text_meta = None
        if text_active:
            new_text, replace_ratio, text_meta = text_perturber.perturb(
                tokens=None,
                text=text,
                mask=txt_mask_bool,
                tokenizer=getattr(adapter, "tokenizer", None),
                txt_ratio_max=txt_budget,
                min_txt_replacements=min_txt_replacements,
                keywords=list(keywords or []),
                text_topk=6,
                embed_eps=embed_budget,
                token_embeddings=token_embeddings,
            )
            if new_text:
                text = new_text

        sim = compute_cosine(image_feat_arr, pop_vec)
        dist = float(np.linalg.norm(image_feat_arr - pop_vec))
        norm_prod = np.linalg.norm(image_feat_arr) * np.linalg.norm(pop_vec) + 1e-12
        dot_val = float(np.dot(image_feat_arr, pop_vec) / norm_prod)
        hit_prob = 1.0 / (1.0 + math.exp(-dot_val))
        bce_loss = -math.log(hit_prob + 1e-12)
        info_nce_val = info_nce_loss(image_feat_arr, pop_vec, negatives=[anchor_arr])

        txt_embed_norm = None
        if isinstance(text_meta, dict):
            delta = text_meta.get("embedding_delta")
            if delta is not None:
                arr = np.asarray(delta, dtype=float)
                if arr.size > 0:
                    txt_embed_norm = float(np.linalg.norm(arr))

        rounds_log.append(
            PerturbationRound(
                round_idx=0,
                sim_after_image=sim_after_image,
                sim_target=sim,
                dist_target=dist,
                img_eps_used=float(eps_used),
                txt_ratio=float(replace_ratio),
                img_budget=float(img_budget),
                txt_budget=float(txt_budget),
                psnr_img=float(psnr_img) if not np.isnan(psnr_img) else float("inf"),
                info_nce=float(info_nce_val),
                bce_loss=float(bce_loss),
                hit_prob=float(hit_prob),
                txt_embed_delta_norm=txt_embed_norm,
                stop_reason="single_step",
            )
        )

        return PerturbationResult(
            target_id="",
            rounds=rounds_log,
            final_image=list(image_feat_arr),
            final_text=text,
        )

    if rounds <= 0:
        text_enabled = (
            txt_ratio > 0
            or text_replace_ratio > 0
            or text_embed_eps > 0
            or min_txt_replacements > 0
        )
        if img_eps > 0 or text_enabled:
            return _run_single_step(image_feat, current_text)

    locked_vis_mask = None
    for r in range(rounds):
        img_budget, txt_budget, embed_budget = controller.current()
        image_perturber.eps = img_budget
        text_perturber.ratio = txt_budget

        token_count = _count_tokens(adapter, current_text)
        curr_vis_ratio = max(1e-6, min(1.0, vis_ratio * (vis_decay ** r)))
        computed_vis_mask, txt_mask = get_visual_text_masks(
            adapter,
            image_feat,
            current_text,
            vis_ratio=curr_vis_ratio,
            txt_ratio=txt_ratio,
            token_count=token_count,
            pop_vec=pop_vec,
        )
        if locked_vis_mask is None or not lock_vis_mask:
            vis_mask = computed_vis_mask
            if lock_vis_mask and locked_vis_mask is None:
                locked_vis_mask = np.asarray(vis_mask).astype(bool)
        else:
            vis_mask = locked_vis_mask
        initial_txt_mask = txt_mask
        token_count = len(txt_mask)

        # Step 1: image perturbation
        new_img, psnr_img, eps_used, _img_meta = image_perturber.perturb(
            image_feat,
            mask=vis_mask,
            target_feat=pop_vec,
        )
        image_feat = np.asarray(new_img, dtype=float)
        sim_after_image = compute_cosine(image_feat, pop_vec)

        if mode == "decoupled":
            txt_mask = initial_txt_mask
        else:
            # Step 2: recompute text mask with updated image
            token_count = _count_tokens(adapter, current_text)
            _, txt_mask = get_visual_text_masks(
                adapter,
                image_feat,
                current_text,
                vis_ratio=vis_ratio,
                txt_ratio=txt_ratio,
                token_count=token_count,
                pop_vec=pop_vec,
            )
        token_count = len(txt_mask)

        # Ensure boolean list for TextPerturber
        txt_mask_arr = np.asarray(txt_mask).reshape(-1)
        txt_mask_bool = [bool(m) for m in txt_mask_arr]
        if len(txt_mask_bool) != token_count:
            if len(txt_mask_bool) > token_count:
                txt_mask_bool = txt_mask_bool[:token_count]
            else:
                txt_mask_bool.extend([False] * (token_count - len(txt_mask_bool)))
        token_embeddings = None
        if text_embed_eps > 0.0 and hasattr(adapter, "encode_text_tokens"):
            try:
                enc = adapter.encode_text_tokens(current_text)
                if isinstance(enc, dict):
                    token_embeddings = enc.get("tokens")
            except Exception:
                token_embeddings = None
        if token_embeddings is not None:
            emb_arr = np.asarray(token_embeddings)
            if emb_arr.ndim == 2 and emb_arr.shape[0] == token_count:
                token_embeddings = emb_arr.tolist()
            else:
                token_embeddings = None

        new_text, replace_ratio, text_meta = text_perturber.perturb(
            tokens=None,
            text=current_text,
            mask=txt_mask_bool,
            tokenizer=getattr(adapter, "tokenizer", None),
            txt_ratio_max=txt_budget,
            min_txt_replacements=min_txt_replacements,
            keywords=list(keywords or []),
            text_topk=6,
            embed_eps=embed_budget,
            token_embeddings=token_embeddings,
        )
        if new_text:
            current_text = new_text

        sim = compute_cosine(image_feat, pop_vec)
        dist = float(np.linalg.norm(image_feat - pop_vec))
        norm_prod = np.linalg.norm(image_feat) * np.linalg.norm(pop_vec) + 1e-12
        dot_val = float(np.dot(image_feat, pop_vec) / norm_prod)
        hit_prob = 1.0 / (1.0 + math.exp(-dot_val))
        bce_loss = -math.log(hit_prob + 1e-12)

        info_nce_val = info_nce_loss(image_feat, pop_vec, negatives=[anchor_arr])

        txt_embed_norm = None
        if isinstance(text_meta, dict):
            delta = text_meta.get("embedding_delta")
            if delta is not None:
                arr = np.asarray(delta, dtype=float)
                if arr.size > 0:
                    txt_embed_norm = float(np.linalg.norm(arr))

        improvement = sim - best_sim
        if improvement < -controller.sim_tolerance:
            image_feat = best_state[0].copy()
            current_text = best_state[1]
            reason = "no_improvement"
            rounds_log.append(
                PerturbationRound(
                    round_idx=r,
                    sim_after_image=best_sim,
                    sim_target=best_sim,
                    dist_target=float(np.linalg.norm(image_feat - pop_vec)),
                    img_eps_used=float(eps_used),
                    txt_ratio=float(replace_ratio),
                    img_budget=float(img_budget),
                    txt_budget=float(txt_budget),
                    psnr_img=float(psnr_img) if not np.isnan(psnr_img) else float("inf"),
                    info_nce=float(info_nce_val),
                    bce_loss=float(bce_loss),
                    hit_prob=float(hit_prob),
                    txt_embed_delta_norm=txt_embed_norm,
                    stop_reason=reason,
                )
            )
            if stop_on_plateau:
                break

        rounds_log.append(
            PerturbationRound(
                round_idx=r,
                sim_after_image=sim_after_image,
                sim_target=sim,
                dist_target=dist,
                img_eps_used=float(eps_used),
                txt_ratio=float(replace_ratio),
                img_budget=float(img_budget),
                txt_budget=float(txt_budget),
                psnr_img=float(psnr_img) if not np.isnan(psnr_img) else float("inf"),
                info_nce=float(info_nce_val),
                bce_loss=float(bce_loss),
                hit_prob=float(hit_prob),
                txt_embed_delta_norm=txt_embed_norm,
            )
        )

        controller.update(
            sim=sim,
            info_nce=info_nce_val,
            alignment=dist,
            bce=bce_loss,
            hit_prob=hit_prob,
        )

        if sim > best_sim:
            best_sim = sim
            best_state = (image_feat.copy(), current_text)

        if sim >= sim_threshold:
            rounds_log[-1].stop_reason = "sim_threshold"
            break

    final_img, final_text = best_state
    return PerturbationResult(
        target_id="",
        rounds=rounds_log,
        final_image=list(final_img),
        final_text=final_text,
    )


def single_step_perturb(
    adapter: Any,
    anchor_vec: Iterable[float],
    base_text: str,
    c_pop: Iterable[float],
    *,
    keywords: Iterable[str] | None = None,
    vis_ratio: float = 0.15,
    txt_ratio: float = 0.15,
    img_eps: float = 0.05,
    text_replace_ratio: float = 0.2,
    min_txt_replacements: int = 2,
    text_embed_eps: float = 0.0,
    independent_text: bool = False,
) -> PerturbationResult:
    """Apply a single mixed image/text step without adaptive interaction."""

    anchor_arr = np.asarray(list(anchor_vec), dtype=float)
    image_feat = anchor_arr.copy()
    pop_vec = np.asarray(list(c_pop), dtype=float)
    current_text = base_text

    image_perturber = ImagePerturber(eps=img_eps)
    text_perturber = TextPerturber(ratio=text_replace_ratio)

    token_count = _count_tokens(adapter, current_text)
    vis_mask, txt_mask = get_visual_text_masks(
        adapter=adapter,
        image_feat=image_feat,
        text=current_text,
        vis_ratio=vis_ratio,
        txt_ratio=txt_ratio,
        token_count=token_count,
        pop_vec=pop_vec,
    )

    psnr_img = float("inf")
    eps_used = 0.0
    if img_eps > 0:
        new_img, psnr_img, eps_used, _ = image_perturber.perturb(
            image_feat,
            mask=vis_mask,
            target_feat=pop_vec,
        )
        image_feat = np.asarray(new_img, dtype=float)

    txt_mask_initial = np.asarray(txt_mask).reshape(-1)
    txt_mask_bool = [bool(m) for m in txt_mask_initial]

    if not independent_text and (text_replace_ratio > 0 or min_txt_replacements > 0 or text_embed_eps > 0):
        token_count = _count_tokens(adapter, current_text)
        _, new_txt_mask = get_visual_text_masks(
            adapter,
            image_feat,
            current_text,
            vis_ratio=vis_ratio,
            txt_ratio=txt_ratio,
            token_count=token_count,
            pop_vec=pop_vec,
        )
        txt_mask_bool = [bool(m) for m in np.asarray(new_txt_mask).reshape(-1)]

    text_active = (
        text_replace_ratio > 0
        or min_txt_replacements > 0
        or text_embed_eps > 0
    )
    replace_ratio = 0.0
    txt_embed_norm = None
    if text_active and txt_mask_bool:
        token_embeddings = None
        if text_embed_eps > 0.0 and hasattr(adapter, "encode_text_tokens"):
            try:
                enc = adapter.encode_text_tokens(current_text)
                if isinstance(enc, dict):
                    token_embeddings = enc.get("tokens")
            except Exception:
                token_embeddings = None
        if token_embeddings is not None:
            emb_arr = np.asarray(token_embeddings)
            if emb_arr.ndim == 2 and emb_arr.shape[0] == len(txt_mask_bool):
                token_embeddings = emb_arr.tolist()
            else:
                token_embeddings = None

        new_text, replace_ratio, text_meta = text_perturber.perturb(
            tokens=None,
            text=current_text,
            mask=txt_mask_bool,
            tokenizer=getattr(adapter, "tokenizer", None),
            txt_ratio_max=text_replace_ratio,
            min_txt_replacements=min_txt_replacements,
            keywords=list(keywords or []),
            text_topk=6,
            embed_eps=text_embed_eps,
            token_embeddings=token_embeddings,
        )
        if new_text:
            current_text = new_text
        if isinstance(text_meta, dict):
            delta = text_meta.get("embedding_delta")
            if delta is not None:
                arr = np.asarray(delta, dtype=float)
                if arr.size > 0:
                    txt_embed_norm = float(np.linalg.norm(arr))

    sim = compute_cosine(image_feat, pop_vec)
    dist = float(np.linalg.norm(image_feat - pop_vec))
    norm_prod = np.linalg.norm(image_feat) * np.linalg.norm(pop_vec) + 1e-12
    dot_val = float(np.dot(image_feat, pop_vec) / norm_prod)
    hit_prob = 1.0 / (1.0 + math.exp(-dot_val))
    bce_loss = -math.log(hit_prob + 1e-12)
    info_nce_val = info_nce_loss(image_feat, pop_vec, negatives=[anchor_arr])

    rounds_log = [
        PerturbationRound(
            round_idx=0,
            sim_after_image=sim,
            sim_target=sim,
            dist_target=dist,
            img_eps_used=float(eps_used),
            txt_ratio=float(replace_ratio),
            img_budget=float(img_eps),
            txt_budget=float(text_replace_ratio),
            psnr_img=float(psnr_img) if not np.isnan(psnr_img) else float("inf"),
            info_nce=float(info_nce_val),
            bce_loss=float(bce_loss),
            hit_prob=float(hit_prob),
            txt_embed_delta_norm=txt_embed_norm,
            stop_reason="single_step_mode4",
        )
    ]

    return PerturbationResult(
        target_id="",
        rounds=rounds_log,
        final_image=list(image_feat),
        final_text=current_text,
    )


__all__ = [
    "PerturbationResult",
    "PerturbationRound",
    "single_step_perturb",
    "interactive_perturb_target",
    "get_visual_text_masks",
]
