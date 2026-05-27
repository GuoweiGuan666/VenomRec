"""Image / feature perturbation helpers with PSNR tracking."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def masked_pgd_image(
    x_img_or_feat: Sequence[float],
    mask: Sequence[bool],
    target_feat: Sequence[float],
    eps: float,
    iters: int,
    psnr_min: float,
    peak: Optional[float] = None,
) -> Tuple[List[float], float, float, List[float]]:
    """Masked projected gradient descent with PSNR-based early stopping."""

    orig = [float(v) for v in x_img_or_feat]
    tgt = [float(v) for v in target_feat]
    msk = [bool(v) for v in mask]
    n = min(len(orig), len(tgt), len(msk))
    if n == 0:
        return orig, float("inf"), 0.0, []

    orig = orig[:n]
    tgt = tgt[:n]
    msk = msk[:n]

    pert = orig[:]
    step = eps / max(int(iters), 1)
    psnr_history: List[float] = []

    data_range = peak if peak is not None else max(orig) - min(orig)
    if data_range <= 0:
        data_range = 1.0

    for _ in range(int(iters)):
        prev = pert[:]
        grad = [p - t for p, t in zip(pert, tgt)]
        for i in range(n):
            if not msk[i]:
                continue
            g = grad[i]
            if g > 0:
                pert[i] -= step
            elif g < 0:
                pert[i] += step
            low, high = orig[i] - eps, orig[i] + eps
            pert[i] = max(min(pert[i], high), low)

        mse = sum((o - p) ** 2 for o, p in zip(orig, pert)) / n
        psnr = 10 * math.log10((data_range ** 2) / (mse + 1e-12))
        logging.debug("[image-perturb] iter psnr=%.2f", psnr)
        if psnr < psnr_min:
            pert = prev
            break
        psnr_history.append(psnr)

    mse_final = sum((o - p) ** 2 for o, p in zip(orig, pert)) / n
    psnr_final = 10 * math.log10((data_range ** 2) / (mse_final + 1e-12))
    psnr_history.append(psnr_final)

    changed_idx = [i for i, (o, p, m) in enumerate(zip(orig, pert, msk)) if m and abs(o - p) > 1e-8]
    coverage = len(changed_idx) / max(sum(1 for v in msk if v), 1)
    logging.info("[image-perturb] coverage %.2f%%", coverage * 100)
    return pert, psnr_final, coverage, psnr_history


class ImagePerturber:
    """Convenience wrapper around :func:`masked_pgd_image`."""

    def __init__(
        self,
        eps: float = 0.1,
        iters: int = 3,
        psnr_min: float = 30.0,
        strategy: str = "cosine",
    ) -> None:
        self.eps = eps
        self.iters = iters
        self.psnr_min = psnr_min
        self.strategy = strategy

    def perturb(
        self,
        image: Sequence[float],
        mask: Optional[Sequence[bool]] = None,
        target_feat: Optional[Sequence[float]] = None,
    ) -> Tuple[List[float], float, float, Dict[str, float | List[float]]]:
        x = [float(v) for v in getattr(image, "flatten", lambda: image)()]
        if not x:
            return x, float("inf"), 0.0, {"coverage": 0.0, "psnr_history": []}
        if mask is None or len(mask) == 0:
            mask = [True] * len(x)
        else:
            assert len(mask) == len(x), "mask length must equal number of visual tokens"
        if target_feat is None:
            target_feat = [0.0] * len(x)

        if self.strategy == "cosine":
            pert, psnr, eps_used, metrics = _cosine_aligned_step(
                x,
                target_feat,
                mask,
                eps=self.eps,
            )
            return pert, psnr, eps_used, metrics

        pert, psnr, coverage, history = masked_pgd_image(x, mask, target_feat, self.eps, self.iters, self.psnr_min)
        eps_used = max((abs(a - b) for a, b in zip(x, pert)), default=0.0)
        metrics: Dict[str, float | List[float]] = {"coverage": coverage, "psnr_history": history}
        return pert, psnr, eps_used, metrics


__all__ = ["ImagePerturber", "masked_pgd_image"]


def _cosine_aligned_step(
    x_feat: Sequence[float],
    target_feat: Sequence[float],
    mask: Sequence[bool],
    *,
    eps: float,
) -> Tuple[List[float], float, float, Dict[str, float | List[float]]]:
    """Single-step perturbation that pushes masked dimensions toward the target direction.

    The update moves each masked component towards ``target_feat`` while respecting the
    L_inf budget ``eps``.
    """

    orig = np.asarray(x_feat, dtype=np.float32)
    tgt = np.asarray(target_feat, dtype=np.float32)
    msk = np.asarray(mask, dtype=bool)
    if orig.size == 0 or tgt.size == 0 or msk.size == 0:
        return orig.tolist(), float("inf"), 0.0, {"coverage": 0.0, "psnr_history": []}

    n = min(orig.size, tgt.size, msk.size)
    orig = orig[:n]
    tgt = tgt[:n]
    msk = msk[:n]

    delta = tgt - orig
    delta[~msk] = 0.0
    delta = np.clip(delta, -eps, eps)

    updated = orig + delta

    mse = float(np.mean((orig - updated) ** 2))
    data_range = float(np.max(orig) - np.min(orig))
    if data_range <= 0:
        data_range = 1.0
    psnr = 10 * math.log10((data_range ** 2) / (mse + 1e-12))
    eps_used = float(np.max(np.abs(updated - orig)))
    changed = int(np.count_nonzero(np.abs((updated - orig)[msk]) > 1e-8))
    coverage = float(changed) / max(int(msk.sum()), 1)
    metrics: Dict[str, float | List[float]] = {"coverage": coverage, "psnr_history": [psnr]}
    return updated.tolist(), psnr, eps_used, metrics
