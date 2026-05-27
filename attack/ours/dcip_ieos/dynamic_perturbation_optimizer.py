"""Adaptive controller for perturbation budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AdaptiveBudgetController:
    img_eps: float
    txt_ratio: float
    embed_eps: float = 0.0
    min_img_eps: float = 0.005
    max_img_eps: float = 0.2
    min_txt_ratio: float = 0.05
    max_txt_ratio: float = 0.4
    min_embed_eps: float = 0.0
    max_embed_eps: float = 0.05
    up_factor: float = 1.15
    down_factor: float = 0.85
    history: List[Dict[str, float]] = field(default_factory=list)
    prev_info_nce: float | None = None
    prev_sim: float | None = None
    prev_alignment: float | None = None
    prev_bce: float | None = None
    prev_hit: float | None = None
    bce_tolerance: float = 1e-3
    sim_tolerance: float = 1e-4
    align_tolerance: float = 1e-4
    info_tolerance: float = 1e-4
    hit_tolerance: float = 1e-3
    base_img_eps: float = field(init=False)
    base_txt_ratio: float = field(init=False)
    base_embed_eps: float = field(init=False)

    def __post_init__(self) -> None:
        self.base_img_eps = float(self.img_eps)
        self.base_txt_ratio = float(self.txt_ratio)
        self.base_embed_eps = float(self.embed_eps)

    def current(self) -> tuple[float, float, float]:
        return self.img_eps, self.txt_ratio, self.embed_eps

    def update(
        self,
        *,
        sim: float,
        info_nce: float,
        alignment: float,
        bce: float,
        hit_prob: float,
    ) -> None:
        if self.prev_sim is None:
            trend_value = 0.0
            self.history.append(
                {
                    "sim": float(sim),
                    "info_nce": float(info_nce),
                    "alignment": float(alignment),
                    "bce": float(bce),
                    "hit": float(hit_prob),
                    "trend": trend_value,
                    "img_eps": float(self.img_eps),
                    "txt_ratio": float(self.txt_ratio),
                    "embed_eps": float(self.embed_eps),
                }
            )
            self.prev_info_nce = info_nce
            self.prev_sim = sim
            self.prev_alignment = alignment
            self.prev_bce = bce
            self.prev_hit = hit_prob
            return

        trend_up = sim >= (self.prev_sim + self.sim_tolerance)
        if self.prev_alignment is not None and alignment > self.prev_alignment + self.align_tolerance:
            trend_up = False
        if self.prev_info_nce is not None and info_nce > self.prev_info_nce + self.info_tolerance:
            trend_up = False
        if self.prev_bce is not None and bce > self.prev_bce + self.bce_tolerance:
            trend_up = False
        if self.prev_hit is not None and hit_prob < self.prev_hit - self.hit_tolerance:
            trend_up = False

        trend = "up" if trend_up else "down"
        if trend == "up":
            self.img_eps = min(self.img_eps * self.up_factor, self.max_img_eps)
            self.txt_ratio = min(self.txt_ratio * self.up_factor, self.max_txt_ratio)
            self.embed_eps = min(self.embed_eps * self.up_factor, self.max_embed_eps)
        else:
            self.img_eps = max(self.base_img_eps, self.min_img_eps)
            self.txt_ratio = max(self.base_txt_ratio, self.min_txt_ratio)
            self.embed_eps = max(self.base_embed_eps, self.min_embed_eps)
        self.history.append(
            {
                "sim": float(sim),
                "info_nce": float(info_nce),
                "alignment": float(alignment),
                "bce": float(bce),
                "hit": float(hit_prob),
                "trend": 1.0 if trend == "up" else -1.0,
                "img_eps": float(self.img_eps),
                "txt_ratio": float(self.txt_ratio),
                "embed_eps": float(self.embed_eps),
            }
        )
        self.prev_info_nce = info_nce
        self.prev_sim = sim
        self.prev_alignment = alignment
        self.prev_bce = bce
        self.prev_hit = hit_prob


__all__ = ["AdaptiveBudgetController"]
