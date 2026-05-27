"""Light weight adapter around a victim model.

The real DCIP-IEOS codebase queries a fairly heavy VIP5 model in order to
obtain image and text representations.  The unit tests in this kata only
require minimal behaviour so this module provides a tiny wrapper that mirrors
the original API while remaining dependency free.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import os
import sys
import logging
import traceback

try:
    import numpy as np
    import torch
except Exception:  # pragma: no cover - torch/numpy may be absent
    np = None  # type: ignore
    torch = None  # type: ignore


class VictimAdapter:
    """Thin wrapper exposing a subset of the VIP5 interface.

    Parameters
    ----------
    model:
        Victim model.  The adapter expects ``model`` to expose ``encoder`` with
        ``visual_embedding`` and ``embed_tokens`` methods, matching the layout of
        the research code.  The object is treated as opaque and is only used
        when both ``numpy`` and ``torch`` are available.
    tokenizer:
        Optional tokenizer providing a ``__call__`` method returning a mapping
        with an ``input_ids`` tensor.
    device:
        Device on which to perform computations.  Defaults to ``"cpu"``.
    """

    def __init__(self, model: Any = None, tokenizer: Any | None = None, device: str = "cpu", ckpt_path: str | None = None, dataset: str | None = None, image_feature_type: str = "vitb32", backbone: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ckpt_path = ckpt_path
        self.dataset = dataset or "unknown"
        self.image_feature_type = image_feature_type
        self.backbone = backbone or 't5-base'
        self._vis_cache: Dict[str, Any] = {}

        # Try to build a real VIP5 model if only a checkpoint is provided
        if (self.model is None) and (self.ckpt_path is not None):
            logging.info(
                "[victim-adapter] init: attempting VIP5 load (ckpt=%s, backbone=%s, device=%s)",
                str(self.ckpt_path), str(self.backbone), str(self.device),
            )
            try:
                # If torch is unavailable in the runtime, surface that explicitly
                if 'torch' not in sys.modules:
                    try:
                        import torch  # noqa: F401
                    except Exception as e:  # pragma: no cover
                        logging.warning("[victim-adapter] torch unavailable: %s", str(e))
                        raise
                self._build_vip5_from_ckpt()
            except Exception:
                # Keep lightweight adapter if building fails (details logged in _build_*)
                pass
        elif self.ckpt_path is None:
            logging.info("[victim-adapter] init: no ckpt provided; staying lightweight")

    # ------------------------------------------------------------------
    def d_model(self) -> int:
        """Best effort retrieval of the model's embedding dimension."""
        if self.model is None:
            return 0
        try:
            cfg = getattr(self.model, 'config', None)
            if cfg is not None and hasattr(cfg, 'd_model'):
                return int(getattr(cfg, 'd_model'))
            weight = self.model.encoder.embed_tokens.weight  # type: ignore[attr-defined]
            return int(weight.shape[1])
        except Exception:  # pragma: no cover - fallback for unexpected models
            return 0

    # ------------------------------------------------------------------
    def _build_vip5_from_ckpt(self) -> None:
        """Best-effort: construct VIP5Tuning + tokenizer and load a state_dict ckpt.

        Relies on local src/ modules; falls back silently on failure.
        """
        try:
            # Ensure repo root and src/ on sys.path so that `src.*` and legacy
            # absolute imports used by training code (e.g. `from utils import ...`)
            # both work when invoked from our lightweight pipeline.
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            src_root = os.path.join(proj_root, "src")
            for p in (proj_root, src_root):
                if p not in sys.path:
                    sys.path.append(p)
            logging.info("[victim-adapter] building VIP5 from ckpt=%s", str(self.ckpt_path))
            import torch
            from transformers import T5Tokenizer
            from src.param import Config
            from src.trainer_base import TrainerBase
            from src.model import VIP5Tuning

            # Minimal args to create a config consistent with training
            args = Config(
                backbone=self.backbone,
                tokenizer='t5',
                use_adapter=True,
                reduction_factor=8,
                use_single_adapter=True,
                use_vis_layer_norm=True,
                add_adapter_cross_attn=True,
                use_lm_head_adapter=True,
                image_feature_type=self.image_feature_type,
                image_feature_size_ratio=2,
                losses='sequential,direct,explanation',
                dropout=0.1,
                gen_max_length=64,
            )
            # Provide minimal runtime attributes expected by training utilities
            if not hasattr(args, 'distributed'):
                setattr(args, 'distributed', False)
            if not hasattr(args, 'gpu'):
                setattr(args, 'gpu', 0)
            if not hasattr(args, 'multiGPU'):
                setattr(args, 'multiGPU', False)
            if not hasattr(args, 'local_rank'):
                setattr(args, 'local_rank', 0)
            # Flags used by create_config/unfreeze paths
            if not hasattr(args, 'unfreeze_layer_norms'):
                setattr(args, 'unfreeze_layer_norms', False)
            if not hasattr(args, 'unfreeze_language_model'):
                setattr(args, 'unfreeze_language_model', False)
            if not hasattr(args, 'freeze_ln_statistics'):
                setattr(args, 'freeze_ln_statistics', False)
            if not hasattr(args, 'freeze_bn_statistics'):
                setattr(args, 'freeze_bn_statistics', False)
            if not hasattr(args, 'track_z'):
                setattr(args, 'track_z', False)
            if not hasattr(args, 'whole_word_embed'):
                setattr(args, 'whole_word_embed', True)
            if not hasattr(args, 'category_embed'):
                setattr(args, 'category_embed', True)
            if not hasattr(args, 'non_linearity'):
                # default activation used by adapters
                setattr(args, 'non_linearity', 'relu')
            trainer = TrainerBase(args)
            config = trainer.create_config()
            # Build model weights from T5 base
            model = VIP5Tuning.from_pretrained(args.backbone, config=config)
            if self.ckpt_path:
                sd = torch.load(self.ckpt_path, map_location=self.device)
                if isinstance(sd, dict):
                    # handle nested checkpoints: pick common keys
                    cand = None
                    for k in ("state_dict", "model_state_dict", "model", "module"):
                        if isinstance(sd.get(k, None), dict):
                            cand = sd[k]
                            break
                    if cand is None:
                        cand = sd
                    # Filter out keys with shape mismatches (e.g., tokenizer/vocab diffs)
                    try:
                        target = model.state_dict()
                        filt = {}
                        dropped = []
                        for k, v in cand.items():
                            tv = target.get(k, None)
                            if tv is not None and hasattr(tv, 'shape') and hasattr(v, 'shape'):
                                if tuple(tv.shape) == tuple(v.shape):
                                    filt[k] = v
                                else:
                                    dropped.append(k)
                            else:
                                # keep non-tensor or buffers when safe
                                filt[k] = v if tv is not None else filt.get(k, None)
                        model.load_state_dict(filt, strict=False)
                        if dropped:
                            logging.info("[victim-adapter] dropped %d mismatched keys (e.g., %s)", len(dropped), dropped[:4])
                    except Exception as e:
                        logging.warning("[victim-adapter] filtering state_dict failed: %s; falling back to strict=False", str(e))
                        model.load_state_dict(cand, strict=False)
            else:
                logging.info("[victim-adapter] no ckpt provided; using backbone pretrained weights only")
            model.eval()
            model.to(self.device)
            self.model = model
            # Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(args.backbone)
            logging.info("[victim-adapter] VIP5 loaded from ckpt=%s on %s backbone=%s", str(self.ckpt_path), str(self.device), str(self.backbone))
        except Exception as e:
            logging.warning("[victim-adapter] failed to load VIP5: %s", str(e))
            logging.warning("[victim-adapter] traceback:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    def _ensure_model_ready(self) -> bool:
        if torch is None or np is None:
            logging.warning("[victim-adapter] torch/numpy unavailable; cannot compute cross-modal saliency")
            return False
        if self.model is None or self.tokenizer is None:
            logging.warning("[victim-adapter] no victim model loaded; falling back to heuristic masks")
            return False
        try:
            device = torch.device(self.device if self.device else "cpu")
        except Exception:
            device = torch.device("cpu")
        if "cuda" in str(device) and not torch.cuda.is_available():
            logging.warning("[victim-adapter] requested cuda device %s but CUDA not available; using cpu", str(device))
            device = torch.device("cpu")
        self.device = str(device)
        self.model.to(device)
        self.model.eval()
        return True

    # ------------------------------------------------------------------
    def __call__(self, image: Any, text: str, output_attentions: bool = True, **kwargs) -> Dict[str, Any]:
        """Best-effort callable to supply cross-attention-like maps.

        Returns a dict with a "cross_attentions" 2-D list [V x T] when possible.
        If the heavy model/tokenizer are unavailable, returns an empty dict to
        trigger the downstream fallback.
        """
        if np is None or torch is None or self.model is None or self.tokenizer is None:
            return {}
        try:
            text_raw = str(text or "")
            batch = self.tokenizer(text_raw, return_tensors="pt")
            input_ids = batch["input_ids"].to(self.device)
            img_vec = np.asarray(image, dtype="float32")
            if img_vec.ndim > 1:
                img_vec = img_vec.reshape(-1)
            if img_vec.size == 0:
                return {}
            vis_feats = torch.as_tensor(img_vec[None, :], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                vis_tokens = self.model.encoder.visual_embedding(vis_feats[None, ...])
                if vis_tokens.dim() == 4:
                    vis_tokens = vis_tokens.view(1, -1, vis_tokens.shape[-1])
                txt_tokens = self.model.encoder.embed_tokens(input_ids)
                V = int(vis_tokens.shape[1])
                T_tok = int(txt_tokens.shape[1])
                if V == 0 or T_tok == 0:
                    return {}
                sim = torch.matmul(vis_tokens[0], txt_tokens[0].transpose(0, 1)).abs()
                cross = sim.detach().cpu().numpy()
            return {"cross_attentions": cross.tolist()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def compute_cross_modal_saliency(
        self,
        image_feat: Any,
        text: str,
        *,
        method: str = "rollout",
    ) -> Dict[str, Any]:
        """Return Grad-Rollout style saliency scores for visual & textual tokens."""

        if not self._ensure_model_ready():
            raise RuntimeError("victim model not available")

        assert torch is not None and np is not None  # mypy appeasement

        device = torch.device(self.device)
        text = str(text or "")
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
        input_ids_text = tokens["input_ids"].to(device)
        attn_mask_text = tokens.get("attention_mask")
        if attn_mask_text is None:
            attn_mask_text = torch.ones_like(input_ids_text)
        attn_mask_text = attn_mask_text.to(device)

        cfg = getattr(self.model.encoder, "config", None)
        feat_dim = int(getattr(cfg, "feat_dim", len(np.asarray(image_feat).reshape(-1))))
        n_vis_tokens = int(getattr(cfg, "n_vis_tokens", 1))

        image_arr = np.asarray(image_feat, dtype=np.float32).reshape(-1)
        if image_arr.size < feat_dim:
            pad = np.zeros(feat_dim, dtype=np.float32)
            pad[:image_arr.size] = image_arr
            image_arr = pad
        elif image_arr.size > feat_dim:
            image_arr = image_arr[:feat_dim]

        vis_feats = torch.from_numpy(image_arr).to(device=device, dtype=torch.float32).view(1, 1, feat_dim)
        with torch.no_grad():
            vis_tokens = self.model.encoder.visual_embedding(vis_feats)
            if vis_tokens.dim() == 4:
                vis_tokens = vis_tokens.view(vis_tokens.shape[0], -1, vis_tokens.shape[-1])
        vis_slots = int(vis_tokens.shape[1]) if vis_tokens.numel() > 0 else max(1, n_vis_tokens)

        text_len = int(input_ids_text.size(1))
        enc_len = vis_slots + text_len
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full((1, enc_len), pad_id, dtype=torch.long, device=device)
        input_ids[0, vis_slots:] = input_ids_text[0]

        attention_mask = torch.ones_like(input_ids)
        whole_word_ids = torch.zeros_like(input_ids)
        if text_len > 0:
            whole_word_ids[0, vis_slots:] = torch.arange(1, text_len + 1, device=device)
        category_ids = torch.zeros_like(input_ids)
        category_ids[0, :vis_slots] = 1

        decoder_input_ids = input_ids_text.clone().to(device)
        decoder_attention_mask = attn_mask_text.clone().to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                whole_word_ids=whole_word_ids,
                category_ids=category_ids,
                vis_feats=vis_feats,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=True,
                return_dict=True,
                task="sequential",
            )

        cross = getattr(outputs, "cross_attentions", None)
        if not cross:
            raise RuntimeError("victim model did not return cross-attentions")

        # Stack layers -> (layers, heads, tgt_len, enc_len)
        stack = torch.stack([att[0] for att in cross], dim=0)
        rollout = stack.mean(dim=1)  # average heads
        rollout = rollout.mean(dim=1)  # average decoder positions
        if method == "rollout":
            attn_scores = rollout.mean(dim=0)
        else:
            attn_scores = rollout.mean(dim=0)

        visual_scores = attn_scores[:vis_slots]
        if visual_scores.numel() == 0:
            visual_scores = torch.zeros(vis_slots, device=device)
        text_scores = attn_scores[vis_slots:vis_slots + text_len]
        if text_scores.numel() == 0:
            text_scores = torch.zeros(text_len, device=device)

        text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids_text[0].tolist())

        return {
            "visual_scores": visual_scores.detach().cpu().numpy(),
            "text_scores": text_scores.detach().cpu().numpy(),
            "text_tokens": text_tokens,
            "n_vis_slots": vis_slots,
        }

    # ------------------------------------------------------------------
    def encode_image_tokens(self, clip_feat: Any) -> Any:
        """Return visual tokens for ``clip_feat``.

        When the environment lacks ``torch``/``numpy`` the input is returned as
        is which keeps the adapter usable in the light-weight tests.
        """

        if np is None or torch is None or self.model is None:
            return clip_feat
        arr = np.asarray(clip_feat, dtype="float32")
        if arr.ndim == 1:
            arr = arr[None, :]
        tensor = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            out = self.model.encoder.visual_embedding(tensor[None, ...])  # type: ignore[attr-defined]
        return out.squeeze(0).cpu().numpy()
    
    # ------------------------------------------------------------------
    def extract_raw_image_feats(self, pixel: Any) -> Any:
        """Return raw visual features prior to projection.

        The heavy research code extracts patch/grid features from the vision
        backbone before they are fed into ``visual_embedding``.  In the light
        weight environment we simply return the input when the real model is not
        available which keeps the adapter functional for the unit tests.
        """

        if self.model is not None and hasattr(self.model, "extract_raw_image_feats"):
            try:
                with torch.no_grad():
                    feats = self.model.extract_raw_image_feats(pixel)
                if torch is not None and isinstance(feats, torch.Tensor):
                    return feats.detach()
                return feats
            except Exception:  # pragma: no cover - fall back to simple path
                pass
        if np is None:
            return pixel
        return np.asarray(pixel, dtype="float32")

    # ------------------------------------------------------------------
    def encode_text_tokens(self, text: str) -> Dict[str, Any]:
        """Return token level and pooled text embeddings."""

        if np is None or torch is None or self.model is None or self.tokenizer is None:
            vec = [float(ord(c)) for c in text]
            pooled = float(sum(vec) / len(vec)) if vec else 0.0
            return {"tokens": vec, "pooled": pooled}

        batch = self.tokenizer(text, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            tokens = self.model.encoder.embed_tokens(batch["input_ids"])  # type: ignore[attr-defined]
        arr = tokens.squeeze(0).cpu().numpy()
        pooled = arr.mean(axis=0)
        return {"tokens": arr, "pooled": pooled}

    # ------------------------------------------------------------------
    def pooled_image(self, img_tokens: Any) -> Any:
        """Return mean pooled image representation."""
        if np is None:
            return img_tokens
        arr = np.asarray(img_tokens, dtype="float32")
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=0)

    # ------------------------------------------------------------------
    def compute_fused_repr(self, image_feat: Any, text_str: str, repr_layer: str = "enc_last") -> Any:
        """Return a fused user–item representation h(u,i). 

        Best-effort implementation:
        - If a real victim model/tokenizer is available, try to use its encoder
          outputs; otherwise fall back to mean-pooled visual features possibly
          blended with pooled token embeddings.
        - The return is a 1-D numpy array (float32).
        """
        if np is None:
            # Fallback: just return the raw feature vector
            return image_feat

        # If a real model + tokenizer are available, use encoder forward to obtain a fused embedding
        try:
            import torch
            if self.model is not None and self.tokenizer is not None:
                with torch.no_grad():
                    enc = self.model.encoder
                    # Tokenize text
                    batch = self.tokenizer(str(text_str) if text_str is not None else "", return_tensors="pt")
                    input_ids = batch["input_ids"].to(self.device)
                    B, L = input_ids.shape
                    # Visual feats: expect [B, V_W_L, feat_dim]; use V_W_L=1, feat_dim from image_feat
                    img = np.asarray(image_feat, dtype="float32").reshape(1, 1, -1)
                    vis_feats = torch.as_tensor(img, device=self.device)
                    # category_ids: mark first K positions for visual tokens
                    n_vis_tokens = int(getattr(enc.config, 'n_vis_tokens', 2))
                    k = max(1, n_vis_tokens)
                    cat = torch.zeros((B, L), dtype=torch.long, device=self.device)
                    cat[:, :min(k, L)] = 1
                    # whole_word_ids: zeros
                    ww = torch.zeros((B, L), dtype=torch.long, device=self.device)
                    out = enc(
                        input_ids=input_ids,
                        whole_word_ids=ww,
                        category_ids=cat,
                        vis_feats=vis_feats,
                        return_dict=True,
                    )
                    hid = out.last_hidden_state  # [B,L,d]
                    fused = hid.mean(dim=1).detach().cpu().numpy().reshape(-1)
                    return fused.astype("float32")
        except Exception:
            pass

        # Lightweight fallback: average pooled image embedding with simple pooled text signal
        try:
            img_tok = self.encode_image_tokens(image_feat)
            img_vec = self.pooled_image(img_tok)
        except Exception:
            img_vec = np.asarray(image_feat, dtype="float32").reshape(-1)
        txt_vec = None
        try:
            tinfo = self.encode_text_tokens(text_str)
            pooled = tinfo.get("pooled")
            if isinstance(pooled, (list, tuple)):
                txt_vec = np.asarray(pooled, dtype="float32").reshape(-1)
            elif isinstance(pooled, (int, float)):
                txt_vec = np.full_like(np.asarray(img_vec, dtype="float32"), float(pooled))
        except Exception:
            txt_vec = None
        img_vec = np.asarray(img_vec, dtype="float32").reshape(-1)
        if txt_vec is None:
            return img_vec
        n = min(len(img_vec), len(txt_vec))
        if n <= 0:
            return img_vec
        fused = (img_vec[:n] + txt_vec[:n]) / 2.0
        return fused.astype("float32")
