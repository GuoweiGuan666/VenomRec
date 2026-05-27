#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for mining the competition pool for the DCIP-IEOS attack.

This module contains two different pieces of functionality:

* :class:`PoolMiner` – a tiny helper used by the poisoning pipeline to
  persist the mined information to disk.  The class exposes a
  :meth:`build_competition_pool` method which performs the actual
  computation of the competition set for each target item.
* A small command line interface (kept for backwards compatibility with the
  original project) which simply executes the full poisoning pipeline when
  run as a script.

The real project uses a fairly involved procedure relying on a frozen VIP5
model and popularity statistics.  Re‑creating the exact original behaviour is
out of scope for the unit tests in this kata, however the implementation below
captures the essential logic:

1.  For every target item we compute a cosine‑similarity based nearest
    neighbour set ``C(t)``.
2.  The mean embedding of this set acts as the anchor embedding
    ``E_avg(C)``.
3.  Frequently occurring tokens in the neighbours' textual metadata are used
    as mined ``keywords``.

The resulting structure is serialised to ``competition_pool.json`` under the
provided cache directory so that subsequent stages of the pipeline can easily
consume it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import gzip
import pickle
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional heavy dependencies
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - fallback if sklearn is missing
    KMeans = PCA = TfidfVectorizer = None

try:  # torch is optional at runtime
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = None


try:  # pragma: no cover - PIL may be absent
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


def load_image_feature_any(
    asin: Any,
    img_feat_entry: Any,
    data_root: str,
    dataset: str,
    victim_model: Any,
    preprocess: Optional[Callable[[Any], Any]] = None,
):
    """Return image features regardless of input form.

    The real project stores image features either as pre-computed vectors or as
    relative paths to image files.  The light-weight implementation mimics this
    behaviour by supporting both options.  When a path is supplied the function
    attempts to load the image and, if available, queries the ``victim_model``
    for raw features.  Falling back to a flattened pixel representation keeps
    the helper usable even when heavy dependencies such as ``torch`` or
    ``PIL`` are missing.
    """

    src = "vector"
    if isinstance(img_feat_entry, str):
        abs_path = (
            img_feat_entry
            if os.path.isabs(img_feat_entry)
            else os.path.join(data_root, dataset, img_feat_entry)
        )
        if not os.path.isfile(abs_path):  # pragma: no cover - sanity check
            raise FileNotFoundError(f"Image path not found: {abs_path}")
        if Image is None:  # pragma: no cover - PIL missing
            raise RuntimeError("PIL is required to load image paths")
        img = Image.open(abs_path).convert("RGB")
        if preprocess is not None:
            pixel = preprocess(img)
        else:
            arr = np.asarray(img, dtype="float32") / 255.0
            pixel = torch.from_numpy(np.transpose(arr, (2, 0, 1))) if torch is not None else np.transpose(arr, (2, 0, 1))
        if victim_model is not None and torch is not None and hasattr(victim_model.encoder, "visual_embedding"):
            tensor = pixel if isinstance(pixel, torch.Tensor) else torch.as_tensor(pixel)
            tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
            tensor = tensor.to(getattr(victim_model, "device", "cpu"))
            with torch.no_grad():
                emb = victim_model.encoder.visual_embedding(tensor)  # type: ignore[attr-defined]
            if emb.ndim == 4:
                emb = emb.view(emb.shape[0], -1, emb.shape[-1])
            feats_np = emb.detach().cpu().numpy()[0]
            src = "path->encoder"
        else:
            feats_np = np.asarray(pixel, dtype="float32").reshape(-1)
            src = "path->pixels"
    else:
        feats_np = np.asarray(img_feat_entry, dtype="float32")
        if feats_np.ndim == 1 and victim_model is not None:
            d_model = getattr(victim_model, "d_model", lambda: feats_np.shape[0])()
            feats_np = np.tile(feats_np[None, :], (max(8, 1), 1))
            src = "vector->repeat"

    assert np.isfinite(feats_np).all(), "Non-finite image features"
    if victim_model is not None and hasattr(victim_model, "d_model") and feats_np.ndim == 2:
        d_model = getattr(victim_model, "d_model")
        d_model = d_model() if callable(d_model) else int(d_model)
        assert (
            feats_np.shape[1] == d_model and feats_np.shape[0] >= 1
        ), f"img_emb shape {feats_np.shape}, expect (*,{d_model})"
        logging.info("Encoded %s -> L_img=%d", asin, feats_np.shape[0])
    logging.debug(
        "Loaded image feats: asin=%s shape=%s from %s",
        asin,
        feats_np.shape,
        src,
    )
    return feats_np, src

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def forward_inference(model: Any, image_input: Any, text_input: Any) -> Dict[str, Any]:
    """Run frozen VIP5 visual/text embedding inference for pool mining."""

    if torch is None:
        raise RuntimeError("torch is required when a victim model is provided")

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if isinstance(image_input, np.ndarray):
        image_input = torch.from_numpy(image_input)
    if isinstance(text_input, np.ndarray):
        text_input = torch.from_numpy(text_input)

    device = next(model.parameters()).device
    image_input = image_input.to(device)
    text_input = text_input.to(device)

    with torch.no_grad():
        image_embedding = model.encoder.visual_embedding(image_input)
        text_embedding = model.encoder.embed_tokens(text_input)

    return {
        "image_embedding": image_embedding,
        "text_embedding": text_embedding,
    }


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON serialisable structure."""

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class PoolMiner:
    """Compute and persist competition pool information.

    The miner expects each entry in the pool to contain an ``embedding`` field
    (a list or NumPy compatible sequence) and optionally ``id``, ``text`` and
    ``popularity`` fields.  The latter is used only as a light‑weight weighting
    factor when selecting the nearest neighbours.
    """

    def __init__(self, cache_dir: str, dataset: str = "unknown") -> None:
        """Initialise the miner.

        Parameters
        ----------
        cache_dir:
            Directory used to store cached artefacts.
        dataset:
            Name of the dataset; determines the output file name.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.out_path = os.path.join(
            self.cache_dir, f"competition_pool_{dataset}.json"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def build_competition_pool(
        pool: Iterable[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Build a competition pool description.

        Parameters
        ----------
        pool:
            Iterable of item dictionaries.  Each dictionary is expected to
            contain an ``embedding`` field and may optionally expose ``id``,
            ``text``/``keywords`` and ``popularity`` fields.
        top_k:
            Number of nearest neighbours to keep for each target item.

        Returns
        -------
        list of dict
            For every target item ``t`` the returned list contains a dictionary
            with the following keys:

            ``target``
                Identifier of the target item.
            ``competitors``
                Identifiers of the ``top_k`` nearest neighbours ``C(t)``.
            ``anchor``
                Mean embedding of the competitors ``E_avg(C)``.
            ``keywords``
                Mined keywords extracted from the competitors' textual
                metadata.
        """

        pool_list = list(pool)
        if not pool_list:
            return []

        # ------------------------------------------------------------------
        # Prepare embeddings and popularity weights
        embeddings = np.stack(
            [np.asarray(entry.get("embedding", []), dtype=float) for entry in pool_list]
        )
        pops = np.asarray(
            [float(entry.get("popularity", 0.0)) for entry in pool_list], dtype=float
        )

        # normalise popularity to [0,1] to be used as weights
        if pops.size and pops.max() > 0:
            pops = pops / pops.max()
        else:
            pops = np.ones(len(pool_list), dtype=float)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # avoid division by zero for degenerate embeddings
        norms[norms == 0] = 1e-12
        norm_emb = embeddings / norms
        sim_matrix = norm_emb @ norm_emb.T

        results: List[Dict[str, Any]] = []
        for idx, entry in enumerate(pool_list):
            sims = sim_matrix[idx] * pops  # incorporate popularity

            # sort descending, remove the item itself
            neighbour_indices = np.argsort(-sims)
            neighbour_indices = [i for i in neighbour_indices if i != idx][:top_k]

            competitors = [pool_list[i].get("id", i) for i in neighbour_indices]

            # Anchor embedding
            if neighbour_indices:
                vecs = [embeddings[i] for i in neighbour_indices]
                if not vecs:
                    raise RuntimeError(
                        f"No neighbor embeddings for target {entry.get('id', idx)}"
                    )
                anchor_vec = np.mean(np.stack(vecs, axis=0), axis=0)
                d_model = embeddings.shape[1]
                assert (
                    anchor_vec.ndim == 1 and anchor_vec.shape[0] == d_model
                ), (
                    f"Bad anchor_dim={anchor_vec.shape}, expect [d_model={d_model}]. "
                    f"neighbor shapes={[np.asarray(v).shape for v in vecs[:5]]}"
                )
            else:
                anchor_vec = np.zeros_like(embeddings[0])

            # Mine keywords from text/keywords fields
            tokens: List[str] = []
            for i in neighbour_indices:
                text = pool_list[i].get("keywords") or pool_list[i].get("text", "")
                if isinstance(text, (list, tuple)):
                    tokens.extend(str(t).lower() for t in text)
                else:
                    tokens.extend(str(text).lower().split())
            counts = Counter(tokens)
            keywords = [w for w, _ in counts.most_common(5)]

            results.append(
                {
                    "target": entry.get("id", idx),
                    "competitors": competitors,
                    "anchor": _to_jsonable(anchor_vec),
                    "keywords": keywords,
                }
            )

        return results

    # ------------------------------------------------------------------
    def save(self, pool: Iterable[Dict[str, Any]]) -> None:
        """Build the competition pool and serialize it to ``out_path``.

        Only entries containing non-empty ``neighbors`` and ``anchor`` fields are
        persisted.  Missing or empty fields trigger a warning and the
        corresponding targets are skipped.
        """

        raw_data = self.build_competition_pool(pool)
        validated: List[Dict[str, Any]] = []
        for entry in raw_data:
            neighbors = entry.get("competitors", [])
            anchor = entry.get("anchor", [])
            if not neighbors or not anchor:
                logging.warning(
                    "Skipping target %s due to empty neighbors/anchor",
                    entry.get("target"),
                )
                continue
            validated.append(
                {
                    "target": entry.get("target"),
                    "neighbors": neighbors,
                    "anchor": anchor,
                    "keywords": entry.get("keywords", []),
                }
            )

        obj_jsonable = _to_jsonable(validated)
        try:
            json.dumps(obj_jsonable)
        except TypeError as e:
            raise AssertionError(f"not JSON-safe: {e}")

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(obj_jsonable, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Anchor computation using victim model
# ---------------------------------------------------------------------------
def compute_anchor_with_victim(
    model: Any,
    tokenizer: Any,
    neighbor_items: Iterable[Dict[str, Any]],
    *,
    w_img: float = 0.6,
    w_txt: float = 0.4,
) -> np.ndarray:
    """Return an anchor embedding based on visual/textual tokens.

    ``neighbor_items`` is an iterable of dictionaries that contain at least
    ``image`` (or ``image_feat``) and ``text`` entries.  The helper averages the
    image and text token representations separately before fusing them.  The
    function is intentionally tolerant and falls back to simple numeric
    encodings when the heavy victim model or tokenizer are unavailable.
    """

    from .saliency_extractor import project_visual_tokens  # Local import

    img_vecs: List[Any] = []
    txt_vecs: List[Any] = []

    for it in neighbor_items:
        feats = it.get("image") or it.get("image_feat")
        if feats is not None:
            vis_tokens = project_visual_tokens(model, feats)
            if torch is not None and isinstance(vis_tokens, torch.Tensor):
                img_vecs.append(vis_tokens.mean(dim=1))
            else:
                arr = np.asarray(vis_tokens)
                img_vecs.append(arr.mean(axis=1) if arr.ndim == 3 else arr)

        text = it.get("text", "")
        if (
            tokenizer is not None
            and hasattr(model, "encoder")
            and hasattr(model.encoder, "embed_tokens")
            and torch is not None
        ):
            try:
                batch = tokenizer(text, return_tensors="pt")
                input_ids = batch.get("input_ids")
                if isinstance(input_ids, torch.Tensor):
                    with torch.no_grad():
                        t_tokens = model.encoder.embed_tokens(input_ids.to(model.device))  # type: ignore[attr-defined]
                    txt_vecs.append(t_tokens.mean(dim=1))
                    continue
            except Exception:
                pass
        arr = np.asarray([float(ord(c)) for c in str(text)], dtype="float32")
        txt_vecs.append(arr.mean() if arr.ndim == 1 else arr.mean(axis=0))

    if not img_vecs and not txt_vecs:
        return np.zeros(0, dtype="float32")

    if torch is not None and any(isinstance(v, torch.Tensor) for v in img_vecs + txt_vecs):
        img = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in img_vecs],
            dim=0,
        ).mean(dim=0)
        txt = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in txt_vecs],
            dim=0,
        ).mean(dim=0)
        anchor = (w_img * img + w_txt * txt).squeeze(0)
        anchor_np = anchor.detach().cpu().numpy()
    else:
        img = np.mean(np.stack([np.asarray(v) for v in img_vecs], axis=0), axis=0)
        txt = np.mean(np.stack([np.asarray(v) for v in txt_vecs], axis=0), axis=0)
        anchor_np = w_img * img + w_txt * txt
        anchor_np = np.asarray(anchor_np).reshape(-1)

    d_model = getattr(model, "d_model", lambda: anchor_np.shape[0])()
    assert anchor_np.ndim == 1 and anchor_np.shape[0] == d_model, (
        f"Bad anchor shape: {anchor_np.shape}; expected ({d_model},)"
    )
    return anchor_np



# ----------------------------------------------------------------------

def build_competition_pool(
    dataset: str,
    pop_path: str,
    targets: Iterable[int],
    model: Any,
    *,
    cache_dir: Optional[str] = None,
    item_loader: Optional[Callable[[int], Dict[str, Any]]] = None,
    w_img: float = 0.6,
    w_txt: float = 0.4,
    pca_dim: Optional[int] = None,
    kmeans_k: int = 8,
    c_size: int = 20,
    keyword_top: int = 50,
    id_mode: str = "auto",
    min_keywords: int = 5,
    allow_missing_image: bool = True,
    allow_missing_text: bool = True,
    feat_root: Optional[str] = None,
    feat_backbone: Optional[str] = None,
    min_vis_tokens: int = 1,
    allow_path_images: bool = False,
    img_dim: int = 512,
) -> Dict[str, Any]:
    """Build and cache the competition pool for a dataset.

    Parameters
    ----------
    dataset:
        Name of the dataset used purely for book‑keeping.
    pop_path:
        Path to the text file containing the high popularity items.  The file
        is expected to list entries of the form ``Item: <ASIN> (ID: <idx>)``.
    targets:
        Iterable of low popularity item IDs to attack.  Neighbours are mined
        for these targets from the high popularity set.
    model:
        A (possibly stubbed) VIP5 model to be passed to
        :func:`forward_inference`.
    cache_dir:
        Directory where the resulting JSON cache will be written.  Defaults to
        ``attack/ours/dcip_ieos/caches``.
    item_loader:
        Optional callable ``item_loader(item_id) -> dict`` returning the raw
        ``image_input``, ``text_input`` and ``text`` fields for the given item.
        When omitted a minimal stub returning empty arrays is used which keeps
        the function functional for unit tests without the real dataset.
    w_img, w_txt:
        Weights for combining the image and text embeddings.
    pca_dim:
        If not ``None`` the fused embeddings are reduced using PCA to this
        dimensionality.
    kmeans_k:
        Number of clusters used for an optional KMeans step.  If the clustering
        fails for any reason the code silently falls back to a single cluster
        mode.
    c_size:
        Number of nearest neighbours to keep for each target item.
    keyword_top:
        Number of keywords to extract using TF‑IDF.
    id_mode:
        One of ``'auto'``, ``'asin'`` or ``'id'`` controlling how identifiers
        in the popularity files are interpreted. ``'auto'`` prefers ASINs and
        falls back to numeric IDs via the dataset mapping.
    min_keywords:
        Minimum number of keywords required for a target. Targets falling
        short are supplemented with synthetic placeholders.
    allow_missing_image:
        When ``True`` (default) missing image features are replaced with
        zero vectors. If ``False`` a missing feature triggers a ``KeyError``.
    allow_missing_text:
        When ``True`` (default) targets without text fall back to empty
        strings and are tracked in ``missing_text_ids``. If ``False`` a
        missing text field raises a ``KeyError``.
    feat_root / feat_backbone:
        Location of pre-computed ``.npy`` image features.  Features are
        resolved as ``<feat_root>/<feat_backbone>/<dataset>/<id>.npy``.
    min_vis_tokens:
        Minimum number of visual tokens accepted; kept for API compatibility.
    allow_path_images:
        Allow decoding image file paths when provided by the loader. Disabled
        by default to avoid degenerate single-token vectors.
    img_dim:
        Fallback dimensionality used when an image feature file is missing.
    """

    # ------------------------------------------------------------------
    # Resolve dataset metadata for textual fallbacks and id mapping
    dataset_dir = os.path.join(PROJ_ROOT, "data", dataset)
    meta_path = os.path.join(dataset_dir, "meta.json.gz")
    review_path = os.path.join(dataset_dir, "review_splits.pkl")
    datamap_path = os.path.join(dataset_dir, "datamaps.json")

    meta_titles: Dict[str, str] = {}
    id2asin: Dict[str, str] = {}
    asin2id: Dict[str, str] = {}

    if os.path.exists(datamap_path):
        try:
            with open(datamap_path, "r", encoding="utf-8") as f:
                dm = json.load(f)
            mapping = None
            for key in ("item2id", "asin2id", "asin_to_id"):
                if isinstance(dm.get(key), dict):
                    mapping = dm[key]
                    break
            if mapping:
                for asin, idx in mapping.items():
                    id2asin[str(idx)] = str(asin)
                    asin2id[str(asin).upper()] = str(idx)
            mapping = None
            for key in ("id2asin", "id_to_asin"):
                if isinstance(dm.get(key), dict):
                    mapping = dm[key]
                    break
            if mapping:
                for idx, asin in mapping.items():
                    id2asin[str(idx)] = str(asin)
                    asin2id[str(asin).upper()] = str(idx)
        except Exception:  # pragma: no cover - corrupted datamap
            pass

    if os.path.exists(meta_path):
        try:
            with gzip.open(meta_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:  # pragma: no cover - fallback for non-JSON lines
                        obj = eval(line)
                    asin = str(obj.get("asin") or obj.get("id") or "")
                    title = str(obj.get("title", ""))
                    if asin:
                        meta_titles[asin] = title
                    idx_val = obj.get("id")
                    if idx_val is not None and str(idx_val) not in id2asin:
                        id2asin[str(idx_val)] = asin
        except Exception:  # pragma: no cover - corrupted meta file
            pass

    review_texts: Dict[str, str] = {}
    if os.path.exists(review_path):
        try:
            with open(review_path, "rb") as f:
                reviews = pickle.load(f)

            def _collect(container: Any) -> None:
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, (list, tuple)) and v:
                            v0 = v[0]
                        else:
                            v0 = v
                        if isinstance(v0, dict):
                            txt = str(v0.get("text") or v0.get("reviewText") or "")
                        else:
                            txt = str(v0)
                        review_texts[str(k)] = txt
                elif isinstance(container, list):
                    for v in container:
                        if isinstance(v, dict):
                            k = v.get("asin") or v.get("item") or v.get("id")
                            txt = str(v.get("text") or v.get("reviewText") or "")
                            if k is not None:
                                review_texts[str(k)] = txt

            _collect(reviews)
            if isinstance(reviews, dict):
                for val in reviews.values():
                    _collect(val)
        except Exception:  # pragma: no cover - corrupted review file
            pass


    # ------------------------------------------------------------------
    # 1) Parse the popularity file to obtain the set ``H`` of candidate items
    asin_pattern = re.compile(r"Item:\s*([A-Z0-9]+)")
    id_pattern = re.compile(r"ID:\s*(\d+)")
    high_pop_ids: List[str] = []
    high_pop_asins: List[str] = []
    with open(pop_path, "r", encoding="utf-8") as f:
        for line in f:
            asin: Optional[str] = None
            idx_val: Optional[str] = None
            if id_mode != "id":
                m = asin_pattern.search(line)
                if m:
                    asin = m.group(1)
            if id_mode != "asin":
                m = id_pattern.search(line)
                if m:
                    idx_val = m.group(1)
                    if asin is None:
                        asin = id2asin.get(idx_val)
                        if asin is None and id_mode == "id":
                            asin = idx_val
            if asin:
                high_pop_asins.append(asin)
                high_pop_ids.append(idx_val or asin)

    if not high_pop_asins:
        raise ValueError(f"No items parsed from {pop_path!r}")

    # ------------------------------------------------------------------
    # 2) Compute fused embeddings ``E(.)`` for items in ``H`` and targets ``T``
    if item_loader is None:
        # simple stub used in the tests – returns empty inputs
        def item_loader(_item_id: int) -> Dict[str, Any]:  # type: ignore
            return {
                "image_input": np.zeros(1, dtype=float),
                "text_input": np.zeros(1, dtype=float),
                "text": "",
                "category_ids": np.zeros(1, dtype=int),
            }
        
    feat_dir = None
    if feat_root and feat_backbone:
        feat_dir = os.path.join(feat_root, feat_backbone, dataset)

    img_feat_dim = 0
    
    
    def get_image_vec(
        asin: str, item_id: Optional[str] = None
    ) -> tuple[Optional[np.ndarray], bool, bool]:
        """Return image array for ``asin``.

        Returns a tuple ``(array, synthetic, from_path)`` where ``array`` is a
        1‑D ``float32`` vector. ``synthetic`` is ``True`` when a zero vector was
        used as a fallback and ``from_path`` is kept for backwards compatibility
        but always ``False`` in this simplified implementation.
        """

        nonlocal img_feat_dim
        if feat_dir is not None:
            candidates = [asin, asin.upper(), asin.lower()]
            if item_id is not None:
                item_str = str(item_id)
                candidates.extend([item_str, item_str.upper(), item_str.lower()])
            for cand in candidates:
                path = os.path.join(feat_dir, f"{cand}.npy")
                if os.path.isfile(path):
                    try:
                        arr = np.load(path).astype("float32").reshape(-1)
                        img_feat_dim = arr.shape[-1]
                        return arr, False, False
                    except Exception as e:
                        logging.warning("[ImageFeat] failed to load %s: %s", path, e)

        dim = img_feat_dim or int(img_dim)
        return np.zeros(dim, dtype=np.float32), True, False

    fused_high: List[np.ndarray] = []
    texts: List[str] = []
    raw_items: Dict[str, Dict[str, Any]] = {}
    missing_text_ids: List[int] = []
    synth_img_count = 0
    high_img_tokens: List[np.ndarray] = []
    high_txt_tokens: List[np.ndarray] = []
    for idx, item_id in enumerate(high_pop_ids):
        item = item_loader(item_id) or {}
        asin = high_pop_asins[idx]
        img_in, synth, _ = get_image_vec(asin, item_id)
        if synth:
            if not allow_missing_image:
                raise KeyError(f"image not found for item {item_id}")
            synth_img_count += 1
        txt_in = np.asarray(item.get("text_input", np.zeros(1, dtype=float)), dtype=float)
        text = str(item.get("text", "") or "")
        if not text:
            text = (
                meta_titles.get(asin, "")
                or review_texts.get(asin, "")
                or review_texts.get(str(item_id), "")
            )
        if not text:
            if allow_missing_text:
                missing_sources = []
                if not meta_titles.get(asin):
                    missing_sources.append("meta.json")
                if not (review_texts.get(asin) or review_texts.get(str(item_id))):
                    missing_sources.append("review_splits.pkl")
                logging.warning(
                    "No text found for item %s; missing sources: %s",
                    item_id,
                    ", ".join(missing_sources) or "none",
                )
                try:
                    missing_text_ids.append(int(str(item_id)))
                except Exception:
                    pass
            else:
                raise KeyError(f"text not found for item {item_id}")

        raw_items[asin] = {"image": _to_jsonable(img_in), "text": text}

        if model is not None:
            outputs = forward_inference(model, img_in, txt_in)
            img_emb = outputs.get("image_embedding")
            txt_emb = outputs.get("text_embedding")
            if torch is not None:
                img_emb = img_emb.detach().cpu().numpy() if isinstance(img_emb, torch.Tensor) else np.asarray(img_emb)
                txt_emb = txt_emb.detach().cpu().numpy() if isinstance(txt_emb, torch.Tensor) else np.asarray(txt_emb)
            else:
                img_emb = np.asarray(img_emb)
                txt_emb = np.asarray(txt_emb)
            if img_emb.ndim == 3:
                img_emb = img_emb[0]
            if txt_emb.ndim == 3:
                txt_emb = txt_emb[0]
            d_model = img_emb.shape[1] if img_emb.ndim == 2 else txt_emb.shape[1]
            assert img_emb.ndim == 2 and img_emb.shape[1] == d_model and img_emb.shape[0] >= 1, (
                f"img_emb shape {img_emb.shape} invalid"
            )
            e_img_item = np.mean(img_emb, axis=0)
            e_txt_item = np.mean(txt_emb, axis=0)
            fused = w_img * e_img_item + w_txt * e_txt_item
            fused_high.append(fused)
            high_img_tokens.append(img_emb)
            high_txt_tokens.append(txt_emb)
        else:  # pragma: no cover - used only in CLI fallbacks
            img_emb = img_in.reshape(1, -1)
            txt_emb = txt_in.reshape(1, -1)
            fused = w_img * img_emb.mean(axis=0) + w_txt * txt_emb.mean(axis=0)
            fused_high.append(fused)
            high_img_tokens.append(img_emb)
            high_txt_tokens.append(txt_emb)
        texts.append(text)

    fused_tgts: List[np.ndarray] = []
    target_asins: List[str] = []
    target_img_tokens: List[np.ndarray] = []
    target_txt_tokens: List[np.ndarray] = []
    targets = list(targets)
    for item_id in targets:
        asin = id2asin.get(str(item_id), str(item_id))
        target_asins.append(asin)
        item = item_loader(item_id) or {}
        img_in, synth, _ = get_image_vec(asin, item_id)
        if synth:
            if not allow_missing_image:
                raise KeyError(f"image not found for item {item_id}")
            synth_img_count += 1
        txt_in = np.asarray(item.get("text_input", np.zeros(1, dtype=float)), dtype=float)
        text = str(item.get("text", "") or "")
        if not text:
            text = (
                meta_titles.get(asin, "")
                or review_texts.get(asin, "")
                or review_texts.get(str(item_id), "")
            )
        if not text:
            if allow_missing_text:
                missing_sources = []
                if not meta_titles.get(asin):
                    missing_sources.append("meta.json")
                if not (review_texts.get(asin) or review_texts.get(str(item_id))):
                    missing_sources.append("review_splits.pkl")
                logging.warning(
                    "No text found for item %s; missing sources: %s",
                    item_id,
                    ", ".join(missing_sources) or "none",
                )
                try:
                    missing_text_ids.append(int(str(item_id)))
                except Exception:
                    pass
            else:
                raise KeyError(f"text not found for item {item_id}")

        raw_items[asin] = {"image": _to_jsonable(img_in), "text": text}

        if model is not None:
            outputs = forward_inference(model, img_in, txt_in)
            img_emb = outputs.get("image_embedding")
            txt_emb = outputs.get("text_embedding")
            if torch is not None:
                img_emb = img_emb.detach().cpu().numpy() if isinstance(img_emb, torch.Tensor) else np.asarray(img_emb)
                txt_emb = txt_emb.detach().cpu().numpy() if isinstance(txt_emb, torch.Tensor) else np.asarray(txt_emb)
            else:
                img_emb = np.asarray(img_emb)
                txt_emb = np.asarray(txt_emb)
            if img_emb.ndim == 3:
                img_emb = img_emb[0]
            if txt_emb.ndim == 3:
                txt_emb = txt_emb[0]
            d_model = img_emb.shape[1] if img_emb.ndim == 2 else txt_emb.shape[1]
            assert img_emb.ndim == 2 and img_emb.shape[1] == d_model and img_emb.shape[0] >= 1, (
                f"img_emb shape {img_emb.shape} invalid"
            )
            e_img_item = np.mean(img_emb, axis=0)
            e_txt_item = np.mean(txt_emb, axis=0)
            fused = w_img * e_img_item + w_txt * e_txt_item
            fused_tgts.append(fused)
            target_img_tokens.append(img_emb)
            target_txt_tokens.append(txt_emb)
        else:  # pragma: no cover - used only in CLI fallbacks
            img_emb = img_in.reshape(1, -1)
            txt_emb = txt_in.reshape(1, -1)
            fused = w_img * img_emb.mean(axis=0) + w_txt * txt_emb.mean(axis=0)
            fused_tgts.append(fused)
            target_img_tokens.append(img_emb)
            target_txt_tokens.append(txt_emb)

    high_matrix = np.vstack(fused_high)
    tgt_matrix = np.vstack(fused_tgts) if fused_tgts else np.zeros((0, high_matrix.shape[1]))

    # Optional dimensionality reduction
    if pca_dim and PCA is not None and high_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0)
        combined = np.vstack([high_matrix, tgt_matrix]) if tgt_matrix.size else high_matrix
        combined = pca.fit_transform(combined)
        high_matrix = combined[: len(high_pop_ids)]
        if tgt_matrix.size:
            tgt_matrix = combined[len(high_pop_ids) :]
    # ------------------------------------------------------------------
    # 3) Optional KMeans clustering
    clusters = None
    labels = np.zeros(len(high_pop_ids), dtype=int)
    tgt_labels = np.zeros(len(targets), dtype=int)
    if kmeans_k and KMeans is not None and len(high_pop_ids) >= kmeans_k:
        try:
            km = KMeans(n_clusters=kmeans_k, n_init=10, random_state=0)
            labels = km.fit_predict(high_matrix)
            clusters = {"centroids": km.cluster_centers_.tolist()}
            if tgt_matrix.size:
                tgt_labels = km.predict(tgt_matrix)
        except Exception:
            clusters = None
            labels = np.zeros(len(high_pop_ids), dtype=int)
            tgt_labels = np.zeros(len(targets), dtype=int)

    # ------------------------------------------------------------------
    # 4) For each target compute nearest neighbours inside its cluster
    pool: Dict[str, Dict[str, Any]] = {}
    keywords: Dict[str, List[str]] = {}
    for idx, item_id in enumerate(targets):
        if len(high_pop_ids) == 0:
            cluster_members = np.array([], dtype=int)
        else:
            if kmeans_k and KMeans is not None and clusters is not None:
                cluster_members = np.where(labels == tgt_labels[idx])[0]
            else:
                cluster_members = np.arange(len(high_pop_ids))

        if cluster_members.size:
            emb = tgt_matrix[idx]
            others = high_matrix[cluster_members]
            norms = np.linalg.norm(others, axis=1) * (np.linalg.norm(emb) + 1e-12)
            sims = (others @ emb) / np.where(norms == 0, 1e-12, norms)
            order = np.argsort(-sims)[:c_size]
            neigh_indices = [cluster_members[i] for i in order]
        else:
            neigh_indices = []

        comp_ids = [high_pop_asins[i] for i in neigh_indices]
        if neigh_indices:
            img_vecs = [np.mean(high_img_tokens[i], axis=0) for i in neigh_indices]
            txt_vecs = [np.mean(high_txt_tokens[i], axis=0) for i in neigh_indices]
            e_img = np.mean(np.stack(img_vecs, axis=0), axis=0)
            e_txt = np.mean(np.stack(txt_vecs, axis=0), axis=0)
            anchor_vec = w_img * e_img + w_txt * e_txt
            d_model = anchor_vec.shape[0]
            assert anchor_vec.ndim == 1 and anchor_vec.shape[0] == d_model, (
                f"anchor_dim={anchor_vec.shape}, expect ({d_model},)"
            )
        else:
            anchor_vec = (
                tgt_matrix[idx] if tgt_matrix.size else np.zeros(high_matrix.shape[1])
            )

        logging.info(
            "anchor_dim=%d head=%s tail=%s",
            anchor_vec.shape[0],
            np.round(anchor_vec[:3], 4).tolist(),
            np.round(anchor_vec[-3:], 4).tolist(),
        )

        target_asin = target_asins[idx]
        pool[target_asin] = {
            "competitors": comp_ids,
            "anchor": _to_jsonable(anchor_vec),
        }

        neigh_texts = [texts[i] for i in neigh_indices]
        if neigh_texts and TfidfVectorizer is not None:
            try:
                vect = TfidfVectorizer(max_features=keyword_top)
                tfidf = vect.fit_transform(neigh_texts)
                scores = np.asarray(tfidf.sum(axis=0)).ravel()
                order = np.argsort(-scores)[:keyword_top]
                top_terms = vect.get_feature_names_out()[order]
                keywords[target_asin] = top_terms.tolist()
            except Exception:
                counts = Counter(" ".join(neigh_texts).split())
                keywords[target_asin] = [w for w, _ in counts.most_common(keyword_top)]
        else:
            counts = Counter(" ".join(neigh_texts).split())
            keywords[target_asin] = [w for w, _ in counts.most_common(keyword_top)]


    # ------------------------------------------------------------------
    # 4a) Fill in synthetic keywords for targets lacking mined ones
    synthetic_flags: Dict[str, bool] = {}
    for asin in target_asins:
        kw_list = keywords.get(asin, [])
        if not kw_list:
            text_src = raw_items.get(asin, {}).get("text", "")
            tokens = re.findall(r"\w+", text_src)
            if not tokens:
                tokens = ["item", "product"]
            kw_list = tokens[:keyword_top]
            synthetic_flags[asin] = True
        if len(kw_list) < min_keywords:
            needed = min_keywords - len(kw_list)
            kw_list.extend(["kw" + str(i) for i in range(needed)])
            synthetic_flags[asin] = True
        keywords[asin] = kw_list

    for asin in target_asins:
        pool.setdefault(asin, {})["keywords"] = keywords.get(asin, [])
        if synthetic_flags.get(asin, False):
            pool[asin]["synthetic"] = True


    # ------------------------------------------------------------------
    # 4b) Gather per-item metadata (title and image features)
    items_meta: Dict[str, Dict[str, Any]] = {}

    for asin in high_pop_asins + target_asins:
        idx_str = asin2id.get(asin.upper(), asin)
        title = meta_titles.get(asin) or review_texts.get(asin) or review_texts.get(idx_str) or ""

        vec, synth, _ = get_image_vec(asin, idx_str)
        if vec is None:
            continue
        if synth:
            npy_candidates = [
                os.path.join(dataset_dir, f"{asin}.npy"),
                os.path.join(dataset_dir, f"{idx_str}.npy"),
                os.path.join(dataset_dir, "image_features", f"{asin}.npy"),
            ]
            for path in npy_candidates:
                if os.path.exists(path):
                    try:
                        arr = np.load(path)
                        vec = np.asarray(arr, dtype=float).ravel()
                        synth = False
                        if img_feat_dim == 0:
                            img_feat_dim = len(vec)
                        break
                    except Exception:  # pragma: no cover - invalid npy
                        vec = None
        if vec is None:
            continue
        items_meta[asin] = {"title": title, "image_feat": vec.tolist()}

    # ------------------------------------------------------------------
    total_items = len(high_pop_ids) + len(target_asins)
    logging.info(
        "Image features synthesised for %d/%d items; missing text for %d items",
        synth_img_count,
        total_items,
        len(missing_text_ids),
    )

    # ------------------------------------------------------------------
    # 5) Persist to disk
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "caches")
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"competition_pool_{dataset}.json")

    data = {
        "dataset": dataset,
        "high_pop": high_pop_asins,
        "targets": target_asins,
        "clusters": clusters,
        "pool": pool,
        "raw_items": raw_items,
        "missing_text_ids": sorted(set(missing_text_ids)),
        "items": items_meta,
        "params": {
            "w_img": w_img,
            "w_txt": w_txt,
            "pca_dim": pca_dim,
            "kmeans_k": kmeans_k,
            "c_size": c_size,
            "keyword_top": keyword_top,
        },
    }

    obj_jsonable = _to_jsonable(data)
    try:
        json.dumps(obj_jsonable)
    except TypeError as e:
        raise AssertionError(f"not JSON-safe: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj_jsonable, f, ensure_ascii=False, indent=2)

    return data


# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line options for the module CLI.

    Two modes are supported:

    * When ``--input-pool`` is provided the full poisoning pipeline is
      executed (legacy behaviour).
    * Otherwise ``--dataset`` and ``--pop-path`` are expected and the
      competition pool mining routine is invoked.
    """

    parser = argparse.ArgumentParser(description="Utilities for DCIP-IEOS pool mining")
    parser.add_argument("--dataset", help="dataset name")
    parser.add_argument("--pop-path", help="path to high popularity items file")
    parser.add_argument("--input-pool", help="raw competition pool JSON for full pipeline")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "caches"),
        help="directory for cached artifacts",
    )
    parser.add_argument("--pca-dim", type=int, default=None, help="optional PCA dimensionality")
    parser.add_argument("--kmeans-k", type=int, default=8, help="number of KMeans clusters")
    parser.add_argument("--no-kmeans", action="store_true", help="disable KMeans clustering")
    parser.add_argument("--c-size", type=int, default=20, help="number of neighbours per target")
    parser.add_argument("--keywords-top", type=int, default=50, help="number of mined keywords")
    parser.add_argument("--min-keywords", type=int, default=5, help="minimum keywords per target")
    parser.add_argument(
        "--id-mode",
        choices=["auto", "asin", "id"],
        default="auto",
        help="Identifier interpretation for pop/target files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    

    if args.input_pool:
        with open(args.input_pool, "r", encoding="utf-8") as f:
            pool = json.load(f)

        # Local import to avoid a circular dependency – ``poison_pipeline`` imports
        # :class:`PoolMiner` from this module.
        from .poison_pipeline import PoisonPipeline

        pipeline = PoisonPipeline(args.output_dir, args.dataset or "unknown")
        pipeline.run(pool)
        return

    if not args.dataset or not args.pop_path:
        raise SystemExit("--dataset and --pop-path are required when mining the competition pool")

    build_competition_pool(
        dataset=args.dataset,
        pop_path=args.pop_path,
        targets=[],
        model=None,  # Model loading is out of scope for this utility
        cache_dir=args.output_dir,
        pca_dim=args.pca_dim,
        kmeans_k=None if args.no_kmeans else args.kmeans_k,
        c_size=args.c_size,
        keyword_top=args.keywords_top,
        id_mode=args.id_mode,
        min_keywords=args.min_keywords,
    )


if __name__ == "__main__":
    main()
