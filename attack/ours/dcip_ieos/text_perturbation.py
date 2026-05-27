"""Text perturbation helpers with synonym replacement and embedding tweaks."""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "jeans": ["denims", "trousers"],
    "fit": ["regular-fit", "slim-fit"],
    "cotton": ["soft-cotton", "organic-cotton"],
    "mens": ["menswear", "male"],
    "speidel": ["speidel-brand", "speidel-co"],
    "black": ["jet-black", "charcoal"],
    "sport": ["athletic", "sporty"],
    "metal": ["steel", "metallic"],
    "watch": ["timepiece", "wristwatch"],
    "band": ["strap", "loop"],
    "fits": ["accommodates", "suits"],
    "casio": ["casio-brand", "casio-co"],
    "long": ["extended", "elongated"],
}

STOPWORDS = {"the", "and", "of", "a", "to", "for", "in", "on"}
SUBWORD_MARKERS = {"▁", "##"}


def normalize_token(tok: str) -> str:
    return tok.lower().strip("#▁'\".,:;!?()[]{}")


def _strip_markers(tok: str) -> str:
    t = tok
    for m in SUBWORD_MARKERS:
        t = t.replace(m, "")
    return t


def is_replaceable_token(tok: str) -> bool:
    if tok is None:
        return False
    t = normalize_token(_strip_markers(str(tok)))
    if len(t) <= 1:
        return False
    if t.isdigit() or t in STOPWORDS:
        return False
    return True


def _token_to_str(tokenizer: Any, tok_id: Any) -> str:
    if tokenizer is None:
        return str(tok_id)
    try:
        if hasattr(tokenizer, "decode"):
            return str(tokenizer.decode([tok_id]))
    except Exception:
        pass
    return str(tok_id)


def _ensure_tokens(tokens: Optional[Sequence[Any]], tokenizer: Any, text: Optional[str]) -> List[Any]:
    if tokens is not None:
        return list(tokens)
    if tokenizer is None or text is None:
        raise AssertionError("need tokenizer + text when tokens not supplied")
    encoded = tokenizer.encode(text)
    if not isinstance(encoded, list):
        raise AssertionError("tokenizer.encode must return a list")
    return list(encoded)


def _embedding_delta(
    embeddings: Optional[Sequence[Sequence[float]]],
    mask: Sequence[bool],
    embed_eps: float,
) -> Optional[List[List[float]]]:
    if embeddings is None or embed_eps <= 0:
        return None
    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(mask):
        logging.warning("[text-perturb] embedding shape mismatch; skipping embed tweak")
        return None
    noise = np.zeros_like(arr)
    active_idx = np.where(np.asarray(mask, dtype=bool))[0]
    if active_idx.size == 0:
        return None
    noise[active_idx] = np.random.uniform(-embed_eps, embed_eps, size=(active_idx.size, arr.shape[1]))
    return noise.tolist()


class TextPerturber:
    """Guided token replacement with optional embedding perturbation."""

    def __init__(self, ratio: float = 0.3) -> None:
        self.ratio = ratio
        self.default_keywords = {"good": "great", "bad": "poor", "product": "item"}

    def perturb(
        self,
        tokens: Optional[Sequence[Any]] = None,
        text: Optional[str] = None,
        mask: Optional[Sequence[bool]] = None,
        tokenizer: Optional[Any] = None,
        *,
        txt_ratio_max: Optional[float] = None,
        keywords: Optional[Iterable[str] | Dict[str, str]] = None,
        min_txt_replacements: int = 1,
        embed_eps: float = 0.0,
        token_embeddings: Optional[Sequence[Sequence[float]]] = None,
        text_topk: int = 3,
        score_fn: Optional[Any] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        if mask is None:
            raise AssertionError("text mask required")
        raw_tokens = _ensure_tokens(tokens, tokenizer, text)
        mask_list = list(mask)
        if len(raw_tokens) != len(mask_list):
            logging.warning(
                "[text-perturb] mask/token mismatch (mask=%d tokens=%d); aligning",
                len(mask_list),
                len(raw_tokens),
            )
            if len(mask_list) > len(raw_tokens):
                mask_list = mask_list[: len(raw_tokens)]
            else:
                mask_list.extend([False] * (len(raw_tokens) - len(mask_list)))
        mask = mask_list

        kw = keywords if keywords is not None else self.default_keywords
        ratio = txt_ratio_max if txt_ratio_max is not None else self.ratio

        total_masked = sum(1 for m in mask if m)
        allowed = int(math.floor((ratio if ratio is not None else 0.0) * total_masked + 1e-9))
        if allowed < min_txt_replacements and total_masked > 0:
            allowed = min(min_txt_replacements, total_masked)

        replaced = 0
        new_tokens = list(raw_tokens)

        for idx, include in enumerate(mask):
            if not include or replaced >= allowed:
                continue
            base = _token_to_str(tokenizer, raw_tokens[idx])
            if not is_replaceable_token(base):
                continue
            norm = normalize_token(base)
            candidates = list(DOMAIN_SYNONYMS.get(norm, []))
            if kw:
                kwset = {normalize_token(k) for k in kw}
                keyword_based = [c for c in candidates if normalize_token(c) in kwset]
                fallback_kw = [k for k in kw if normalize_token(k) != norm]
                candidates = keyword_based or fallback_kw or candidates
            if not candidates and norm:
                candidates = [f"soft-{norm}"]
            candidates = [c for c in candidates if normalize_token(c) != norm]
            if not candidates:
                continue
            candidates = candidates[: max(int(text_topk), 1)]
            chosen = candidates[0]
            if score_fn is not None:
                best_score = None
                for cand in candidates:
                    try:
                        score = float(score_fn(idx, cand))
                    except Exception:
                        score = float("inf")
                    if best_score is None or score < best_score:
                        best_score, chosen = score, cand
            encoded = tokenizer.encode(chosen) if tokenizer is not None else [chosen]
            if not encoded:
                continue
            if len(encoded) > 1:
                encoded = encoded[:1]
            new_tokens[idx] = encoded[0]
            replaced += 1

        if replaced < min_txt_replacements and allowed >= 1:
            for idx, include in enumerate(mask):
                if not include:
                    continue
                base = _token_to_str(tokenizer, raw_tokens[idx])
                if not is_replaceable_token(base):
                    continue
                norm = normalize_token(base)
                cands = [c for c in DOMAIN_SYNONYMS.get(norm, []) if normalize_token(c) != norm]
                if not cands and kw:
                    cands = [k for k in kw if normalize_token(k) != norm]
                if not cands and norm:
                    cands = [f"soft-{norm}"]
                if not cands:
                    continue
                encoded = tokenizer.encode(cands[0]) if tokenizer is not None else [cands[0]]
                if encoded:
                    if len(encoded) > 1:
                        encoded = encoded[:1]
                    new_tokens[idx] = encoded[0]
                    replaced += 1
                break

        replace_ratio = replaced / max(total_masked, 1)
        logging.info("[text-perturb] replaced %d of %d masked tokens (ratio %.2f)", replaced, total_masked, replace_ratio)

        embedding_delta = _embedding_delta(token_embeddings, mask, embed_eps)
        metadata: Dict[str, Any] = {"replaced": replaced, "total": total_masked, "tokens": list(new_tokens)}
        if embedding_delta is not None:
            metadata["embedding_delta"] = embedding_delta

        if tokenizer is not None and hasattr(tokenizer, "decode"):
            new_text = tokenizer.decode(new_tokens)
        else:
            new_text = " ".join(str(t) for t in new_tokens)

        return new_text, replace_ratio, metadata


__all__ = [
    "DOMAIN_SYNONYMS",
    "guided_text_paraphrase",
    "TextPerturber",
    "normalize_token",
    "is_replaceable_token",
]


def guided_text_paraphrase(
    tokens: Sequence[str],
    mask: Sequence[bool],
    keywords: Iterable[str] | Dict[str, str],
    ratio: float,
    synonym_table: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[List[str], int, int]:
    """Backward compatible wrapper returning token list + counts."""

    if len(tokens) != len(mask):
        logging.warning(
            "[text-perturb] guided paraphrase mask/token mismatch (mask=%d tokens=%d); aligning",
            len(mask),
            len(tokens),
        )
        mask = list(mask)
        if len(mask) > len(tokens):
            mask = mask[: len(tokens)]
        else:
            mask.extend([False] * (len(tokens) - len(mask)))
    kw = keywords
    if synonym_table is not None and not isinstance(kw, dict):
        kw = {k: synonym_table.get(k, [k])[0] for k in keywords}
    perturb = TextPerturber(ratio=ratio)
    new_text, _, meta = perturb.perturb(
        tokens=tokens,
        mask=mask,
        tokenizer=None,
        txt_ratio_max=ratio,
        keywords=kw,
        min_txt_replacements=0,
    )
    result_tokens = meta.get("tokens")
    if result_tokens is None:
        result_tokens = new_text.split()
    return list(result_tokens), int(meta.get("replaced", 0)), int(meta.get("total", len(tokens)))
