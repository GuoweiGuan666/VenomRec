# minimal utility to project char-level mask to token-level mask
from __future__ import annotations

from typing import Iterable, List, Tuple, Sequence


def project_char_mask_to_tokens(text: str, char_mask: Sequence[int | bool], tokenizer) -> Tuple[List, List[int]]:
    """Project a character-level mask to token-level.

    Parameters
    ----------
    text:
        Original text string.
    char_mask:
        Sequence marking which characters are masked. ``len(char_mask)`` must
        match ``len(text)``; otherwise the function falls back to an empty mask.
    tokenizer:
        Tokenizer providing ``encode`` and ``decode`` methods operating on
        subword tokens.

    Returns
    -------
    token_ids, token_mask : List, List[int]
        Token IDs obtained from ``tokenizer.encode(text)`` and a mask aligned to
        those tokens where ``1`` indicates that at least one character within
        the token span was masked.
    """

    if text is None:
        text = ""
    if not char_mask or len(char_mask) != len(text):
        tokens = tokenizer.encode(text)
        return tokens, [0] * len(tokens)

    token_ids = tokenizer.encode(text)
    token_mask = [0] * len(token_ids)

    i = 0  # cursor over characters in ``text``
    for t_idx, tid in enumerate(token_ids):
        piece = tokenizer.decode([tid])
        if not piece:
            continue
        piece_norm = piece.lstrip()
        while i < len(text) and text[i].isspace():
            i += 1
        span_len = len(piece_norm)
        j = min(len(text), i + span_len)
        any_masked = any(char_mask[k] for k in range(i, j)) if span_len > 0 else False
        token_mask[t_idx] = 1 if any_masked else 0
        i = j

    return token_ids, token_mask


__all__ = ["project_char_mask_to_tokens"]