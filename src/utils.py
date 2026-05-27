import os
import re
import numpy as np
import torch
import torch.distributed as dist
import collections
import logging
from typing import Optional, List


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


_ID_PATTERN = re.compile(r"(\d+)")


def parse_item_id(raw) -> Optional[int]:
    """Extract a numeric item id from decoder output.

    Accepts strings such as "123", "Item: 123", or "123,". Returns ``None``
    when no digits are found so callers can safely skip unparsable entries.
    """

    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match = _ID_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def load_target_items_from_path(path: str) -> List[int]:
    """Load target item ids from a file path.

    Expected line formats include:
    - ``Item: <asin> (ID: <n>)`` (we extract the numeric ID)
    - ``<n>`` (plain numeric id)
    """

    items: List[int] = []
    if not path or not os.path.exists(path):
        return items

    id_pattern = re.compile(r"ID:\s*(\d+)")
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            text = line.strip()
            if not text:
                continue
            match = id_pattern.search(text)
            if match:
                try:
                    items.append(int(match.group(1)))
                except ValueError:
                    continue
                continue
            if text.isdigit():
                try:
                    items.append(int(text))
                except ValueError:
                    continue
    return items


def load_target_items(split: str, base_dir: Optional[str] = None) -> List[int]:
    """Load target item ids for a dataset split.

    The analysis files contain lines of the form ``Item: <asin> (ID: <n>)``.
    Some datasets may only contain the numeric id. We return the numeric ids
    because direct-task targets are stored as indices.
    """

    if not split:
        return []

    base_dir = base_dir or os.getcwd()
    path = os.path.join(
        base_dir,
        "analysis",
        "results",
        split,
        f"low_pop_items_{split}_lowcount_1.txt",
    )
    return load_target_items_from_path(path)
