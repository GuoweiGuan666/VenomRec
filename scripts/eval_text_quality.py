#!/usr/bin/env python3
"""ROUGE-based text quality evaluation for poisoned artefacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from notebooks.evaluate.utils import rouge_score
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Unable to import notebooks.evaluate.utils. Ensure the repository root is on PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate text quality (ROUGE) for poisoned artefacts")
    parser.add_argument("--dataset", default="clothing", help="Dataset split name")
    parser.add_argument(
        "--poison-subdir",
        required=True,
        help="Subdirectory under poison_text_pairs/<dataset>/poisoned containing text_pairs.jsonl",
    )
    parser.add_argument(
        "--pairs-root",
        default=str(PROJ_ROOT / "poison_text_pairs"),
        help="Base directory storing text_pairs.jsonl files (default: <repo>/poison_text_pairs)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save aggregated metrics (JSON). Defaults to text_quality_eval.json next to text_pairs.jsonl",
    )
    return parser.parse_args()


def load_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"text_pairs file not found: {path}")
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            original = record.get("original_text", "")
            adversarial = record.get("adversarial_text", "")
            if isinstance(original, str) and isinstance(adversarial, str):
                pairs.append({
                    "target": str(record.get("target", "")),
                    "original": original,
                    "adversarial": adversarial,
                })
    if not pairs:
        raise RuntimeError(f"No valid text pairs found in {path}")
    return pairs


def compute_rouge(pairs: list[dict[str, str]]):
    metrics = []
    for pair in pairs:
        try:
            score = rouge_score([pair["original"]], [pair["adversarial"]])
        except ValueError:
            continue
        metrics.append(score)
    if not metrics:
        raise RuntimeError("Unable to compute ROUGE: all samples were empty.")
    keys = metrics[0].keys()
    aggregated = {k: mean(m[k] for m in metrics) for k in keys}
    return aggregated, len(metrics)


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs_root) / args.dataset / "poisoned" / args.poison_subdir / "text_pairs.jsonl"
    pairs = load_pairs(pairs_path)
    rouge_avg, sample_count = compute_rouge(pairs)

    print("=== Text Quality (ROUGE) ===")
    print(f"Samples: {sample_count}")
    for key, value in sorted(rouge_avg.items()):
        print(f"{key}: {value:.4f}")

    output_path = Path(args.output) if args.output else pairs_path.with_name("text_quality_eval.json")
    payload = {
        "pairs_path": str(pairs_path),
        "samples": sample_count,
        "rouge": rouge_avg,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
