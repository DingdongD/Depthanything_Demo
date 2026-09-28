#!/usr/bin/env python3
"""Compare one teacher-forced block-scale candidate with its board baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class Metric:
    def __init__(self) -> None:
        self.error_sq = 0.0
        self.actual_sq = 0.0
        self.reference_sq = 0.0
        self.dot = 0.0

    def add(self, actual: np.ndarray, reference: np.ndarray) -> None:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if actual.shape != reference.shape:
            raise ValueError(f"shape mismatch: {actual.shape} != {reference.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError("non-finite tensor")
        error = actual - reference
        self.error_sq += float(error @ error)
        self.actual_sq += float(actual @ actual)
        self.reference_sq += float(reference @ reference)
        self.dot += float(actual @ reference)

    def finish(self) -> dict[str, float]:
        return {
            "rel_l2": float(np.sqrt(self.error_sq / max(self.reference_sq, 1e-30))),
            "cosine": float(
                self.dot / np.sqrt(max(self.actual_sq * self.reference_sq, 1e-30))
            ),
        }


def trace_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*.npz")) or sorted(root.glob("*/*.npz"))


def summarize(
    paths: list[Path], reference_dir: Path, layer: int, depth_only: bool = False
) -> dict:
    keys = {
        "attention": (f"attention_l{layer:02d}", f"encoder_l{layer:02d}_attention"),
        "post": (f"encoder_l{layer:02d}_post", f"encoder_l{layer:02d}_post"),
        "fc2": (f"encoder_l{layer:02d}_fc2", f"encoder_l{layer:02d}_fc2"),
        "block": (f"block_l{layer:02d}", f"block_l{layer:02d}"),
        "depth": ("depth", "depth"),
    }
    if depth_only:
        keys = {"depth": keys["depth"]}
    metrics = {name: Metric() for name in keys}
    for path in paths:
        reference_path = reference_dir / path.name
        if not reference_path.is_file():
            reference_path = reference_dir / path.relative_to(path.parent.parent)
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference missing for {path}")
        with np.load(path, allow_pickle=False) as actual, np.load(
            reference_path, allow_pickle=False
        ) as reference:
            for name, (actual_key, reference_key) in keys.items():
                metrics[name].add(actual[actual_key], reference[reference_key])
    return {name: metric.finish() for name, metric in metrics.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_paths = trace_paths(args.baseline_dir)
    candidate_paths = trace_paths(args.candidate_dir)
    if [path.name for path in baseline_paths] != [path.name for path in candidate_paths]:
        raise ValueError("baseline and candidate sample sets differ")
    if not 0 < args.train_count < len(candidate_paths):
        raise ValueError("train-count must leave at least one validation sample")

    splits = {
        "all": slice(None),
        "train": slice(0, args.train_count),
        "validation": slice(args.train_count, None),
    }
    report = {
        "schema": "depthanything-u250-block-scale-candidate-v1",
        "layer": args.layer,
        "sample_count": len(candidate_paths),
        "samples": [path.stem for path in candidate_paths],
        "baseline": {
            name: summarize(
                baseline_paths[selection], args.reference_dir, args.layer, args.depth_only
            )
            for name, selection in splits.items()
        },
        "candidate": {
            name: summarize(
                candidate_paths[selection], args.reference_dir, args.layer, args.depth_only
            )
            for name, selection in splits.items()
        },
    }
    report["delta_candidate_minus_baseline"] = {
        split: {
            key: {
                metric: report["candidate"][split][key][metric]
                - report["baseline"][split][key][metric]
                for metric in ("rel_l2", "cosine")
            }
            for key in report["baseline"][split]
        }
        for split in splits
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["delta_candidate_minus_baseline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
