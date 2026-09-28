#!/usr/bin/env python3
"""Compare baseline and intervention depth outputs on identical samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_da2k_board_outputs import continuous_metrics, load_depth


METRICS = ("rel_l2", "mae", "cosine", "pearson", "affine_rel_l2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = []
    for relative_text in args.samples:
        relative = Path(relative_text)
        baseline = load_depth(args.baseline_root / relative.with_suffix(".npz"))
        candidate = load_depth(args.candidate_root / relative.with_suffix(".npz"))
        reference = load_depth(args.reference_root / relative.with_suffix(".npy"))
        if baseline.shape != candidate.shape or baseline.shape != reference.shape:
            raise ValueError(f"shape mismatch for {relative}")
        baseline_metrics = continuous_metrics(baseline, reference)
        candidate_metrics = continuous_metrics(candidate, reference)
        samples.append({
            "sample": str(relative),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_baseline": {
                key: candidate_metrics[key] - baseline_metrics[key]
                for key in METRICS
            },
        })
    aggregate = {}
    for variant in ("baseline", "candidate"):
        aggregate[variant] = {
            key: float(np.mean([sample[variant][key] for sample in samples]))
            for key in METRICS
        }
    aggregate["delta_candidate_minus_baseline"] = {
        key: aggregate["candidate"][key] - aggregate["baseline"][key]
        for key in METRICS
    }
    result = {
        "schema_version": 1,
        "samples": samples,
        "aggregate": aggregate,
        "rel_l2_samples_improved": sum(
            sample["candidate"]["rel_l2"] < sample["baseline"]["rel_l2"]
            for sample in samples
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
