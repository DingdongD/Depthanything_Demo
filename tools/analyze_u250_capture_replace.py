#!/usr/bin/env python3
"""Rank one-boundary FP32 replacement effects on final-depth accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_da2k_board_outputs import (
    aggregate, continuous_metrics, load_depth, pair_metrics,
)


LOWER_IS_BETTER = ("rel_l2", "mae", "affine_rel_l2")
HIGHER_IS_BETTER = (
    "cosine", "pearson", "board_pair_accuracy", "board_fp32_pair_agreement",
)


def evaluate(root: Path, records: list[tuple[str, dict]], dimensions: dict,
             annotations: dict, teacher_root: Path) -> tuple[dict, dict[str, dict]]:
    samples = []
    by_id = {}
    for split, record in records:
        sample_id = record["sample_id"]
        output = root / split / f"{sample_id}.npz"
        if not output.is_file():
            raise FileNotFoundError(output)
        board = load_depth(output)
        teacher = load_depth(teacher_root / split / f"{sample_id}.npy")
        dims = dimensions[sample_id]
        result = {
            "sample_id": sample_id, "split": split, "scene": record["scene"],
            **continuous_metrics(board, teacher),
            **pair_metrics(board, teacher, annotations[record["path"]],
                           dims["width"], dims["height"]),
        }
        count = result["pairs"]
        result["board_pair_accuracy"] = result["board_pair_correct"] / count
        result["fp32_pair_accuracy"] = result["fp32_pair_correct"] / count
        result["board_fp32_pair_agreement"] = result["board_fp32_pair_agree"] / count
        samples.append(result)
        by_id[sample_id] = result
    return aggregate(samples), by_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dimensions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    dimensions = json.loads(args.dimensions.read_text())
    annotations = json.loads(args.annotations.read_text())
    records = [(split, record) for split, values in manifest["splits"].items()
               for record in values]
    baseline, baseline_samples = evaluate(
        args.baseline_root, records, dimensions, annotations, args.teacher_root
    )
    interventions = []
    for root in sorted(path for path in args.sweep_root.iterdir() if path.is_dir()):
        metrics, samples = evaluate(
            root, records, dimensions, annotations, args.teacher_root
        )
        recovery = {
            key: baseline[key] - metrics[key] for key in LOWER_IS_BETTER
        }
        recovery.update({
            key: metrics[key] - baseline[key]
            for key in HIGHER_IS_BETTER if key in baseline
        })
        per_sample_rel = np.asarray([
            baseline_samples[key]["rel_l2"] - samples[key]["rel_l2"]
            for key in sorted(samples)
        ], dtype=np.float64)
        interventions.append({
            "target": root.name, "metrics": metrics, "recovery": recovery,
            "rel_l2_recovery_std": float(per_sample_rel.std()),
            "rel_l2_samples_improved": int(np.count_nonzero(per_sample_rel > 0)),
            "rel_l2_samples_regressed": int(np.count_nonzero(per_sample_rel < 0)),
        })
    by_target = {item["target"]: item for item in interventions}
    for prefix, count in (("encoder_block", 12), ("decoder_conv", 32)):
        for index in range(count):
            item = by_target.get(f"{prefix}_{index:02d}")
            if item is None:
                continue
            previous = by_target.get(f"{prefix}_{index - 1:02d}")
            item["incremental_recovery_vs_previous_boundary"] = (
                None if previous is None else {
                    key: item["recovery"][key] - previous["recovery"][key]
                    for key in item["recovery"]
                }
            )
    interventions.sort(key=lambda item: item["recovery"]["rel_l2"], reverse=True)
    report = {
        "schema": "depthanything-u250-capture-replace-v1",
        "method": (
            "one intervention per run: replace one live r80 boundary with the "
            "corresponding full-FP32 capture, then execute the remaining r80 graph"
        ),
        "interpretation": (
            "Recovery is causal for the selected boundary but includes accumulated "
            "upstream error removed by that capture; layer recoveries are not additive."
        ),
        "incremental_interpretation": (
            "For the sequential encoder, adjacent-boundary recovery differences isolate "
            "the intervening block on a reference input as observed through the remaining "
            "r80 suffix. Decoder differences are descriptive only because its graph has "
            "multi-scale residual branches."
        ),
        "samples": len(records), "baseline": baseline,
        "interventions": interventions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "baseline": baseline,
        "top": [{"target": item["target"], **item["recovery"]}
                for item in interventions[:10]],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
