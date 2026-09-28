#!/usr/bin/env python3
"""Select per-head Q/K/V scales only when both calibration splits improve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.calibration.read_text())
    if float(report.get("av_output_gain", 0.0)) != 1.0:
        raise ValueError("calibration violates unit-amplitude AV policy")
    selected: dict[str, dict[str, float]] = {}
    decisions = []
    for item in report["heads"]:
        baseline = item["baseline"]
        candidate = item["selected"]
        gains = {
            split: (
                float(baseline[split]["relative_l2"])
                - float(candidate[split]["relative_l2"])
            ) / float(baseline[split]["relative_l2"])
            for split in ("training", "validation")
        }
        accepted = all(
            value >= args.min_relative_improvement for value in gains.values()
        )
        if accepted:
            selected[str(item["head"])] = {
                name: float(item["selected_scales_bf16"][name])
                for name in ("q", "k", "v")
            }
        decisions.append({
            "head": int(item["head"]),
            "accepted": accepted,
            "training_relative_improvement": gains["training"],
            "validation_relative_improvement": gains["validation"],
        })
    result = {
        "schema": "depthanything-u250-qkv-scale-selection-v1",
        "layer": int(report["layer"]),
        "av_output_gain": 1.0,
        "heads": selected,
        "selection_gate": {
            "minimum_relative_improvement_on_both_splits":
                args.min_relative_improvement,
            "decisions": decisions,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
