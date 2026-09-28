#!/usr/bin/env python3
"""Apply a train/validation gate to a dual-range calibration report."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.calibration.read_text())
    contract = json.loads(args.contract.read_text())
    layer = int(report["layer"])
    attention = contract["encoder"][layer]["attention"]
    probability = attention.get("dual_range_probability")
    if probability is not None:
        current = probability["heads"]
    else:
        current = {
            str(index): {
                "threshold": head["scales_bf16"]["probability"]["threshold"],
                "residual_step": head["scales_bf16"]["probability"]["residual"],
            }
            for index, head in enumerate(attention["heads"])
        }
    result = copy.deepcopy(report)
    decisions = []
    for head in result["heads"]:
        index = str(head["head"])
        old = current[index]
        baseline = min(
            head["candidates"],
            key=lambda item: abs(float(item["threshold"]) - float(old["threshold"]))
            + abs(float(item["residual_step"]) - float(old["residual_step"])),
        )
        candidate = head["selected"]
        old_train = baseline["training"]["raw_attention"]["relative_l2"]
        old_validation = baseline["validation"]["raw_attention"]["relative_l2"]
        new_train = candidate["training"]["raw_attention"]["relative_l2"]
        new_validation = candidate["validation"]["raw_attention"]["relative_l2"]
        train_gain = (old_train - new_train) / old_train
        validation_gain = (old_validation - new_validation) / old_validation
        accepted = (
            train_gain >= args.min_relative_improvement
            and validation_gain >= args.min_relative_improvement
        )
        if not accepted:
            head["selected"] = baseline
        decisions.append({
            "head": int(index),
            "accepted": accepted,
            "training_relative_improvement": train_gain,
            "validation_relative_improvement": validation_gain,
            "selected_threshold": head["selected"]["threshold"],
            "selected_residual_step": head["selected"]["residual_step"],
        })
    result["selection_gate"] = {
        "minimum_relative_improvement_on_both_splits":
            args.min_relative_improvement,
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decisions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
