#!/usr/bin/env python3
"""Replace encoder A8 input scales with statistics from a hybrid U250 run."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--hybrid-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.base_profile.read_text())
    summary = json.loads(args.hybrid_summary.read_text())
    statistics = summary["hybrid_calibration"]
    output = copy.deepcopy(profile)
    changed = []
    for operator in output["operators"]:
        if operator.get("kind") != "encoder_linear":
            continue
        activation = operator["activation"]
        if activation not in statistics:
            raise KeyError(f"missing hybrid statistics for {activation}")
        old_scale = float(operator["a8_scale"])
        new_scale = float(statistics[activation]["abs_p9999"]) / 127.0
        operator["a8_scale"] = new_scale
        changed.append({
            "node": operator["node"], "activation": activation,
            "old_scale": old_scale, "new_scale": new_scale,
        })
    for activation, values in statistics.items():
        if activation not in output["tensors"]:
            continue
        output["tensors"][activation].update(values)
        output["tensors"][activation]["a8_scale"] = float(values["abs_p9999"]) / 127.0
    output["scale_policy"] = (
        "encoder linear activations: hybrid U250 abs_p9999/127; "
        "non-encoder entries unchanged from base profile"
    )
    output["hybrid_calibration_source"] = str(args.hybrid_summary.resolve())
    output["hybrid_calibration_changes"] = changed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "operators_changed": len(changed)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
