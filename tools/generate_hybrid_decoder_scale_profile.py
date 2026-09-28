#!/usr/bin/env python3
"""Replace decoder Conv A8 scales with max-abs statistics from a hybrid run."""

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
    summary_text = args.hybrid_summary.read_text().strip()
    if summary_text.startswith("HYBRID_SUMMARY="):
        summary_text = summary_text.removeprefix("HYBRID_SUMMARY=")
    summary = json.loads(summary_text)
    statistics = summary["decoder_calibration"]
    output = copy.deepcopy(profile)
    changed = []
    for operator in output["operators"]:
        if operator.get("kind") != "decoder_conv":
            continue
        node = operator["node"]
        if node not in statistics:
            raise KeyError(f"missing hybrid statistics for {node}")
        old_scale = float(operator["a8_scale"])
        new_scale = float(statistics[node]["max_abs"]) / 127.0
        operator["a8_scale"] = new_scale
        changed.append({"node": node, "old_scale": old_scale,
                        "new_scale": new_scale})
    if len(changed) != 32:
        raise ValueError(f"expected 32 decoder Conv scales, got {len(changed)}")
    for node, values in statistics.items():
        if node in output["tensors"]:
            output["tensors"][node].update(values)
            output["tensors"][node]["a8_scale"] = float(values["max_abs"]) / 127.0
    output["scale_policy"] = (
        "decoder Conv activations: hybrid U250 symmetric signed INT8 max_abs/127; "
        "non-decoder entries unchanged from base profile"
    )
    output["hybrid_decoder_calibration_source"] = str(args.hybrid_summary.resolve())
    output["hybrid_decoder_calibration_changes"] = changed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "operators_changed": len(changed)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
