#!/usr/bin/env python3
"""Lower the four decoder LayerNorm cores to an existing resident SPU kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def lower_decoder_layernorm(contract: dict, kernel: str) -> dict:
    resident = {item["name"] for item in contract.get("resident_kernels", [])}
    # Current contracts keep resident records in a separate bank manifest, so
    # an absent inline list is valid.  If present, it remains authoritative.
    if resident and kernel not in resident:
        raise ValueError(f"resident kernel is missing: {kernel}")
    lowered = []
    schedule = []
    for group in contract["decoder"]:
        if group["backend"] != "host":
            schedule.append(group)
            continue
        remaining = []
        for node in group["nodes"]:
            if node["op_type"] != "LayerNormalization":
                remaining.append(node)
                continue
            if len(node["inputs"]) < 3 or len(node["outputs"]) != 1:
                raise ValueError(f"{node['name']}: incomplete affine LayerNorm")
            lowered.append({
                "backend": "npu_layernorm", "source_node": node["name"],
                "input_tensor": node["inputs"][0],
                "scale_tensor": node["inputs"][1],
                "bias_tensor": node["inputs"][2],
                "output_tensor": node["outputs"][0], "kernel": kernel,
                "core_precision": "BF16", "affine": "host_fp32",
            })
        if remaining:
            schedule.append({**group, "nodes": remaining})
    if len(lowered) != 4:
        raise ValueError(f"expected four decoder LayerNorms, found {len(lowered)}")
    contract = {**contract, "decoder": lowered + schedule}
    totals = dict(contract["execution_totals"])
    totals["decoder_npu_calls"] = int(totals["decoder_npu_calls"]) + 4
    totals["npu_calls_per_inference"] = int(totals["npu_calls_per_inference"]) + 4
    contract["execution_totals"] = totals
    contract["decoder_capture_policy"] = {
        "layers": [2, 5, 8, 11], "storage": "resident_BF16_NDWC",
        "layernorm_submission": "one dependent-free C++ chain",
        "host_affine": True,
    }
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--kernel", default="tail_norm1_from_tokens")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads(args.bank_manifest.read_text())
    names = {item["name"] for item in bank["cases"]}
    if args.kernel not in names:
        raise ValueError(f"bank manifest is missing {args.kernel}")
    lowered = lower_decoder_layernorm(json.loads(args.contract.read_text()), args.kernel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lowered, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "decoder_npu_calls":
        lowered["execution_totals"]["decoder_npu_calls"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
