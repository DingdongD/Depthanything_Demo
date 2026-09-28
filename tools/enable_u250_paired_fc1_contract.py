#!/usr/bin/env python3
"""Replace six single-output FC1 programs per encoder layer with three pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYERS = 12
PAIRS_PER_LAYER = 3


def enable_paired_fc1(contract: dict, bank: dict) -> dict:
    resident = {item["name"] for item in bank["cases"]}
    result = {**contract, "encoder": [dict(block) for block in contract["encoder"]]}
    removed = 0
    for layer, block in enumerate(result["encoder"]):
        if int(block["layer"]) != layer:
            raise ValueError("encoder layers are not in canonical order")
        mlp = dict(block["mlp"])
        expected = [f"mlp_fc1_l{layer:02d}_c{chunk:02d}" for chunk in range(6)]
        if mlp["fc1_kernels"] != expected:
            raise ValueError(f"layer {layer}: unexpected FC1 kernel order")
        paired = [
            f"mlp_fc1_pair_l{layer:02d}_p{pair:02d}"
            for pair in range(PAIRS_PER_LAYER)
        ]
        missing = set(paired) - resident
        if missing:
            raise ValueError(f"layer {layer}: resident bank is missing {sorted(missing)}")
        mlp.update({
            "fc1_kernels": paired,
            "fc1_outputs_per_kernel": 2,
            "fc1_dispatch_policy": "two-output shared-input",
            "fc1_original_kernels": expected,
        })
        result["encoder"][layer] = {**block, "mlp": mlp}
        removed += len(expected) - len(paired)

    if len(result["encoder"]) != LAYERS or removed != LAYERS * PAIRS_PER_LAYER:
        raise ValueError("contract does not contain all 12 encoder layers")
    totals = dict(result["execution_totals"])
    totals["encoder_npu_calls"] = int(totals["encoder_npu_calls"]) - removed
    totals["npu_calls_per_inference"] = int(totals["npu_calls_per_inference"]) - removed
    totals["resident_kernel_variants"] = len(bank["cases"])
    result["execution_totals"] = totals
    result["bank"] = {
        **result.get("bank", {}),
        "bytes": int(bank["bank_size_bytes"]),
        "sha256": bank["bank_sha256"],
        "resident_kernels": len(bank["cases"]),
        "required_fm_io_bytes": int(bank["required_fm_io_bytes"]),
        "shared_fm_workspace_bytes": int(bank["shared_fm_workspace_bytes"]),
    }
    result["encoder_fc1_dispatch_policy"] = {
        "programs_per_layer": PAIRS_PER_LAYER,
        "outputs_per_program": 2,
        "physical_dispatch_reduction": removed,
        "logical_channel_order": "pair-major then output-index",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = enable_paired_fc1(
        json.loads(args.contract.read_text()),
        json.loads(args.bank_manifest.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "npu_calls_per_inference": result["execution_totals"]["npu_calls_per_inference"],
        "resident_kernels": result["bank"]["resident_kernels"],
        "bank_sha256": result["bank"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
