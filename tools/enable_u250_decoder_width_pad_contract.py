#!/usr/bin/env python3
"""Replace one decoder Conv with width-padded physical kernel variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_source(items: list[dict], source: str, key: str) -> dict:
    matches = [item for item in items if item.get(key) == source]
    if len(matches) != 1:
        raise ValueError(f"expected one {source} entry in {key}, got {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--variant-manifest", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-node", default="/depth_head/resize_layers.3/Conv"
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
    variant = json.loads(args.variant_manifest.read_text())
    bank = json.loads(args.bank_manifest.read_text())
    kernels = variant["kernels"]
    bank_names = {item["name"] for item in bank["cases"]}
    missing = sorted({item["name"] for item in kernels} - bank_names)
    if missing:
        raise ValueError(f"width-padded kernels missing from bank: {missing}")

    contract_step = find_source(contract["decoder"], args.source_node, "source_node")
    plan_step = find_source(plan["decoder_steps"], args.source_node, "name")
    for step in (contract_step, plan_step):
        step["kernels"] = kernels
        step["physical_input_width"] = int(variant["input_width"])
        step["physical_output_width"] = int(variant["output_width"])
        step["logical_output_width"] = int(variant["crop_width"])
    contract["bank"] = {
        "bytes": int(bank["bank_size_bytes"]),
        "file": bank["bank_file"],
        "sha256": bank["bank_sha256"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_out = args.output_dir / "depthanything_u250_runtime_contract.json"
    plan_out = args.output_dir / "depthanything_u250_host_plan.json"
    contract_out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "bank_sha256": bank["bank_sha256"],
        "contract": str(contract_out),
        "host_plan": str(plan_out),
        "kernels_replaced": len(kernels),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
