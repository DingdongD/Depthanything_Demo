#!/usr/bin/env python3
"""Replace one resident U250 kernel image without moving any bank address."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replace_u250_fc1_pairs_in_bank import parse_cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    host_plan = json.loads(args.host_plan.read_text())
    bank_path = args.manifest.parent / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("baseline bank extent does not match manifest")

    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    positions = {item["name"]: index for index, item in enumerate(ordered)}
    if args.kernel not in positions:
        raise ValueError(f"kernel not found: {args.kernel}")
    index = positions[args.kernel]
    old = ordered[index]
    begin = int(old["offset_bytes"])
    end = (int(ordered[index + 1]["offset_bytes"])
           if index + 1 < len(ordered) else len(bank))
    payload = args.binary.read_bytes()
    if not payload or len(payload) % 256 or len(payload) > end - begin:
        raise ValueError("replacement image does not fit resident slot")

    metadata = parse_cfg(args.cfg)
    if metadata["inputs"] != old["inputs"] or metadata["outputs"] != old["outputs"]:
        raise ValueError("replacement tensor ABI differs from resident kernel")
    if metadata["isa_ranges"] != old["isa_ranges"]:
        raise ValueError("replacement instruction ABI differs from resident kernel")
    bank[begin:end] = payload + bytes(end - begin - len(payload))

    # CFG ISA/parameter base addresses use 256-byte units.  The resident
    # bank's larger slot alignment only reserves guard space between images.
    if begin % 256:
        raise ValueError("resident kernel offset is not 256-byte addressable")
    relocation = begin // 256
    bases = [value + relocation for value in metadata["base_addresses"]]
    bases[4] = int(manifest["shared_fm_base_units"])
    replacement = {
        **old,
        "base_addresses_local": metadata["base_addresses"],
        "base_addresses": bases,
        "source_cfg": str(args.cfg.resolve()),
        "source_ddr": str(args.binary.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "calibrated_input_scale": args.input_scale,
    }
    manifest["cases"] = [replacement if item["name"] == args.kernel else item
                         for item in manifest["cases"]]
    bank_sha = hashlib.sha256(bank).hexdigest()
    manifest["bank_sha256"] = bank_sha
    manifest["single_kernel_replacement"] = {
        "kernel": args.kernel,
        "input_scale": args.input_scale,
        "policy": "in-place-preserve-all-addresses",
    }

    changed = 0
    for step in contract["decoder"]:
        if any(kernel["name"] == args.kernel for kernel in step.get("kernels", [])):
            step["input_quantization"]["scale"] = args.input_scale
            changed += 1
    for layer in contract.get("encoder", []):
        qkv = layer.get("qkv", {})
        if qkv.get("kernel") == args.kernel:
            qkv["input_quantization"]["scale"] = args.input_scale
            changed += 1
    if changed != 1:
        raise ValueError(f"expected one runtime contract match, got {changed}")
    plan_changed = 0
    for step in host_plan["decoder_steps"]:
        if any(kernel["name"] == args.kernel for kernel in step.get("kernels", [])):
            step["input_scale"] = args.input_scale
            plan_changed += 1
    # Encoder QKV quantization lives in the runtime contract.  The host plan
    # only describes LayerNorm boundaries, so it intentionally has no match.
    if plan_changed not in (0, 1):
        raise ValueError(f"expected at most one host-plan match, got {plan_changed}")
    contract["bank"]["sha256"] = bank_sha
    contract["bank"]["bytes"] = len(bank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_out = args.output_dir / manifest["bank_file"]
    bank_out.write_bytes(bank)
    manifest_out = args.output_dir / "resident_kernel_bank_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract_out = args.output_dir / "depthanything_u250_runtime_contract.json"
    contract_out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    plan_out = args.output_dir / "depthanything_u250_host_plan.json"
    plan_out.write_text(json.dumps(host_plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "bank": str(bank_out), "bank_sha256": bank_sha,
        "contract": str(contract_out), "host_plan": str(plan_out),
        "input_scale": args.input_scale,
        "manifest": str(manifest_out), "slot_bytes": end - begin,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
