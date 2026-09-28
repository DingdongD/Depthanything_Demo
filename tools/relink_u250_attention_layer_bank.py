#!/usr/bin/env python3
"""Reflow a resident bank after replacing an attention layer with larger images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replace_u250_fc1_pairs_in_bank import parse_cfg


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--kernel-manifest", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
    kernel_manifest = json.loads(args.kernel_manifest.read_text())
    replacements = {item["name"]: item for item in kernel_manifest["kernels"]}
    if len(replacements) != 6:
        raise ValueError("exactly six attention heads are required")
    alignment = int(manifest["alignment_bytes"])
    if manifest.get("shared_fm_placement") != "suffix":
        raise ValueError("reflow currently requires suffix shared FM")

    image = bytearray()
    records = []
    for old in sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"])):
        name = old["name"]
        if name in replacements:
            cfg = args.compiled_dir / f"{name}_cfg.txt"
            binary = args.compiled_dir / f"{name}_ddr.bin"
        else:
            cfg = Path(old["source_cfg"])
            binary = Path(old["source_ddr"])
        metadata = parse_cfg(cfg)
        if metadata["inputs"] != old["inputs"] or metadata["outputs"] != old["outputs"]:
            raise ValueError(f"{name}: tensor ABI changed")
        payload = binary.read_bytes()
        if not payload or len(payload) % 256:
            raise ValueError(f"{name}: invalid DDR image extent")
        offset = align(len(image), alignment)
        image.extend(bytes(offset - len(image)))
        image.extend(payload)
        relocation = offset // 256
        bases = [value + relocation for value in metadata["base_addresses"]]
        records.append({
            **old,
            "group": args.compiled_dir.name if name in replacements else old["group"],
            "source_cfg": str(cfg.resolve()),
            "source_ddr": str(binary.resolve()),
            "offset_bytes": offset,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "base_addresses_local": metadata["base_addresses"],
            "base_addresses": bases,
            "isa_ranges": metadata["isa_ranges"],
            "inputs": metadata["inputs"],
            "outputs": metadata["outputs"],
        })

    shared_bytes = int(manifest["shared_fm_workspace_bytes"])
    shared_offset = align(len(image), alignment)
    image.extend(bytes(shared_offset - len(image)))
    image.extend(bytes(shared_bytes))
    image.extend(bytes(align(len(image), alignment) - len(image)))
    shared_units = shared_offset // 256
    for record in records:
        record["base_addresses"][4] = shared_units
    bank_sha = hashlib.sha256(image).hexdigest()
    replacement_record = {
        "layer": int(kernel_manifest["layer"]),
        "kernels": sorted(replacements),
        "policy": "full-bank-reflow-for-larger-instruction-images",
        "strategy": kernel_manifest["strategy"],
        "fine_step": kernel_manifest["fine_step"],
        "threshold": kernel_manifest["threshold"],
        "residual_step": kernel_manifest["residual_step"],
        "per_head_probability": kernel_manifest.get("per_head_probability", {}),
        "v_unchanged": True,
        "av_output_gain": 1.0,
    }
    replacement_history = list(manifest.get("attention_layer_replacements", []))
    previous = manifest.get("attention_layer_replacement")
    if not replacement_history:
        for layer_index, layer_record in enumerate(contract.get("encoder", [])):
            probability = layer_record.get("attention", {}).get(
                "dual_range_probability"
            )
            if not probability:
                continue
            inherited = {
                "layer": layer_index,
                "kernels": [
                    f"attention2_l{layer_index:02d}_h{head:02d}"
                    for head in range(6)
                ],
                "policy": "inherited-dual-range-layer",
                **probability,
            }
            if (previous and previous.get("kernels")
                    == inherited["kernels"]):
                inherited.update(previous)
                inherited["layer"] = layer_index
            replacement_history.append(inherited)
    replacement_history = [
        item for item in replacement_history
        if int(item.get("layer", -1)) != int(kernel_manifest["layer"])
    ]
    replacement_history.append(replacement_record)
    manifest.update({
        "bank_size_bytes": len(image),
        "bank_sha256": bank_sha,
        "shared_fm_base_units": shared_units,
        "cases": records,
        "attention_layer_replacement": replacement_record,
        "attention_layer_replacements": replacement_history,
    })
    contract["bank"].update({"bytes": len(image), "sha256": bank_sha})
    attention = contract["encoder"][int(kernel_manifest["layer"])]["attention"]
    attention["implementation"] = (
        "fixed-scale INT8 QK + dual-range SPU/EPU A8 probability + two INT8 AV"
    )
    attention["dual_range_probability"] = {
        "fine_step": kernel_manifest["fine_step"],
        "threshold": kernel_manifest["threshold"],
        "residual_step": kernel_manifest["residual_step"],
        "heads": kernel_manifest.get("per_head_probability", {}),
        "v_unchanged": True,
        "av_output_gain": 1.0,
    }
    for item in kernel_manifest["kernels"]:
        head = int(item["head"])
        emitted = item.get("scales_bf16", {})
        runtime_scales = attention["heads"][head]["scales_bf16"]
        for name in ("q", "k", "v"):
            if name in emitted:
                runtime_scales[name] = float(emitted[name])
        runtime_scales["av_v"] = float(runtime_scales["v"])
        runtime_scales["av_output_gain"] = 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / manifest["bank_file"]).write_bytes(image)
    (args.output_dir / "resident_kernel_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "depthanything_u250_runtime_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "depthanything_u250_host_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "bank_bytes": len(image), "bank_sha256": bank_sha,
        "cases": len(records), "shared_fm_base_units": shared_units,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
