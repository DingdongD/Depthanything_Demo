#!/usr/bin/env python3
"""Replace all ViT patch-projection images with calibrated A8xB8 variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

try:
    from .replace_u250_fc1_pairs_in_bank import parse_cfg
except ImportError:
    from replace_u250_fc1_pairs_in_bank import parse_cfg


def tensor_interface(tensor: dict, *, include_bitdepth: bool = True) -> tuple:
    values = (tuple(tensor["dims"]), tensor["layout"])
    return values + ((int(tensor["bitdepth"]),) if include_bitdepth else ())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--input-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.input_scale <= 0:
        raise ValueError("input scale must be positive")

    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    host_plan = json.loads(args.host_plan.read_text())
    frontend = contract.get("frontend", {}).get("patch_projection", {})
    kernels = list(frontend.get("kernels", []))
    if len(kernels) != 6 or any(not name.startswith("patch_projection_co") for name in kernels):
        raise ValueError(f"unexpected patch-projection kernel set: {kernels}")

    bank_path = args.manifest.parent / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("baseline bank extent does not match manifest")
    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    positions = {item["name"]: index for index, item in enumerate(ordered)}
    by_name = {item["name"]: item for item in manifest["cases"]}
    replacements = []
    cfg_output = args.output_dir / "cfg"

    for name in kernels:
        if name not in positions:
            raise ValueError(f"kernel missing from resident bank: {name}")
        index = positions[name]
        old = by_name[name]
        begin = int(old["offset_bytes"])
        end = (int(ordered[index + 1]["offset_bytes"])
               if index + 1 < len(ordered) else len(bank))
        case_dir = args.compiled_dir / name
        cfg = case_dir / f"{name}_cfg.txt"
        binary = case_dir / f"{name}_ddr.bin"
        metadata = parse_cfg(cfg)
        payload = binary.read_bytes()
        if not payload or len(payload) % 256 or len(payload) > end - begin:
            raise ValueError(f"{name}: replacement image does not fit resident slot")
        if len(metadata["inputs"]) != 1 or len(metadata["outputs"]) != 1:
            raise ValueError(f"{name}: expected one input and one output")
        if tensor_interface(metadata["inputs"][0], include_bitdepth=False) != \
                tensor_interface(old["inputs"][0], include_bitdepth=False):
            raise ValueError(f"{name}: input shape/layout changed")
        if int(old["inputs"][0]["bitdepth"]) != 16 or \
                int(metadata["inputs"][0]["bitdepth"]) != 8:
            raise ValueError(f"{name}: expected BF16-to-INT8 input transition")
        if tensor_interface(metadata["outputs"][0]) != tensor_interface(old["outputs"][0]):
            raise ValueError(f"{name}: output tensor ABI changed")
        if metadata["isa_ranges"] != old["isa_ranges"]:
            raise ValueError(f"{name}: instruction ABI changed")

        bank[begin:end] = payload + bytes(end - begin - len(payload))
        relocation = begin // 256
        bases = [value + relocation for value in metadata["base_addresses"]]
        bases[4] = int(manifest["shared_fm_base_units"])
        replacement = {
            **old,
            "base_addresses_local": metadata["base_addresses"],
            "base_addresses": bases,
            "source_cfg": str(cfg.resolve()),
            "source_ddr": str(binary.resolve()),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": metadata["inputs"],
            "outputs": metadata["outputs"],
            "calibrated_input_scale": args.input_scale,
        }
        by_name[name] = replacement
        replacements.append({
            "kernel": name,
            "input_scale": args.input_scale,
            "policy": "calibrated-a8b8-in-place-preserve-bank-offset",
        })

    manifest["cases"] = [by_name[item["name"]] for item in manifest["cases"]]
    previous = list(manifest.get("kernel_replacements", []))
    if not previous and manifest.get("single_kernel_replacement"):
        previous.append(manifest["single_kernel_replacement"])
    manifest["kernel_replacements"] = previous + replacements
    manifest.pop("single_kernel_replacement", None)
    bank_sha = hashlib.sha256(bank).hexdigest()
    manifest["bank_sha256"] = bank_sha

    frontend["input_dtype"] = "INT8"
    frontend["input_quantization"] = {
        "dtype": "INT8", "scale": args.input_scale, "symmetric": True,
    }
    contract["bank"]["sha256"] = bank_sha
    contract["bank"]["bytes"] = len(bank)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / manifest["bank_file"]).write_bytes(bank)
    (args.output_dir / "resident_kernel_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "depthanything_u250_runtime_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "depthanything_u250_host_plan.json").write_text(
        json.dumps(host_plan, indent=2, sort_keys=True) + "\n")
    cfg_output.mkdir(parents=True, exist_ok=True)
    for name in kernels:
        shutil.copy2(args.compiled_dir / name / f"{name}_cfg.txt",
                     cfg_output / f"{name}_cfg.txt")
    print(json.dumps({
        "bank": str(args.output_dir / manifest["bank_file"]),
        "bank_sha256": bank_sha,
        "input_scale": args.input_scale,
        "replaced_kernels": kernels,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
