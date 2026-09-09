#!/usr/bin/env python3
"""Replace six attention heads in-place while preserving resident addresses."""

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
    parser.add_argument("--kernel-manifest", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
    kernel_manifest = json.loads(args.kernel_manifest.read_text())
    bank_path = args.manifest.parent / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    positions = {item["name"]: index for index, item in enumerate(ordered)}
    replacements = {}
    for record in kernel_manifest["kernels"]:
        name = record["name"]
        index = positions[name]
        old = ordered[index]
        begin = int(old["offset_bytes"])
        end = (int(ordered[index + 1]["offset_bytes"])
               if index + 1 < len(ordered) else len(bank))
        cfg = args.compiled_dir / f"{name}_cfg.txt"
        binary = args.compiled_dir / f"{name}_ddr.bin"
        payload = binary.read_bytes()
        metadata = parse_cfg(cfg)
        if len(payload) > end - begin or len(payload) % 256:
            raise ValueError(f"{name}: replacement image does not fit")
        if metadata["inputs"] != old["inputs"] or metadata["outputs"] != old["outputs"]:
            raise ValueError(f"{name}: tensor ABI differs")
        if metadata["isa_ranges"] != old["isa_ranges"]:
            raise ValueError(f"{name}: instruction ABI differs")
        if begin % 256:
            raise ValueError(f"{name}: offset is not 256-byte addressable")
        bank[begin:end] = payload + bytes(end - begin - len(payload))
        bases = [value + begin // 256 for value in metadata["base_addresses"]]
        bases[4] = int(manifest["shared_fm_base_units"])
        replacements[name] = {
            **old,
            "base_addresses_local": metadata["base_addresses"],
            "base_addresses": bases,
            "source_cfg": str(cfg.resolve()),
            "source_ddr": str(binary.resolve()),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "calibrated_k_scale": record["k_scale_bf16"],
        }
    if len(replacements) != 6:
        raise ValueError(f"expected six replacements, got {len(replacements)}")
    manifest["cases"] = [replacements.get(item["name"], item)
                         for item in manifest["cases"]]
    bank_sha = hashlib.sha256(bank).hexdigest()
    manifest["bank_sha256"] = bank_sha
    manifest["attention_layer_replacement"] = {
        "kernels": sorted(replacements),
        "policy": "in-place-preserve-all-addresses",
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
        json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bank_sha256": bank_sha, "kernels": sorted(replacements)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
