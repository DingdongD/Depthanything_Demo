#!/usr/bin/env python3
"""Build a minimal resident-bank package for one compiled U250 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replace_u250_fc1_pairs_in_bank import parse_cfg


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--workspace-bytes", type=int, default=24 * 1024 * 1024)
    args = parser.parse_args()

    metadata = parse_cfg(args.cfg)
    payload = args.binary.read_bytes()
    if not payload or len(payload) % 256:
        raise ValueError("compiled DDR image must be non-empty and 256-byte aligned")
    if args.alignment <= 0 or args.alignment % 256:
        raise ValueError("bank alignment must be a positive multiple of 256")
    if args.workspace_bytes <= 0 or args.workspace_bytes % 256:
        raise ValueError("workspace size must be a positive multiple of 256")

    shared_offset = align(len(payload), args.alignment)
    image = payload + bytes(shared_offset - len(payload) + args.workspace_bytes)
    image += bytes(align(len(image), args.alignment) - len(image))
    shared_units = shared_offset // 256
    bases = list(metadata["base_addresses"])
    bases[4] = shared_units
    bank_name = f"{args.name}_resident_bank.bin"
    bank_sha = hashlib.sha256(image).hexdigest()
    record = {
        "name": args.name,
        "group": "standalone-probe",
        "offset_bytes": 0,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_cfg": str(args.cfg.resolve()),
        "source_ddr": str(args.binary.resolve()),
        "base_addresses_local": metadata["base_addresses"],
        "base_addresses": bases,
        "isa_ranges": metadata["isa_ranges"],
        "inputs": metadata["inputs"],
        "outputs": metadata["outputs"],
    }
    manifest = {
        "schema_version": 1,
        "bank_file": bank_name,
        "bank_size_bytes": len(image),
        "bank_sha256": bank_sha,
        "alignment_bytes": args.alignment,
        "shared_fm_placement": "suffix",
        "shared_fm_base_units": shared_units,
        "shared_fm_workspace_bytes": args.workspace_bytes,
        "cases": [record],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / bank_name).write_bytes(image)
    (args.output_dir / "resident_kernel_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    cfg_dir = args.output_dir / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / f"{args.name}_cfg.txt").write_bytes(args.cfg.read_bytes())
    io_order = args.cfg.with_name(
        args.cfg.name.replace("_cfg.txt", ".IO_order.yaml")
    )
    if io_order.exists():
        (cfg_dir / f"{args.name}.IO_order.yaml").write_bytes(
            io_order.read_bytes()
        )
    print(json.dumps({
        "bank_bytes": len(image),
        "bank_sha256": bank_sha,
        "kernel": args.name,
        "shared_fm_base_units": shared_units,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
