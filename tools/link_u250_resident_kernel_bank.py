#!/usr/bin/env python3
"""Relocate compiled DS images into one once-loaded U250 DDR kernel bank."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shutil


BASE_KEYS = (
    "cfg_isa_base_addr", "isa_wei_base_addr", "isa_sca_base_addr",
    "isa_cwp_base_addr", "isa_fm_base_addr", "isa_emac_base_addr",
)


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_tensor(line: str) -> dict:
    def number(pattern: str) -> int:
        match = re.search(pattern, line)
        if match is None:
            raise ValueError(f"malformed cfg tensor line: {line}")
        return int(match.group(1), 0)
    dims_match = re.search(r"Dims: \[([^]]+)\]", line)
    if dims_match is None:
        raise ValueError(f"missing dims: {line}")
    return {
        "address": number(r"Address: (\d+)"),
        "size_per_bank": number(r"Size: (\d+)"),
        "dims": [int(x.strip()) for x in dims_match.group(1).split(",")],
        "bitdepth": number(r"bitdepth: (\d+)"),
        "layout": re.search(r"Layout: (\S+)", line).group(1),
    }


def parse_cfg(path: Path) -> dict:
    text = path.read_text()
    values = {}
    inputs = []
    outputs = []
    for line in text.splitlines():
        if line.startswith("Address: "):
            inputs.append(parse_tensor(line))
        elif line.startswith("Output Address: "):
            outputs.append(parse_tensor(line.removeprefix("Output ")))
        elif ":" in line:
            key, raw = line.split(":", 1)
            if key in BASE_KEYS or key in ("isa_op_range", "isa_emac_range"):
                values[key] = int(raw.strip(), 0)
    missing = [key for key in BASE_KEYS if key not in values]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return {
        "base_addresses": [values[key] for key in BASE_KEYS],
        "isa_ranges": [values["isa_op_range"], values["isa_emac_range"]],
        "inputs": inputs, "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude-path", action="append", default=[])
    parser.add_argument(
        "--shared-fm-workspace-bytes",
        type=int,
        default=0,
        help="reserve an interleaved low-address FM arena shared by every case",
    )
    parser.add_argument(
        "--shared-fm-placement",
        choices=("prefix", "suffix"),
        default="suffix",
    )
    args = parser.parse_args()
    if args.alignment < 256 or args.alignment % 256:
        raise ValueError("alignment must be a multiple of 256 bytes")
    if (args.shared_fm_workspace_bytes < 0
            or args.shared_fm_workspace_bytes % args.alignment):
        raise ValueError("shared FM workspace must be alignment-sized")
    cases = []
    for directory in args.case_dir:
        cfg_paths = directory.rglob("*_cfg.txt") if args.recursive else directory.glob("*_cfg.txt")
        for cfg in sorted(cfg_paths):
            relative_cfg = cfg.relative_to(directory).as_posix()
            if any(fnmatch.fnmatch(relative_cfg, pattern)
                   for pattern in args.exclude_path):
                continue
            stem = cfg.name.removesuffix("_cfg.txt")
            if args.include and not any(fnmatch.fnmatch(stem, pattern)
                                        for pattern in args.include):
                continue
            binary = cfg.parent / f"{stem}_ddr.bin"
            if not binary.is_file():
                raise FileNotFoundError(binary)
            relative_parent = cfg.parent.relative_to(directory)
            group = directory.name if str(relative_parent) == "." else (
                directory.name + "/" + str(relative_parent)
            )
            cases.append((group, stem, cfg, binary))
    if not cases:
        raise ValueError("no compiled cases found")
    names = [stem for _, stem, _, _ in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate case names: {duplicates}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank = bytearray(
        args.shared_fm_workspace_bytes
        if args.shared_fm_workspace_bytes and args.shared_fm_placement == "prefix"
        else 0
    )
    records = []
    for group, stem, cfg, binary in cases:
        offset = align(len(bank), args.alignment)
        bank.extend(b"\0" * (offset - len(bank)))
        payload = binary.read_bytes()
        if len(payload) % 256:
            raise ValueError(f"{binary}: size is not 256-byte aligned")
        bank.extend(payload)
        metadata = parse_cfg(cfg)
        relocation_units = offset // 256
        relocated_bases = [x + relocation_units
                           for x in metadata["base_addresses"]]
        records.append({
            "group": group, "name": stem,
            "source_ddr": str(binary.resolve()),
            "source_cfg": str(cfg.resolve()),
            "offset_bytes": offset, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "base_addresses_local": metadata["base_addresses"],
            "base_addresses": relocated_bases,
            "isa_ranges": metadata["isa_ranges"],
            "inputs": metadata["inputs"], "outputs": metadata["outputs"],
        })
    required_fm_bytes = max(
        (
            tensor["address"] * 256 + tensor["size_per_bank"] * 2
            for record in records
            for tensor in record["inputs"] + record["outputs"]
        ),
        default=0,
    )
    if (args.shared_fm_workspace_bytes
            and args.shared_fm_workspace_bytes < required_fm_bytes):
        raise ValueError(
            f"shared FM workspace {args.shared_fm_workspace_bytes} is smaller "
            f"than network IO requirement {required_fm_bytes}"
        )
    shared_fm_base_bytes = None
    if args.shared_fm_workspace_bytes:
        if args.shared_fm_placement == "prefix":
            shared_fm_base_bytes = 0
        else:
            shared_fm_base_bytes = align(len(bank), args.alignment)
            bank.extend(b"\0" * (shared_fm_base_bytes - len(bank)))
            bank.extend(b"\0" * args.shared_fm_workspace_bytes)
        shared_fm_units = shared_fm_base_bytes // 256
        for record in records:
            record["base_addresses"][4] = shared_fm_units
    bank.extend(b"\0" * (align(len(bank), args.alignment) - len(bank)))
    output = args.output_dir / "depthanything_u250_resident_kernel_bank.bin"
    output.write_bytes(bank)
    manifest = {
        "schema_version": 1,
        "format": "DS two-bank interleaved image",
        "alignment_bytes": args.alignment,
        "bank_file": output.name,
        "bank_size_bytes": len(bank),
        "bank_sha256": hashlib.sha256(bank).hexdigest(),
        "static_h2c_writes": 2,
        "shared_fm_base_units": (
            shared_fm_base_bytes // 256
            if shared_fm_base_bytes is not None else None
        ),
        "shared_fm_placement": args.shared_fm_placement,
        "shared_fm_workspace_bytes": args.shared_fm_workspace_bytes,
        "required_fm_io_bytes": required_fm_bytes,
        "cases": records,
    }
    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cfg_dir = args.output_dir / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    for _, stem, cfg, _ in cases:
        shutil.copyfile(cfg, cfg_dir / f"{stem}_cfg.txt")
        io_order = cfg.with_name(cfg.name.replace("_cfg.txt", ".IO_order.yaml"))
        if io_order.is_file():
            shutil.copyfile(io_order, cfg_dir / f"{stem}.IO_order.yaml")
    print(json.dumps({"cases": len(records), "bytes": len(bank),
                      "sha256": manifest["bank_sha256"],
                      "cfg_dir": str(cfg_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
