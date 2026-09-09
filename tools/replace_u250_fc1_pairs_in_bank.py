#!/usr/bin/env python3
"""Replace pairs of FC1 images in place without moving any later U250 address."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


BASE_KEYS = (
    "cfg_isa_base_addr", "isa_wei_base_addr", "isa_sca_base_addr",
    "isa_cwp_base_addr", "isa_fm_base_addr", "isa_emac_base_addr",
)


def parse_tensor(line: str) -> dict:
    def number(pattern: str) -> int:
        match = re.search(pattern, line)
        if match is None:
            raise ValueError(f"malformed cfg tensor line: {line}")
        return int(match.group(1), 0)

    dims = re.search(r"Dims: \[([^]]+)\]", line)
    layout = re.search(r"Layout: (\S+)", line)
    if dims is None or layout is None:
        raise ValueError(f"malformed cfg tensor metadata: {line}")
    return {
        "address": number(r"Address: (\d+)"),
        "size_per_bank": number(r"Size: (\d+)"),
        "dims": [int(value.strip()) for value in dims.group(1).split(",")],
        "bitdepth": number(r"bitdepth: (\d+)"),
        "layout": layout.group(1),
    }


def parse_cfg(path: Path) -> dict:
    values = {}
    inputs = []
    outputs = []
    for line in path.read_text().splitlines():
        if line.startswith("Address: "):
            inputs.append(parse_tensor(line))
        elif line.startswith("Output Address: "):
            outputs.append(parse_tensor(line.removeprefix("Output ")))
        elif ":" in line:
            key, raw = line.split(":", 1)
            if key in BASE_KEYS or key in {"isa_op_range", "isa_emac_range"}:
                values[key] = int(raw.strip(), 0)
    missing = [key for key in BASE_KEYS if key not in values]
    if missing or "isa_op_range" not in values or "isa_emac_range" not in values:
        raise ValueError(f"{path}: incomplete address metadata")
    return {
        "base_addresses": [values[key] for key in BASE_KEYS],
        "isa_ranges": [values["isa_op_range"], values["isa_emac_range"]],
        "inputs": inputs,
        "outputs": outputs,
    }


def replace_pairs(manifest: dict, bank: bytes, compiled_dir: Path) -> tuple[dict, bytes]:
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("baseline bank extent does not match its manifest")
    records = list(manifest["cases"])
    by_name = {record["name"]: record for record in records}
    by_offset = sorted(records, key=lambda record: int(record["offset_bytes"]))
    offset_position = {record["name"]: index
                       for index, record in enumerate(by_offset)}
    next_offset = {
        record["name"]: int(by_offset[index + 1]["offset_bytes"])
        for index, record in enumerate(by_offset[:-1])
    }
    image = bytearray(bank)
    replacements = {}
    total_guard = 0
    for layer in range(12):
        for pair in range(3):
            first_name = f"mlp_fc1_l{layer:02d}_c{pair * 2:02d}"
            second_name = f"mlp_fc1_l{layer:02d}_c{pair * 2 + 1:02d}"
            if first_name not in by_name or second_name not in by_name:
                raise ValueError(f"missing original FC1 slots for layer {layer} pair {pair}")
            first = by_name[first_name]
            second = by_name[second_name]
            first_position = offset_position[first_name]
            second_position = offset_position[second_name]
            if (second_position != first_position + 1
                    or by_offset[first_position + 1]["name"] != second_name):
                raise ValueError("paired FC1 images are not adjacent in the resident bank")
            begin = int(first["offset_bytes"])
            second_begin = int(second["offset_bytes"])
            end = next_offset.get(second_name)
            if end is None or second_begin <= begin or end <= second_begin:
                raise ValueError("FC1 slots are not contiguous in the resident bank")
            name = f"mlp_fc1_pair_l{layer:02d}_p{pair:02d}"
            cfg = compiled_dir / f"{name}_cfg.txt"
            binary = compiled_dir / f"{name}_ddr.bin"
            payload = binary.read_bytes()
            if not payload or len(payload) % 256 or len(payload) > end - begin:
                raise ValueError(f"{name}: paired image does not fit original slots")
            metadata = parse_cfg(cfg)
            if len(metadata["inputs"]) != 1 or len(metadata["outputs"]) != 2:
                raise ValueError(f"{name}: expected one input and two outputs")
            image[begin:end] = payload + bytes(end - begin - len(payload))
            relocation = begin // 256
            bases = [value + relocation for value in metadata["base_addresses"]]
            bases[4] = int(manifest["shared_fm_base_units"])
            guard = end - begin - len(payload)
            total_guard += guard
            replacements[first_name] = {
                "group": compiled_dir.name,
                "name": name,
                "source_ddr": str(binary.resolve()),
                "source_cfg": str(cfg.resolve()),
                "offset_bytes": begin,
                "size_bytes": len(payload),
                "slot_bytes": end - begin,
                "guard_bytes": guard,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "base_addresses_local": metadata["base_addresses"],
                "base_addresses": bases,
                "isa_ranges": metadata["isa_ranges"],
                "inputs": metadata["inputs"],
                "outputs": metadata["outputs"],
                "replaces": [first_name, second_name],
            }
            replacements[second_name] = None

    cases = []
    for record in records:
        replacement = replacements.get(record["name"], record)
        if replacement is not None:
            cases.append(replacement)
    if len(cases) != len(records) - 36:
        raise ValueError("paired replacement did not reduce exactly 36 resident cases")
    result = {**manifest, "cases": cases}
    result.update({
        "bank_sha256": hashlib.sha256(image).hexdigest(),
        "fc1_pair_replacement_policy": "in-place-preserve-all-later-addresses",
        "fc1_pair_replacements": 36,
        "fc1_pair_guard_bytes": total_guard,
    })
    return result, bytes(image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    bank_path = args.manifest.parent / manifest["bank_file"]
    result, bank = replace_pairs(manifest, bank_path.read_bytes(), args.compiled_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / result["bank_file"]).write_bytes(bank)
    output = args.output_dir / "resident_kernel_bank_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(output), "cases": len(result["cases"]),
        "bank_size_bytes": len(bank), "bank_sha256": result["bank_sha256"],
        "guard_bytes": result["fc1_pair_guard_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
