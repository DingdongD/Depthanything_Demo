#!/usr/bin/env python3
"""Append twelve fused QKV+attention programs before a resident bank's FM suffix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .replace_u250_fc1_pairs_in_bank import parse_cfg
    from .run_u250_depthanything_hybrid import enrich_cfg_tensors
    from .u250_layout_descriptors import build_case_descriptors
except ImportError:  # Direct execution keeps tools/ on sys.path.
    from replace_u250_fc1_pairs_in_bank import parse_cfg
    from run_u250_depthanything_hybrid import enrich_cfg_tensors
    from u250_layout_descriptors import build_case_descriptors


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def append_programs(
    manifest: dict, bank: bytes, programs: list[dict], alignment: int,
) -> tuple[dict, bytes]:
    """Append payloads while preserving the complete shared-FM suffix."""
    if manifest.get("shared_fm_placement") != "suffix":
        raise ValueError("resident bank must use suffix shared-FM placement")
    old_shared = int(manifest["shared_fm_base_units"]) * 256
    workspace = int(manifest["shared_fm_workspace_bytes"])
    if old_shared < 0 or old_shared + workspace > len(bank):
        raise ValueError("resident bank shared-FM suffix is outside the bank")
    if alignment <= 0 or alignment % 256:
        raise ValueError("alignment must be a positive multiple of 256")

    prefix = bytearray(bank[:old_shared])
    suffix = bank[old_shared:]
    records = list(manifest["cases"])
    existing = {record["name"] for record in records}
    appended = []
    for program in programs:
        name = program["name"]
        payload = bytes(program["payload"])
        metadata = program["metadata"]
        if name in existing:
            raise ValueError(f"resident kernel already exists: {name}")
        if not payload or len(payload) % 256:
            raise ValueError(f"{name}: payload must be non-empty and 256-byte aligned")
        offset = align(len(prefix), alignment)
        prefix.extend(bytes(offset - len(prefix)))
        prefix.extend(payload)
        relocation = offset // 256
        bases = [int(value) + relocation
                 for value in metadata["base_addresses"]]
        record = {
            "name": name,
            "group": "fused-qkv-attention/no-gain",
            "offset_bytes": offset,
            "size_bytes": len(payload),
            "sha256": sha256(payload),
            "source_cfg": str(Path(program["cfg"]).resolve()),
            "source_ddr": str(Path(program["binary"]).resolve()),
            "base_addresses_local": metadata["base_addresses"],
            "base_addresses": bases,
            "isa_ranges": metadata["isa_ranges"],
            "inputs": metadata["inputs"],
            "outputs": metadata["outputs"],
            "layer": int(program["layer"]),
            "output_order": "chunk-major",
        }
        records.append(record)
        appended.append(record)
        existing.add(name)

    new_shared = align(len(prefix), alignment)
    prefix.extend(bytes(new_shared - len(prefix)))
    new_shared_units = new_shared // 256
    for record in records:
        bases = list(record["base_addresses"])
        if len(bases) < 5:
            raise ValueError(f"{record['name']}: incomplete base-address ABI")
        bases[4] = new_shared_units
        record["base_addresses"] = bases
    image = bytes(prefix) + suffix
    result = dict(manifest)
    result.update({
        "cases": records,
        "bank_size_bytes": len(image),
        "bank_sha256": sha256(image),
        "shared_fm_base_units": new_shared_units,
        "fused_qkv_attention_append": {
            "kernels": [record["name"] for record in appended],
            "layers": [record["layer"] for record in appended],
            "policy": "append-before-shared-fm-suffix-preserve-fallbacks",
        },
    })
    return result, image


def validate_and_rebind_codec_report(
    source_report_path: Path,
    source_manifest_path: Path,
    manifest_path: Path,
    cfg_dir: Path,
    output_path: Path,
) -> None:
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_digest = sha256(source_manifest_bytes)
    report = json.loads(source_report_path.read_text())
    if report.get("qualified") is not True:
        raise ValueError("source native-codec report is not qualified")
    if report.get("manifest_sha256") != source_digest:
        raise ValueError("source native-codec report does not match source manifest")

    manifest_bytes = manifest_path.read_bytes()
    records = {item["name"]: item
               for item in json.loads(manifest_bytes)["cases"]}
    enriched = {}
    for name, record in records.items():
        cfg_path = cfg_dir / f"{name}_cfg.txt"
        enriched[name] = enrich_cfg_tensors(
            cfg_path, name, record, cfg_path.read_text()
        )
    descriptors = build_case_descriptors(enriched)
    qualified = {item["identity"]: item for item in report["descriptors"]}
    used = set()
    for name, directions in descriptors.items():
        for direction, values in directions.items():
            for descriptor in values:
                identity = descriptor.identity()
                entry = qualified.get(identity)
                if entry is None:
                    raise ValueError(
                        f"{name}: {direction} {descriptor.index}: descriptor "
                        f"{identity} has no source qualification"
                    )
                expected = asdict(descriptor)
                expected.pop("index")
                expected["dims"] = list(descriptor.dims)
                if {key: entry.get(key) for key in expected} != expected:
                    raise ValueError(f"{name}: source qualification fields differ")
                for flag in (
                    "native_exact", "pack_exact", "unpack_exact",
                    "production_enabled",
                ):
                    if entry.get(flag) is not True:
                        raise ValueError(f"{name}: descriptor is not {flag}")
                benchmark = entry.get("benchmark", {})
                operation = "pack" if direction == "input" else "unpack"
                if (benchmark.get("production_enabled") is not True
                        or benchmark.get("operation") != operation):
                    raise ValueError(
                        f"{name}: descriptor benchmark is not qualified"
                    )
                used.add(identity)

    report["manifest_sha256"] = sha256(manifest_bytes)
    report["rebound_from_manifest_sha256"] = source_digest
    report["rebind_reason"] = (
        "appended fused QKV+attention programs use only independently "
        "qualified production tensor descriptors"
    )
    report["rebind_active_descriptor_count"] = len(used)
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def install_qualified_codec_report(
    report_path: Path,
    manifest_path: Path,
    cfg_dir: Path,
    output_path: Path,
) -> None:
    """Install a full codec qualification produced for the appended package."""
    manifest_bytes = manifest_path.read_bytes()
    report = json.loads(report_path.read_text())
    if report.get("qualified") is not True:
        raise ValueError("supplied native-codec report is not qualified")
    if report.get("manifest_sha256") != sha256(manifest_bytes):
        raise ValueError(
            "supplied native-codec report does not match appended manifest"
        )

    records = {item["name"]: item
               for item in json.loads(manifest_bytes)["cases"]}
    enriched = {}
    for name, record in records.items():
        cfg_path = cfg_dir / f"{name}_cfg.txt"
        enriched[name] = enrich_cfg_tensors(
            cfg_path, name, record, cfg_path.read_text()
        )
    descriptors = build_case_descriptors(enriched)
    qualified = {item["identity"]: item for item in report["descriptors"]}
    for name, directions in descriptors.items():
        for direction, values in directions.items():
            for descriptor in values:
                entry = qualified.get(descriptor.identity())
                if entry is None:
                    raise ValueError(
                        f"{name}: {direction} {descriptor.index}: descriptor "
                        "has no supplied qualification"
                    )
                expected = asdict(descriptor)
                expected.pop("index")
                expected["dims"] = list(descriptor.dims)
                if {key: entry.get(key) for key in expected} != expected:
                    raise ValueError(
                        f"{name}: supplied qualification fields differ"
                    )
                for flag in (
                    "native_exact", "pack_exact", "unpack_exact",
                    "production_enabled",
                ):
                    if entry.get(flag) is not True:
                        raise ValueError(
                            f"{name}: supplied descriptor is not {flag}"
                        )
    shutil.copyfile(report_path, output_path)


def parse_layer_sources(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        layer_text, separator, directory = value.partition("=")
        if not separator:
            raise ValueError("--layer-source must use LAYER=DIRECTORY")
        layer = int(layer_text)
        if not 0 <= layer < 12 or layer in result:
            raise ValueError(f"invalid or duplicate fused layer: {layer}")
        result[layer] = Path(directory)
    return result


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",") if item]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique encoder layer indices")
    if any(layer < 0 or layer >= 12 for layer in layers):
        raise ValueError("--layers must be within [0, 11]")
    return sorted(layers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument(
        "--compiled-root", type=Path, required=True,
        help="root containing l00..l11 compiler output directories",
    )
    parser.add_argument(
        "--layer-source", action="append", default=[],
        help="override one compiler directory as LAYER=DIRECTORY",
    )
    parser.add_argument(
        "--layers", default=",".join(str(layer) for layer in range(12)),
        help="comma-separated encoder layers to append (default: all 12)",
    )
    parser.add_argument(
        "--qualified-codec-report", type=Path,
        help=(
            "full CPU codec qualification generated against the deterministic "
            "appended manifest; required when fusion introduces a new tensor "
            "descriptor"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    overrides = parse_layer_sources(args.layer_source)
    layers = parse_layers(args.layers)
    if not set(overrides) <= set(layers):
        raise ValueError("--layer-source overrides must be selected by --layers")
    source_manifest_path = (
        args.source_package / "resident_kernel_bank_manifest.json"
    )
    source_report_path = args.source_package / "native_codec_report_active.json"
    if not source_report_path.exists():
        source_report_path = args.source_package / "native_codec_report.json"
    manifest = json.loads(source_manifest_path.read_text())
    bank_path = args.source_package / manifest["bank_file"]
    bank = bank_path.read_bytes()
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("source resident bank extent differs from manifest")

    programs = []
    for layer in layers:
        directory = overrides.get(layer, args.compiled_root / f"l{layer:02d}")
        stem = f"qkv_attention_fused_l{layer:02d}_a8_to_12xbf16"
        cfg = directory / f"{stem}_cfg.txt"
        binary = directory / f"{stem}_ddr.bin"
        metadata = parse_cfg(cfg)
        if (len(metadata["inputs"]) != 1
                or metadata["inputs"][0]["bitdepth"] != 8
                or len(metadata["outputs"]) != 12
                or any(item["bitdepth"] != 16
                       for item in metadata["outputs"])):
            raise ValueError(f"layer {layer}: unexpected fused tensor ABI")
        programs.append({
            "name": f"qkv_attention_fused_l{layer:02d}",
            "layer": layer,
            "cfg": cfg,
            "binary": binary,
            "payload": binary.read_bytes(),
            "metadata": metadata,
        })

    updated, image = append_programs(
        manifest, bank, programs, int(manifest.get("alignment_bytes", 4096))
    )
    shutil.copytree(args.source_package, args.output_dir)
    output_bank = args.output_dir / updated["bank_file"]
    output_bank.write_bytes(image)
    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    manifest_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")

    cfg_dir = args.output_dir / "cfg"
    for program in programs:
        shutil.copyfile(
            program["cfg"], cfg_dir / f"{program['name']}_cfg.txt"
        )

    contract_path = args.output_dir / "depthanything_u250_runtime_contract.json"
    contract = json.loads(contract_path.read_text())
    for layer in layers:
        block = contract["encoder"][layer]
        heads = block["attention"]["heads"]
        if len(heads) != 6:
            raise ValueError(f"layer {layer}: expected six attention heads")
        block["qkv_attention_fused"] = {
            "kernel": f"qkv_attention_fused_l{layer:02d}",
            "input_quantization": dict(block["qkv"]["input_quantization"]),
            "output_layout": "attention_chunks",
            "output_order": "chunk-major",
            "heads": 6,
            "valid_widths_head_major": [
                width
                for head in heads
                for call in head["calls"]
                for width in (
                    int(call["q0_rows"][1]) - int(call["q0_rows"][0]),
                    int(call["q1_rows"][1]) - int(call["q1_rows"][0]),
                )
            ],
            "amplitude_gain": 1.0,
        }
    contract["bank"].update({
        "bytes": len(image),
        "sha256": updated["bank_sha256"],
        "resident_kernels": len(updated["cases"]),
    })
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    active_report = args.output_dir / "native_codec_report_active.json"
    if args.qualified_codec_report is not None:
        install_qualified_codec_report(
            args.qualified_codec_report, manifest_path, cfg_dir, active_report,
        )
    else:
        validate_and_rebind_codec_report(
            source_report_path, source_manifest_path, manifest_path, cfg_dir,
            active_report,
        )
    print(json.dumps({
        "appended_kernels": len(programs),
        "bank_bytes": len(image),
        "bank_sha256": updated["bank_sha256"],
        "output_dir": str(args.output_dir),
        "shared_fm_base_units": updated["shared_fm_base_units"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
