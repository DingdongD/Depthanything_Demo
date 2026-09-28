#!/usr/bin/env python3
"""Install one ABI-changing fused LayerNorm+QKV image in a resident bank.

The binary remains in the existing QKV instruction slot, while the tensor ABI
is intentionally changed from host-normalized INT8 to residual BF16.  Every
codec-visible descriptor must already have an exact production qualification
in the source native-codec report; this tool never manufactures one.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from replace_u250_fc1_pairs_in_bank import parse_cfg
from run_u250_depthanything_hybrid import enrich_cfg_tensors
from u250_layout_descriptors import build_case_descriptors


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    records = {item["name"]: item for item in json.loads(manifest_bytes)["cases"]}
    enriched = {}
    for name, record in records.items():
        cfg_path = cfg_dir / f"{name}_cfg.txt"
        cfg_text = cfg_path.read_text()
        enriched[name] = enrich_cfg_tensors(cfg_path, name, record, cfg_text)
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
                actual = {key: entry.get(key) for key in expected}
                if actual != expected:
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
                    raise ValueError(f"{name}: descriptor benchmark is not qualified")
                used.add(identity)

    report["manifest_sha256"] = sha256(manifest_bytes)
    report["rebound_from_manifest_sha256"] = source_digest
    report["rebind_reason"] = (
        "fused LayerNorm+QKV uses only independently qualified BF16/NDWC "
        "and existing output descriptors; all active descriptors revalidated"
    )
    report["rebind_active_descriptor_count"] = len(used)
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--kernel", default="qkv_projection_l00")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--core-input-scale", type=float, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--projection-layout", choices=("branched", "combined"),
        default="branched",
    )
    parser.add_argument(
        "--probe-with-vendor-codec", action="store_true",
        help="Permit a new descriptor only for a vendor-codec board probe",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.source_package, args.output_dir)
    source_manifest_path = args.source_package / "resident_kernel_bank_manifest.json"
    source_report_path = args.source_package / "native_codec_report_active.json"
    if not source_report_path.exists():
        source_report_path = args.source_package / "native_codec_report.json"

    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    contract_path = args.output_dir / "depthanything_u250_runtime_contract.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    bank_path = args.output_dir / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
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
        raise ValueError("replacement image does not fit the resident slot")

    metadata = parse_cfg(args.cfg)
    logical_output_fields = ("size_per_bank", "dims", "bitdepth", "layout")
    if args.projection_layout == "branched":
        if ([{key: item[key] for key in logical_output_fields}
             for item in metadata["outputs"]]
                != [{key: item[key] for key in logical_output_fields}
                    for item in old["outputs"]]):
            raise ValueError("fused QKV outputs must preserve the existing Q/K/V ABI")
    else:
        output = metadata["outputs"]
        if (len(output) != 1 or output[0]["dims"][:-1] != old["outputs"][0]["dims"][:-1]
                or output[0]["dims"][-1] != sum(item["dims"][-1] for item in old["outputs"])
                or output[0]["bitdepth"] != 16 or output[0]["layout"] != "NDWC"):
            raise ValueError("combined QKV output does not concatenate the old outputs")
    if metadata["isa_ranges"] != old["isa_ranges"]:
        raise ValueError("fused QKV instruction ABI differs from resident slot")
    inputs = metadata["inputs"]
    if (len(inputs) != 1 or inputs[0]["bitdepth"] != 16
            or inputs[0]["dims"] != old["inputs"][0]["dims"]
            or inputs[0]["layout"] != old["inputs"][0]["layout"]):
        raise ValueError("expected exactly one shape-preserving BF16 QKV input")
    address_unit_bytes = 128
    fm_peak = max(
        int(item["address"]) * address_unit_bytes + int(item["size_per_bank"])
        for item in inputs + metadata["outputs"]
    )
    if fm_peak > int(manifest["shared_fm_workspace_bytes"]):
        raise ValueError("fused QKV feature-memory extent exceeds shared workspace")

    bank[begin:end] = payload + bytes(end - begin - len(payload))
    relocation = begin // 256
    bases = [value + relocation for value in metadata["base_addresses"]]
    bases[4] = int(manifest["shared_fm_base_units"])
    replacement = {
        **old,
        "group": f"fused_norm_qkv/{args.variant}",
        "source_cfg": str(args.cfg.resolve()),
        "source_ddr": str(args.binary.resolve()),
        "size_bytes": len(payload),
        "sha256": sha256(payload),
        "base_addresses_local": metadata["base_addresses"],
        "base_addresses": bases,
        "inputs": inputs,
        "outputs": metadata["outputs"],
        "isa_ranges": metadata["isa_ranges"],
        "fused_layernorm": {
            "affine": "folded_into_qkv_weights_and_biases",
            "core_input_scale": args.core_input_scale,
            "variant": args.variant,
        },
    }
    manifest["cases"] = [
        replacement if item["name"] == args.kernel else item
        for item in manifest["cases"]
    ]
    bank_sha = sha256(bank)
    manifest["bank_sha256"] = bank_sha
    history = list(manifest.get("kernel_replacements", []))
    history.append({
        "kernel": args.kernel,
        "layer": args.layer,
        "policy": "in-place-fused-layernorm-qkv-bf16-input",
        "variant": args.variant,
        "core_input_scale": args.core_input_scale,
        "affine_gain": 1.0,
    })
    manifest["kernel_replacements"] = history

    block = contract["encoder"][args.layer]
    if block["qkv"]["kernel"] != args.kernel:
        raise ValueError("runtime contract layer does not reference selected QKV")
    block["host_norm1"] = {
        "op": "LayerNormalization",
        "precision": "BF16",
        "fused_into": args.kernel,
        "affine": "folded_into_qkv_weights_and_biases",
        "core_input_quantization": {
            "dtype": "INT8",
            "scale": args.core_input_scale,
            "symmetric": True,
        },
        "variant": args.variant,
    }
    block["qkv"].pop("input_quantization", None)
    block["qkv"].update({
        "input_precision": "BF16",
        "fused_layernorm": True,
        "affine_gain": 1.0,
    })
    if args.projection_layout == "combined":
        block["qkv"]["output_layout"] = "concatenated_qkv"
    contract["bank"].update({"bytes": len(bank), "sha256": bank_sha})

    bank_path.write_bytes(bank)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    canonical_cfg = args.output_dir / "cfg" / f"{args.kernel}_cfg.txt"
    shutil.copyfile(args.cfg, canonical_cfg)
    if args.probe_with_vendor_codec:
        (args.output_dir / "UNQUALIFIED_CODEC_PROBE.json").write_text(
            json.dumps({
                "reason": "new combined-QKV descriptor requires board qualification",
                "required_layout_codec": "vendor",
            }, indent=2, sort_keys=True) + "\n"
        )
    else:
        validate_and_rebind_codec_report(
            source_report_path,
            source_manifest_path,
            manifest_path,
            args.output_dir / "cfg",
            args.output_dir / "native_codec_report_active.json",
        )
    print(json.dumps({
        "bank_sha256": bank_sha,
        "fm_peak_bytes_per_bank": fm_peak,
        "input": inputs[0],
        "kernel": args.kernel,
        "output_dir": str(args.output_dir),
        "slot_bytes": end - begin,
        "variant": args.variant,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
