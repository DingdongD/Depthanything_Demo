#!/usr/bin/env python3
"""Relink a compiled joint calibration candidate into a resident U250 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from append_u250_fused_qkv_attention_bank import validate_and_rebind_codec_report
from replace_u250_fc1_pairs_in_bank import parse_cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--compiled-manifest", type=Path, required=True)
    parser.add_argument("--proposed-contract", type=Path, required=True)
    parser.add_argument("--proposed-host-plan", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-codec-report", type=Path,
        help=("qualified codec report for the deployed extension; defaults "
              "to the report bundled with the base package"),
    )
    parser.add_argument("--output-package", type=Path, required=True)
    args = parser.parse_args()
    if args.output_package.exists():
        raise ValueError(f"output already exists: {args.output_package}")
    compiled = json.loads(args.compiled_manifest.read_text())
    contract = json.loads(args.proposed_contract.read_text())
    plan = json.loads(args.proposed_host_plan.read_text())
    calibration = json.loads(args.calibration_report.read_text())
    sample_manifest = json.loads(args.calibration_manifest.read_text())
    fingerprints = {
        compiled.get("calibration_manifest_sha256"),
        contract.get("calibration", {}).get("manifest_sha256"),
        calibration.get("manifest_sha256"),
        sample_manifest.get("manifest_sha256"),
    }
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("joint calibration fingerprints are inconsistent")

    source_manifest_path = args.base_package / "resident_kernel_bank_manifest.json"
    manifest = json.loads(source_manifest_path.read_text())
    bank_path = args.base_package / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("base resident bank extent mismatch")
    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    positions = {item["name"]: index for index, item in enumerate(ordered)}
    replacements = {}
    for item in compiled["kernels"]:
        name = item["kernel"]
        if name not in positions:
            raise ValueError(f"compiled kernel is absent from resident bank: {name}")
        index = positions[name]
        old = ordered[index]
        begin = int(old["offset_bytes"])
        end = (int(ordered[index + 1]["offset_bytes"])
               if index + 1 < len(ordered) else len(bank))
        cfg, binary = Path(item["cfg"]), Path(item["binary"])
        payload = binary.read_bytes()
        metadata = parse_cfg(cfg)
        if not payload or len(payload) % 256 or len(payload) > end - begin:
            raise ValueError(
                f"{name}: replacement size {len(payload)} exceeds slot {end - begin}"
            )
        if metadata["inputs"] != old["inputs"] or metadata["outputs"] != old["outputs"]:
            raise ValueError(f"{name}: tensor ABI differs from resident slot")
        if metadata["isa_ranges"] != old["isa_ranges"]:
            raise ValueError(f"{name}: instruction ABI differs from resident slot")
        if begin % 256:
            raise ValueError(f"{name}: resident offset is not 256-byte aligned")
        bank[begin:end] = payload + bytes(end - begin - len(payload))
        bases = [value + begin // 256 for value in metadata["base_addresses"]]
        bases[4] = int(manifest["shared_fm_base_units"])
        replacements[name] = {
            **old, "base_addresses_local": metadata["base_addresses"],
            "base_addresses": bases, "source_cfg": str(cfg.resolve()),
            "source_ddr": str(binary.resolve()), "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "joint_calibration_manifest_sha256": next(iter(fingerprints)),
        }
    manifest["cases"] = [replacements.get(item["name"], item)
                         for item in manifest["cases"]]
    manifest["bank_sha256"] = hashlib.sha256(bank).hexdigest()
    manifest["joint_full_graph_calibration"] = {
        "manifest_sha256": next(iter(fingerprints)),
        "replacement_kernels": sorted(replacements),
        "replacement_count": len(replacements),
        "policy": "in-place-preserve-address-and-tensor-abi",
        "deployment_status": "candidate_requires_free_running_gate",
    }
    contract["bank"].update({
        "sha256": manifest["bank_sha256"], "bytes": len(bank),
        "resident_kernels": len(manifest["cases"]),
    })
    contract["calibration"]["deployment_status"] = (
        "candidate_requires_free_running_gate"
    )

    shutil.copytree(args.base_package, args.output_package)
    (args.output_package / manifest["bank_file"]).write_bytes(bank)
    (args.output_package / "resident_kernel_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    contract_path = args.output_package / "depthanything_u250_runtime_contract.json"
    plan_path = args.output_package / "depthanything_u250_host_plan.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    cfg_dir = args.output_package / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    for item in compiled["kernels"]:
        shutil.copyfile(item["cfg"], cfg_dir / f"{item['kernel']}_cfg.txt")
    shutil.copyfile(args.calibration_report,
                    args.output_package / "joint_calibration_report.json")
    shutil.copyfile(args.calibration_manifest,
                    args.output_package / "full_graph_calibration_manifest.json")
    source_report = args.source_codec_report
    if source_report is None:
        source_report = args.base_package / "native_codec_report_active.json"
        if not source_report.is_file():
            source_report = args.base_package / "native_codec_report.json"
    validate_and_rebind_codec_report(
        source_report, source_manifest_path,
        args.output_package / "resident_kernel_bank_manifest.json", cfg_dir,
        args.output_package / "native_codec_report_active.json",
    )
    print(json.dumps({
        "output_package": str(args.output_package.resolve()),
        "bank_sha256": manifest["bank_sha256"],
        "replacement_kernels": len(replacements),
        "deployment_status": contract["calibration"]["deployment_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
