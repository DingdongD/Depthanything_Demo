#!/usr/bin/env python3
"""Install one static-INT8 per-head QKV image in a resident bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from replace_u250_fc1_pairs_in_bank import parse_cfg
from replace_u250_fused_norm_qkv_in_bank import validate_and_rebind_codec_report


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--kernel", default="qkv_projection_l00")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--probe-with-vendor-codec", action="store_true",
        help="leave new output descriptors unqualified for a vendor-codec probe",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.source_package, args.output_dir)

    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    contract_path = args.output_dir / "depthanything_u250_runtime_contract.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    policy = model_manifest.get("policy")
    if (int(model_manifest["layer"]) != args.layer
            or policy not in {
                "static-head-scales-no-gain",
                "attention-input-abi-static-head-scales-no-gain",
            }):
        raise ValueError("model manifest does not describe the selected layer")

    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    index = next((index for index, item in enumerate(ordered)
                  if item["name"] == args.kernel), None)
    if index is None:
        raise ValueError(f"kernel not found: {args.kernel}")
    old = ordered[index]
    begin = int(old["offset_bytes"])
    end = (int(ordered[index + 1]["offset_bytes"])
           if index + 1 < len(ordered) else int(manifest["bank_size_bytes"]))
    payload = args.binary.read_bytes()
    if not payload or len(payload) % 256 or len(payload) > end - begin:
        raise ValueError("replacement image does not fit the resident slot")
    metadata = parse_cfg(args.cfg)
    if metadata["inputs"] != old["inputs"]:
        raise ValueError("head-split QKV must preserve the existing input ABI")
    outputs = metadata["outputs"]
    attention_ready = policy.startswith("attention-input-abi-")
    expected_count = 24 if attention_ready else 18
    if len(outputs) != expected_count:
        raise ValueError(f"expected {expected_count} outputs, got {len(outputs)}")
    expected_dims = list(old["outputs"][0]["dims"])
    expected_dims[-1] //= 6
    if attention_ready:
        expected = (
            [("NDWC", [1, 1, 256, 64])] * 6
            + [("NCHW", [1, 64, 1, 401])] * 6
            + [("NDWC", expected_dims)] * 6
            + [("NDWC", [1, 1, 145, 64])] * 6
        )
        for output, (layout, dims) in zip(outputs, expected):
            if (output["dims"] != dims or output["bitdepth"] != 8
                    or output["layout"] != layout):
                raise ValueError("attention-ready QKV output ABI differs")
    else:
        for output in outputs:
            if (output["dims"] != expected_dims or output["bitdepth"] != 8
                    or output["layout"] != old["outputs"][0]["layout"]):
                raise ValueError(
                    "head-split output ABI is not 1/6-width INT8 NDWC"
                )
    if len(model_manifest["outputs"]) != len(outputs):
        raise ValueError("model manifest output count differs from cfg")

    bank_path = args.output_dir / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    bank[begin:end] = payload + bytes(end - begin - len(payload))
    relocation = begin // 256
    bases = [value + relocation for value in metadata["base_addresses"]]
    bases[4] = int(manifest["shared_fm_base_units"])
    replacement = {
        **old,
        "group": "qkv-head-split-int8",
        "source_cfg": str(args.cfg.resolve()),
        "source_ddr": str(args.binary.resolve()),
        "size_bytes": len(payload),
        "sha256": sha256(payload),
        "base_addresses_local": metadata["base_addresses"],
        "base_addresses": bases,
        "inputs": metadata["inputs"],
        "outputs": outputs,
        "isa_ranges": metadata["isa_ranges"],
    }
    manifest["cases"] = [
        replacement if item["name"] == args.kernel else item
        for item in manifest["cases"]
    ]
    bank_sha = sha256(bank)
    manifest["bank_sha256"] = bank_sha
    manifest.setdefault("kernel_replacements", []).append({
        "kernel": args.kernel,
        "layer": args.layer,
        "policy": (
            "qkv-attention-ready-static-int8-per-head-no-gain"
            if attention_ready else "qkv-static-int8-per-head-no-gain"
        ),
        "output_count": len(outputs),
    })
    block = contract["encoder"][args.layer]
    if block["qkv"]["kernel"] != args.kernel:
        raise ValueError("runtime contract does not reference selected QKV kernel")
    output_scales = model_manifest["outputs"]
    if attention_ready:
        by_role_head = {
            (item["role"], int(item["head"])): item
            for item in output_scales
        }
        output_scales = [
            by_role_head[(role, head)]
            for role in ("q0", "k", "v", "q1")
            for head in range(6)
        ]
    block["qkv"].update({
        "output_layout": (
            "attention_ready_qkv" if attention_ready else "head_split_qkv"
        ),
        "output_precision": "INT8",
        "output_head_scales": output_scales,
        "affine_gain": 1.0,
    })
    contract["bank"].update({"bytes": len(bank), "sha256": bank_sha})

    bank_path.write_bytes(bank)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(args.cfg, args.output_dir / "cfg" / f"{args.kernel}_cfg.txt")
    source_report = args.source_package / "native_codec_report_active.json"
    if not source_report.exists():
        source_report = args.source_package / "native_codec_report.json"
    if args.probe_with_vendor_codec:
        (args.output_dir / "UNQUALIFIED_CODEC_PROBE.json").write_text(
            json.dumps({
                "kernel": args.kernel,
                "new_outputs": len(outputs),
                "required_layout_codec": "vendor until exact qualification",
            }, indent=2, sort_keys=True) + "\n"
        )
    else:
        validate_and_rebind_codec_report(
            source_report,
            args.source_package / "resident_kernel_bank_manifest.json",
            manifest_path,
            args.output_dir / "cfg",
            args.output_dir / "native_codec_report_active.json",
        )
    print(json.dumps({
        "bank_sha256": bank_sha,
        "image_bytes": len(payload),
        "isa_ranges": metadata["isa_ranges"],
        "kernel": args.kernel,
        "output_dir": str(args.output_dir),
        "slot_bytes": end - begin,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
