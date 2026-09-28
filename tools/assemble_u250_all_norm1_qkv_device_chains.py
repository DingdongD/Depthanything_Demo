#!/usr/bin/env python3
"""Assemble all 12 calibrated LayerNorm-to-folded-QKV device chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from assemble_u250_norm_qkv_device_chain import install, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--norm-package", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.base_package, args.output_dir)
    for name in (
        "depthanything_u250_resident_kernel_bank.bin",
        "resident_kernel_bank_manifest.json",
    ):
        shutil.copyfile(args.norm_package / name, args.output_dir / name)
    for cfg in (args.norm_package / "cfg").glob("*_cfg.txt"):
        shutil.copyfile(cfg, args.output_dir / "cfg" / cfg.name)

    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    contract_path = args.output_dir / "depthanything_u250_runtime_contract.json"
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    bank_path = args.output_dir / manifest["bank_file"]
    bank = bytearray(bank_path.read_bytes())
    replacements = []
    scales = {}
    for layer in range(12):
        suffix = f"l{layer:02d}"
        calibration = json.loads((
            args.calibration_root / suffix / "models/folded_qkv/manifest.json"
        ).read_text())
        scale = float(calibration["kernels"][0]["input_scale"])
        scales[str(layer)] = scale
        norm_name = f"encoder_norm1_{suffix}"
        norm_variant = f"{norm_name}_int8"
        qkv_name = f"qkv_projection_{suffix}"
        qkv_variant = f"folded_qkv_{suffix}_p99p99"
        norm_dir = args.compiled_root / norm_variant
        qkv_dir = args.compiled_root / qkv_variant
        norm_cfg = norm_dir / f"{norm_variant}_cfg.txt"
        qkv_cfg = qkv_dir / f"{qkv_variant}_cfg.txt"
        install(manifest, bank, norm_name, norm_cfg,
                norm_dir / f"{norm_variant}_ddr.bin")
        install(manifest, bank, qkv_name, qkv_cfg,
                qkv_dir / f"{qkv_variant}_ddr.bin")
        shutil.copyfile(norm_cfg, args.output_dir / "cfg" / f"{norm_name}_cfg.txt")
        shutil.copyfile(qkv_cfg, args.output_dir / "cfg" / f"{qkv_name}_cfg.txt")
        block = contract["encoder"][layer]
        block["host_norm1"] = {
            "op": "LayerNormalization",
            "precision": "NPU_BF16_TO_INT8",
            "npu_core": norm_name,
            "device_chain_to": qkv_name,
            "affine": "folded_into_qkv_weights_and_biases",
            "output_quantization": {
                "dtype": "INT8", "scale": scale, "symmetric": True,
            },
        }
        block["qkv"]["input_quantization"] = {
            "dtype": "INT8", "scale": scale, "symmetric": True,
            "producer": norm_name,
        }
        block["qkv"]["affine_gain"] = 1.0
        replacements.extend([
            {"kernel": norm_name, "policy": "BF16-to-INT8-device-chain",
             "scale": scale},
            {"kernel": qkv_name, "policy": "affine-folded-no-gain",
             "scale": scale, "affine_gain": 1.0},
        ])

    bank_sha = sha256(bank)
    manifest["bank_sha256"] = bank_sha
    manifest.setdefault("kernel_replacements", []).extend(replacements)
    contract["bank"].update({"bytes": len(bank), "sha256": bank_sha})
    bank_path.write_bytes(bank)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "UNQUALIFIED_CODEC_PROBE.json").write_text(json.dumps({
        "layers": 12,
        "required_scales": scales,
        "required_layout_codec": "native after exact qualification",
        "strategy": "LayerNorm INT8 output directly connected to folded QKV input",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bank_sha256": bank_sha, "layers": 12,
                      "output": str(args.output_dir), "scales": scales},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
