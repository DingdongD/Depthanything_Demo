#!/usr/bin/env python3
"""Assemble an experimental INT8 LayerNorm-to-folded-QKV device chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from replace_u250_fc1_pairs_in_bank import parse_cfg


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def install(manifest: dict, bank: bytearray, name: str, cfg: Path, binary: Path) -> None:
    ordered = sorted(manifest["cases"], key=lambda item: int(item["offset_bytes"]))
    index = next(index for index, item in enumerate(ordered) if item["name"] == name)
    old = ordered[index]
    begin = int(old["offset_bytes"])
    end = int(ordered[index + 1]["offset_bytes"]) if index + 1 < len(ordered) else len(bank)
    payload = binary.read_bytes()
    if not payload or len(payload) % 256 or len(payload) > end - begin:
        raise ValueError(f"{name}: image does not fit resident slot")
    metadata = parse_cfg(cfg)
    if metadata["isa_ranges"] != old["isa_ranges"]:
        raise ValueError(f"{name}: instruction ABI changed")
    bank[begin:end] = payload + bytes(end - begin - len(payload))
    relocation = begin // 256
    bases = [value + relocation for value in metadata["base_addresses"]]
    bases[4] = int(manifest["shared_fm_base_units"])
    replacement = {
        **old,
        "group": "norm-qkv-device-chain",
        "source_cfg": str(cfg.resolve()),
        "source_ddr": str(binary.resolve()),
        "size_bytes": len(payload),
        "sha256": sha256(payload),
        "base_addresses_local": metadata["base_addresses"],
        "base_addresses": bases,
        "inputs": metadata["inputs"],
        "outputs": metadata["outputs"],
        "isa_ranges": metadata["isa_ranges"],
    }
    manifest["cases"] = [replacement if item["name"] == name else item
                         for item in manifest["cases"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--norm-package", type=Path, required=True)
    parser.add_argument("--norm-cfg", type=Path, required=True)
    parser.add_argument("--norm-binary", type=Path, required=True)
    parser.add_argument("--qkv-cfg", type=Path, required=True)
    parser.add_argument("--qkv-binary", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
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
    install(manifest, bank, "encoder_norm1_l00", args.norm_cfg, args.norm_binary)
    install(manifest, bank, "qkv_projection_l00", args.qkv_cfg, args.qkv_binary)
    bank_sha = sha256(bank)
    manifest["bank_sha256"] = bank_sha
    manifest.setdefault("kernel_replacements", []).extend([
        {"kernel": "encoder_norm1_l00", "policy": "BF16-to-INT8-device-chain",
         "scale": args.scale},
        {"kernel": "qkv_projection_l00", "policy": "affine-folded-no-gain",
         "scale": args.scale, "affine_gain": 1.0},
    ])
    contract["bank"].update({"bytes": len(bank), "sha256": bank_sha})
    contract["encoder"][0]["host_norm1"] = {
        "op": "LayerNormalization",
        "precision": "NPU_BF16_TO_INT8",
        "npu_core": "encoder_norm1_l00",
        "device_chain_to": "qkv_projection_l00",
        "affine": "folded_into_qkv_weights_and_biases",
        "output_quantization": {
            "dtype": "INT8", "scale": args.scale, "symmetric": True,
        },
    }
    contract["encoder"][0]["qkv"]["input_quantization"] = {
        "dtype": "INT8", "scale": args.scale, "symmetric": True,
        "producer": "encoder_norm1_l00",
    }
    contract["encoder"][0]["qkv"]["affine_gain"] = 1.0
    bank_path.write_bytes(bank)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(args.norm_cfg, args.output_dir / "cfg/encoder_norm1_l00_cfg.txt")
    shutil.copyfile(args.qkv_cfg, args.output_dir / "cfg/qkv_projection_l00_cfg.txt")
    (args.output_dir / "UNQUALIFIED_CODEC_PROBE.json").write_text(json.dumps({
        "required_layout_codec": "native after exact qualification",
        "strategy": "LayerNorm INT8 output directly connected to folded QKV input",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bank_sha256": bank_sha, "output": str(args.output_dir)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
