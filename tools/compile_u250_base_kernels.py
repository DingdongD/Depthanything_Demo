#!/usr/bin/env python3
"""Compile all exported base-kernel families with the configured DS compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


FAMILIES = (
    ("patch", "compile_u250_patch_projection_kernels.sh"),
    ("qkv", "compile_u250_qkv_projection_kernels.sh"),
    ("attention2", "compile_u250_attention_2chunk_kernels.sh"),
    ("encoder_tail", "compile_u250_encoder_tail_kernels.sh"),
    ("decoder", "compile_u250_decoder_conv_kernels.sh"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    export_manifest = args.export_root / "manifest.json"
    if not export_manifest.is_file():
        parser.error(f"base export manifest is missing: {export_manifest}")
    tools = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    commands = []
    families = {}
    for family, script in FAMILIES:
        source = args.export_root / family
        output = args.output_root / family
        command = ["bash", str(tools / script), str(source), str(output),
                   str(args.jobs)]
        subprocess.run(command, check=True)
        commands.append(command)
        cfgs = sorted(output.rglob("*_cfg.txt"))
        if not cfgs:
            raise RuntimeError(f"{family}: compiler produced no CFG files")
        records = []
        for cfg in cfgs:
            stem = cfg.name.removesuffix("_cfg.txt")
            binary = cfg.with_name(stem + "_ddr.bin")
            if not binary.is_file() or not binary.stat().st_size:
                raise RuntimeError(f"{family}: missing or empty {binary}")
            records.append({
                "name": stem,
                "cfg": str(cfg.resolve()),
                "cfg_sha256": sha256(cfg),
                "binary": str(binary.resolve()),
                "binary_sha256": sha256(binary),
                "binary_bytes": binary.stat().st_size,
            })
        families[family] = records
    result = {
        "schema": "depthanything-u250-base-compiled-v1",
        "export_manifest": str(export_manifest.resolve()),
        "export_manifest_sha256": sha256(export_manifest),
        "commands": commands,
        "families": families,
        "kernel_count": sum(map(len, families.values())),
    }
    path = args.output_root / "manifest.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
