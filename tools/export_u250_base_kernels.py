#!/usr/bin/env python3
"""Export every kernel family required by the base hybrid U250 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SHAPES = {
    280: {"tokens": 401, "patch_grid": 20},
    518: {"tokens": 1370, "patch_grid": 37},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shape", type=int, choices=sorted(SHAPES), required=True)
    parser.add_argument("--attention-profile", type=Path, required=True)
    parser.add_argument("--linear-profile", type=Path, required=True)
    parser.add_argument("--decoder-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-input-scale", type=float)
    parser.add_argument("--qkv-output-scale-manifest", type=Path)
    args = parser.parse_args()
    for path in (args.model, args.attention_profile, args.linear_profile,
                 args.decoder_profile):
        if not path.is_file():
            parser.error(f"required export input is missing: {path}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(f"output root must be empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    tools = Path(__file__).resolve().parent
    contract = SHAPES[args.shape]
    commands: list[list[str]] = []

    def invoke(script: str, *arguments: object) -> None:
        command = [sys.executable, str(tools / script), *map(str, arguments)]
        subprocess.run(command, check=True)
        commands.append(command)

    patch_args: list[object] = [
        "--model", args.model,
        "--output-dir", args.output_root / "patch",
        "--patch-grid", contract["patch_grid"],
    ]
    if args.patch_input_scale is not None:
        patch_args.extend(("--input-scale", args.patch_input_scale))
    invoke("export_u250_patch_projection_kernels.py", *patch_args)

    qkv_args: list[object] = [
        "--model", args.model,
        "--output-dir", args.output_root / "qkv",
        "--scale-profile", args.linear_profile,
        "--tokens", contract["tokens"],
    ]
    if args.qkv_output_scale_manifest:
        qkv_args.extend(("--output-scale-manifest",
                         args.qkv_output_scale_manifest))
    invoke("export_u250_qkv_projection_kernels.py", *qkv_args)

    invoke(
        "export_u250_attention_kernels.py",
        "--profile", args.attention_profile,
        "--output-dir", args.output_root / "attention6",
    )
    invoke(
        "export_u250_attention_2chunk_kernels.py",
        "--attention-manifest", args.output_root / "attention6/manifest.json",
        "--output-dir", args.output_root / "attention2",
    )
    invoke(
        "export_u250_encoder_tail_kernels.py",
        "--model", args.model,
        "--output-dir", args.output_root / "encoder_tail",
        "--scale-profile", args.linear_profile,
        "--tokens", contract["tokens"],
    )
    invoke(
        "export_u250_decoder_conv_kernels.py",
        "--model", args.model,
        "--scale-profile", args.decoder_profile,
        "--output-dir", args.output_root / "decoder",
    )

    manifests = {}
    for family in ("patch", "qkv", "attention6", "attention2",
                   "encoder_tail", "decoder"):
        path = args.output_root / family / "manifest.json"
        if not path.is_file():
            raise RuntimeError(f"{family} exporter did not create {path}")
        manifests[family] = {"path": str(path.resolve()), "sha256": sha256(path)}
    result = {
        "schema": "depthanything-u250-base-export-v1",
        "shape": args.shape,
        **contract,
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "profiles": {
            "attention": sha256(args.attention_profile),
            "linear": sha256(args.linear_profile),
            "decoder": sha256(args.decoder_profile),
        },
        "manifests": manifests,
        "commands": commands,
    }
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
