#!/usr/bin/env python3
"""Link compiled kernels and generate a self-describing U250 runtime package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


FAMILIES = ("patch", "qkv", "attention2", "encoder_tail", "decoder")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-fm-workspace-bytes", type=int,
                        default=24 * 1024 * 1024)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output package already exists: {args.output_dir}")
    for path in (args.model, args.export_root / "manifest.json",
                 args.compiled_root / "manifest.json"):
        if not path.is_file():
            parser.error(f"required package input is missing: {path}")
    tools = Path(__file__).resolve().parent

    link = [
        sys.executable, str(tools / "link_u250_resident_kernel_bank.py"),
        "--recursive",
        "--shared-fm-workspace-bytes", str(args.shared_fm_workspace_bytes),
        "--output-dir", str(args.output_dir),
    ]
    for family in FAMILIES:
        directory = args.compiled_root / family
        if not directory.is_dir():
            parser.error(f"compiled family is missing: {directory}")
        link.extend(("--case-dir", str(directory)))
    subprocess.run(link, check=True)

    contract = args.output_dir / "depthanything_u250_runtime_contract.json"
    contract_command = [
        sys.executable, str(tools / "generate_u250_runtime_contract.py"),
        "--model", str(args.model),
        "--bank-manifest", str(args.output_dir / "resident_kernel_bank_manifest.json"),
        "--qkv-manifest", str(args.export_root / "qkv/manifest.json"),
        "--qkv-model-dir", str(args.export_root / "qkv"),
        "--attention-manifest", str(args.export_root / "attention2/manifest.json"),
        "--encoder-tail-manifest", str(args.export_root / "encoder_tail/manifest.json"),
        "--decoder-manifest", str(args.export_root / "decoder/manifest.json"),
        "--patch-manifest", str(args.export_root / "patch/manifest.json"),
        "--output", str(contract),
    ]
    subprocess.run(contract_command, check=True)

    plan = args.output_dir / "depthanything_u250_host_plan.json"
    params = args.output_dir / "depthanything_u250_host_params.npz"
    plan_command = [
        sys.executable, str(tools / "export_u250_host_plan.py"),
        "--model", str(args.model),
        "--decoder-manifest", str(args.export_root / "decoder/manifest.json"),
        "--patch-manifest", str(args.export_root / "patch/manifest.json"),
        "--frontend-model", str(args.model),
        "--plan", str(plan), "--params", str(params),
    ]
    subprocess.run(plan_command, check=True)

    # The runtime imports this helper from --case-dir before it imports the
    # vendor npz2bin extension from --runtime-dir.  Keep a clean package
    # self-contained instead of relying on a historical case directory.
    codec_helper = tools / "npz_util.py"
    packaged_codec_helper = args.output_dir / codec_helper.name
    shutil.copy2(codec_helper, packaged_codec_helper)

    required = [
        args.output_dir / "resident_kernel_bank_manifest.json",
        contract, plan, params, packaged_codec_helper,
    ]
    bank_manifest = json.loads(required[0].read_text())
    required.append(args.output_dir / bank_manifest["bank_file"])
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"package output is missing or empty: {path}")
    result = {
        "schema": "depthanything-u250-base-package-v1",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "export_manifest_sha256": sha256(args.export_root / "manifest.json"),
        "compiled_manifest_sha256": sha256(args.compiled_root / "manifest.json"),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required
        },
        "commands": [link, contract_command, plan_command],
        "deployment_status": "candidate_requires_codec_and_board_gate",
    }
    output = args.output_dir / "package_build_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
