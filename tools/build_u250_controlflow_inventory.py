#!/usr/bin/env python3
"""Build the independently pinned input inventory for a CPU control-flow gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(package: Path, runtime: Path, collection: str) -> dict:
    manifest_path = package / "resident_kernel_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    case_names = {record["name"] for record in manifest["cases"]}
    cfg = {name + "_cfg.txt": package / "cfg" / (name + "_cfg.txt")
           for name in case_names}
    missing = [str(path) for path in cfg.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing active cfg files: {missing[:3]}")
    inputs = {
        "manifest": manifest_path,
        "contract": package / "depthanything_u250_runtime_contract.json",
        "host-plan": package / "depthanything_u250_host_plan.json",
        "host-params": package / "depthanything_u250_host_params.npz",
        "input": package / "demo05.npy",
        "layout-codec-report": package / "artifacts/u250_native_codec/all_oracle.json",
        "host-executor-report": (
            package / "artifacts/u250_encoder_residency_r73"
            / "host_executor_qualification.json"
        ),
    }
    runtime_inputs = {
        "architecture-16": ("runtime-dir", runtime / "arch_16_mono.yaml"),
        "architecture-256": ("runtime-dir", runtime / "arch_256_mono.yaml"),
        "vendor-codec": (
            "runtime-dir", runtime / "npz2bin.cpython-313-x86_64-linux-gnu.so"
        ),
        "vendor-tensor-helper": ("case-dir", package / "npz_util.py"),
    }
    bank = package / manifest["bank_file"]
    helper = package / "run_u250_resident_compiled_case.py"
    return {
        "schema_version": 5,
        "collection": collection,
        "cfg_sha256": {name: sha256(path) for name, path in sorted(cfg.items())},
        "helper": {"name": helper.name, "sha256": sha256(helper)},
        "inputs": {
            role: {"name": path.name, "sha256": sha256(path)}
            for role, path in inputs.items()
        },
        "resident_bank": {"name": bank.name, "sha256": sha256(bank)},
        "runtime_inputs": {
            role: {"base": base, "name": path.name, "sha256": sha256(path)}
            for role, (base, path) in runtime_inputs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(args.package, args.runtime_dir, args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "cfg": len(result["cfg_sha256"]),
        "resident_bank_sha256": result["resident_bank"]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
