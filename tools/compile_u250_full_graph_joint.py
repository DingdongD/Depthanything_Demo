#!/usr/bin/env python3
"""Compile every ONNX entry in a full-graph joint export manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--compiler", type=Path,
        default=Path("/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py"),
    )
    parser.add_argument(
        "--python", type=Path, default=Path("/opt/conda/bin/python")
    )
    parser.add_argument(
        "--arch-root", type=Path, default=Path("/root/demo/DS_Toolchain_Demo_full")
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    # The compiler runs with each kernel directory as cwd.  Resolve every path
    # used across that boundary so artifacts cannot be written into a nested
    # duplicate of the caller's relative output path.
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    args.compiler = args.compiler.resolve()
    args.python = args.python.resolve()
    args.arch_root = args.arch_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "depthanything-u250-full-graph-joint-export-v1":
        raise ValueError("unsupported joint export manifest")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    python_root = environment.setdefault(
        "PYTHON_ROOT", "/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320"
    )
    environment.setdefault(
        "ACOMPILER_EXTENSION_DIR", python_root + "/RelWithDebInfo/lib"
    )
    arch = ",".join(str(args.arch_root / name) for name in (
        "arch_16_mono.yaml", "arch_256_mono.yaml"
    ))

    def compile_one(record: dict) -> dict:
        name = record["kernel"]
        directory = args.output_root / name
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / name
        cfg = Path(str(prefix) + "_cfg.txt")
        binary = Path(str(prefix) + "_ddr.bin")
        log = directory / "compile.stdout.log"
        command = [
            str(args.python), str(args.compiler), "--model", record["onnx"],
            "--output_path", str(prefix), "--log_path", str(directory),
            "--arch_path", arch, "--layouts", record["layouts"],
            "--codegen", str(record["codegen"]), "--sim", "1", "--addr", "1",
            "--l2_size", "100", "--spill_threshold", "0",
        ]
        with log.open("w") as stream:
            completed = subprocess.run(
                command, cwd=directory, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
        if completed.returncode or not cfg.is_file() or not binary.is_file():
            raise RuntimeError(
                f"{name}: compiler failed with {completed.returncode}; see {log}"
            )
        if not cfg.stat().st_size or not binary.stat().st_size:
            raise RuntimeError(f"{name}: compiler produced an empty CFG/BIN")
        return {
            **record, "cfg": str(cfg.resolve()), "cfg_sha256": sha256(cfg),
            "binary": str(binary.resolve()), "binary_sha256": sha256(binary),
            "binary_bytes": binary.stat().st_size, "compile_log": str(log.resolve()),
        }

    compiled = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compile_one, item): item["kernel"]
                   for item in manifest["kernels"]}
        for future in as_completed(futures):
            result = future.result()
            compiled.append(result)
            print(json.dumps({
                "compiled": result["kernel"],
                "completed": len(compiled), "total": len(futures),
                "binary_bytes": result["binary_bytes"],
            }), flush=True)
    compiled.sort(key=lambda item: item["kernel"])
    result = {
        "schema": "depthanything-u250-full-graph-joint-compiled-v1",
        "calibration_manifest_sha256": manifest["calibration_manifest_sha256"],
        "export_manifest": str(args.manifest.resolve()),
        "kernels": compiled, "kernel_count": len(compiled),
    }
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output.resolve()),
                      "kernels": len(compiled)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
