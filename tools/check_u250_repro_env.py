#!/usr/bin/env python3
"""Validate all external inputs needed for a reproducible U250 build."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REPO_FILES = (
    "ds_models/depth_anything_v2_vits.py",
    "ds_models/depth_anything_v2_vits_280.py",
    "ds_models/static_int8_attention.py",
    "tools/export_depth_anything_v2_ds.py",
    "tools/quantize_depth_anything_v2_ds.py",
    "tools/export_u250_patch_projection_kernels.py",
    "tools/export_u250_qkv_projection_kernels.py",
    "tools/export_u250_attention_2chunk_kernels.py",
    "tools/export_u250_encoder_tail_kernels.py",
    "tools/export_u250_decoder_conv_kernels.py",
    "tools/export_u250_base_kernels.py",
    "tools/compile_u250_base_kernels.py",
    "tools/package_u250_base.py",
    "tools/link_u250_resident_kernel_bank.py",
    "tools/generate_u250_runtime_contract.py",
    "tools/export_u250_host_plan.py",
    "tools/prepare_da2k_calibration_manifest.py",
    "tools/prepare_mixed_calibration_inputs.py",
    "tools/build_u250_full_graph_calibration_set.py",
    "tools/collect_fp32_replacement_traces.py",
    "tools/calibrate_u250_full_graph_joint.py",
    "tools/export_u250_full_graph_joint_candidate.py",
    "tools/compile_u250_full_graph_joint.py",
    "tools/assemble_u250_full_graph_joint_package.py",
    "tools/npz_util.py",
    "tools/run_u250_depthanything_hybrid.py",
)
REQUIRED_PACKAGES = (
    "numpy", "onnx", "onnxruntime", "torch", "torchvision", "cv2", "h5py",
    "matplotlib", "yaml", "pybind11",
)


def path_check(name: str, path: Path | None, kind: str) -> dict:
    resolved = path.expanduser().resolve() if path is not None else None
    exists = bool(
        resolved is not None
        and ((kind == "file" and resolved.is_file())
             or (kind == "dir" and resolved.is_dir())
             or (kind == "path" and resolved.exists()))
    )
    return {
        "name": name,
        "kind": kind,
        "path": str(resolved) if resolved is not None else None,
        "ok": exists,
    }


def executable_check(name: str, value: str | Path | None) -> dict:
    configured = str(value) if value is not None else None
    resolved = shutil.which(configured) if configured else None
    if resolved is None and configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
    return {
        "name": name,
        "kind": "executable",
        "path": resolved or configured,
        "ok": resolved is not None,
    }


def runtime_extension_check(runtime_dir: Path | None) -> dict:
    resolved = runtime_dir.expanduser().resolve() if runtime_dir else None
    matches = (
        sorted(resolved.glob("npz2bin*.so"))
        if resolved and resolved.is_dir() else []
    )
    return {
        "name": "board:npz2bin_extension",
        "kind": "file-glob",
        "path": str(resolved / "npz2bin*.so") if resolved else None,
        "matches": [str(path) for path in matches],
        "ok": len(matches) == 1,
    }


def inspect_environment(args: argparse.Namespace) -> dict:
    toolchain = args.toolchain_root
    python_root = args.python_root
    extension = args.extension_dir
    if extension is None and python_root is not None:
        extension = python_root / "RelWithDebInfo/lib"

    checks = [
        path_check("checkpoint", args.checkpoint, "file"),
        path_check("toolchain_root", toolchain, "dir"),
        path_check(
            "compiler",
            args.compiler or (toolchain / "python_bin/compile.py" if toolchain else None),
            "file",
        ),
        path_check(
            "toolchain_python_libs",
            toolchain / "python_libs" if toolchain else None,
            "dir",
        ),
        path_check(
            "arch_16",
            args.arch_16 or (toolchain / "arch_16_mono.yaml" if toolchain else None),
            "file",
        ),
        path_check(
            "arch_256",
            args.arch_256 or (toolchain / "arch_256_mono.yaml" if toolchain else None),
            "file",
        ),
        path_check("python_root", python_root, "dir"),
        path_check("compiler_extension_dir", extension, "dir"),
        executable_check("compiler_python", args.compiler_python),
    ]
    checks.extend(
        path_check(f"repo:{relative}", REPO_ROOT / relative, "file")
        for relative in REQUIRED_REPO_FILES
    )
    packages = []
    if not args.skip_python_packages:
        packages = [
            {"name": name, "ok": importlib.util.find_spec(name) is not None}
            for name in REQUIRED_PACKAGES
        ]
    board = []
    if args.require_board:
        runtime_dir = args.runtime_dir
        board.extend((
            path_check("board:runtime_dir", runtime_dir, "dir"),
            path_check(
                "board:arch_16",
                runtime_dir / "arch_16_mono.yaml" if runtime_dir else None,
                "file",
            ),
            path_check(
                "board:arch_256",
                runtime_dir / "arch_256_mono.yaml" if runtime_dir else None,
                "file",
            ),
            runtime_extension_check(runtime_dir),
        ))
        for name in (
            "xdma0_user", "xdma0_h2c_0", "xdma0_h2c_1",
            "xdma0_c2h_0", "xdma0_c2h_1",
            *(f"xdma0_events_{index}" for index in range(16)),
        ):
            board.append(path_check(f"board:{name}", Path("/dev") / name, "path"))

    ok = all(item["ok"] for item in checks + packages + board)
    return {
        "schema": "depthanything-u250-repro-doctor-v1",
        "ok": ok,
        "python": sys.executable,
        "checks": checks,
        "python_packages": packages,
        "board_checks": board,
    }


def environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=environment_path("DEPTH_ANYTHING_CHECKPOINT"),
    )
    parser.add_argument(
        "--toolchain-root", type=Path,
        default=environment_path("DS_TOOLCHAIN_ROOT"),
    )
    parser.add_argument(
        "--compiler", type=Path, default=environment_path("DS_COMPILER")
    )
    parser.add_argument(
        "--arch-16", type=Path, default=environment_path("DS_ARCH_16")
    )
    parser.add_argument(
        "--arch-256", type=Path, default=environment_path("DS_ARCH_256")
    )
    parser.add_argument(
        "--python-root", type=Path, default=environment_path("PYTHON_ROOT")
    )
    parser.add_argument(
        "--extension-dir", type=Path,
        default=environment_path("ACOMPILER_EXTENSION_DIR"),
    )
    parser.add_argument(
        "--compiler-python",
        default=os.environ.get("DS_COMPILER_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=environment_path("U250_RUNTIME_DIR")
    )
    parser.add_argument("--require-board", action="store_true")
    parser.add_argument("--skip-python-packages", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_environment(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
