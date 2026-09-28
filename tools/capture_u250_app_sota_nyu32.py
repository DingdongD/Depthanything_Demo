#!/usr/bin/env python3
"""Capture the common NYU32 set with both App-selected U250 SOTA profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


PROFILES = (
    {
        "model_key": "depthanything_u250",
        "shape": 518,
        "input_root": Path(
            "artifacts/u250_accuracy_r110_nyu32/inputs/nyu32"
        ),
        "bank_sha256":
            "0b0149f1926c50dec2d55f12facc8f5870c5b12ef95566c5712157a4c63024b6",
    },
    {
        "model_key": "depthanything_u250_280",
        "shape": 280,
        "input_root": Path(
            "/root/demo/depthanything/Depth-Anything-V2/artifacts/"
            "u250_280_r1/nyu32/inputs/nyu32"
        ),
        "bank_sha256":
            "3148c1e03cd0f4b73d6e11c695a8d9805c5006fd32ac68cf40a427323452018a",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_adapter(app_dir: Path):
    helper = app_dir / "depthanything_u250.py"
    spec = importlib.util.spec_from_file_location("depthanything_u250_sota_capture", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import App adapter: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DepthAnythingU250Adapter


def capture_profile(adapter_type, spec: dict, output_root: Path) -> dict:
    model_key = spec["model_key"]
    shape = int(spec["shape"])
    inputs = sorted(spec["input_root"].resolve().glob("nyu_*.npy"))
    if len(inputs) != 32:
        raise RuntimeError(f"{model_key}: expected 32 inputs, found {len(inputs)}")

    board_root = output_root / str(shape) / "board" / "nyu32"
    board_root.mkdir(parents=True, exist_ok=False)
    adapter = adapter_type(model_key)
    bank_path = adapter.profile["remote_package"] + "/depthanything_u250_resident_kernel_bank.bin"
    actual_bank_sha = adapter._ssh(f"sha256sum {bank_path}", timeout=30).split()[0]
    if actual_bank_sha != spec["bank_sha256"]:
        raise RuntimeError(
            f"{model_key}: bank SHA256 {actual_bank_sha} != {spec['bank_sha256']}"
        )

    remote_root = adapter.profile["remote_stage"] + "/offline-sota-nyu32"
    records = []
    try:
        adapter._ensure_resident()
        first_payload = inputs[0].read_bytes()
        adapter._transfer("put", remote_root + "/frame.npy", first_payload)
        adapter._run_request(
            remote_root + "/frame.npy",
            remote_root + "/warmup.npz",
            f"{model_key}-nyu32-warmup",
        )
        adapter._resident_warmed = True

        for index, source in enumerate(inputs):
            payload = source.read_bytes()
            adapter._transfer("put", remote_root + "/frame.npy", payload)
            started = time.perf_counter()
            message = adapter._run_request(
                remote_root + "/frame.npy",
                remote_root + "/result.npz",
                f"{model_key}-nyu32-{index:02d}",
            )
            request_wall_ms = (time.perf_counter() - started) * 1000.0
            result_bytes = adapter._transfer("get", remote_root + "/result.npz")
            target = board_root / f"{source.stem}.npz"
            target.write_bytes(result_bytes)
            with np.load(target, allow_pickle=False) as archive:
                depth = np.asarray(archive["depth"])
            expected_shape = (1, shape, shape)
            if depth.shape != expected_shape or not np.isfinite(depth).all():
                raise RuntimeError(
                    f"{model_key} {source.stem}: invalid depth {depth.shape}, "
                    f"finite={np.isfinite(depth).all()}"
                )
            summary_bytes = adapter._transfer("get", message["summary"])
            (board_root / f"{source.stem}.summary.json").write_bytes(summary_bytes)
            record = {
                "sample": source.stem,
                "input_sha256": sha256_bytes(payload),
                "output_sha256": sha256_bytes(result_bytes),
                "runtime_wall_ms": message.get("runtime_wall_ms"),
                "request_wall_ms": request_wall_ms,
                "output_shape": list(depth.shape),
                "finite": True,
            }
            records.append(record)
            print(json.dumps({"model": model_key, "index": index, **record}), flush=True)
    finally:
        adapter.invalidate_resident()

    return {
        "model_key": model_key,
        "shape": shape,
        "runtime_label": adapter.profile["runtime_label"],
        "remote_base": adapter.profile["remote_base"],
        "remote_package": adapter.profile["remote_package"],
        "bank_sha256": actual_bank_sha,
        "sample_count": len(records),
        "samples": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("all", "518", "280"), default="all")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    adapter_type = load_adapter(args.app_dir.resolve())
    selected = PROFILES if args.profile == "all" else tuple(
        item for item in PROFILES if str(item["shape"]) == args.profile
    )
    reports = [capture_profile(adapter_type, item, output_root) for item in selected]
    manifest = {
        "schema": "depthanything-u250-app-sota-nyu32-v1",
        "sample_set": "common NYU32",
        "profiles": reports,
    }
    (output_root / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
