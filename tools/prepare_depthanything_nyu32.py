#!/usr/bin/env python3
"""Prepare NYU RGB tensors and matching DepthAnything-V2 FP32 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(rgb_chw: np.ndarray, size: int) -> np.ndarray:
    image = rgb_chw.transpose(1, 2, 0)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    paths = sorted(args.dataset_root.glob("*.h5"))
    if len(paths) != 32:
        raise ValueError(f"expected 32 NYU samples, found {len(paths)}")
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    input_root = args.output_dir / "inputs" / "nyu32"
    teacher_root = args.output_dir / "fp32" / "nyu32"
    input_root.mkdir(parents=True, exist_ok=True)
    teacher_root.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for position, path in enumerate(paths, 1):
            with h5py.File(path, "r") as source:
                rgb = source["rgb"][:]
                depth_shape = list(source["depth"].shape)
            value = preprocess(rgb, args.input_size)
            prediction = model(torch.from_numpy(value).to(device)).float().cpu().numpy()
            sample = f"nyu_{path.stem}"
            input_path = input_root / f"{sample}.npy"
            teacher_path = teacher_root / f"{sample}.npy"
            np.save(input_path, value, allow_pickle=False)
            np.save(teacher_path, prediction, allow_pickle=False)
            records.append({
                "sample": f"nyu32/{sample}", "source_h5": str(path.resolve()),
                "gt_shape": depth_shape, "input_sha256": sha256(input_path),
                "fp32_sha256": sha256(teacher_path),
            })
            print(json.dumps({"sample": sample, "position": position,
                              "total": len(paths)}), flush=True)
    report = {
        "schema": "depthanything-nyu32-inputs-fp32-v1",
        "device": str(device), "input_size": args.input_size,
        "preprocessing": {
            "resize": "cv2.INTER_CUBIC square", "color": "RGB",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "samples": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
