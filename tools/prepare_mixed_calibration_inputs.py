#!/usr/bin/env python3
"""Prepare a deterministic NYU + DA-2K square calibration input list."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced(values: list, count: int) -> list:
    if count < 1 or count > len(values):
        raise ValueError(f"cannot select {count} samples from {len(values)}")
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[int(index)] for index in indices]


def preprocess_image(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyu-input-root", type=Path, required=True)
    parser.add_argument("--nyu-count", type=int, default=16)
    parser.add_argument("--da2k-root", type=Path, required=True)
    parser.add_argument("--da2k-manifest", type=Path, required=True)
    parser.add_argument("--da2k-split", default="tuning")
    parser.add_argument("--da2k-count", type=int, default=16)
    parser.add_argument("--size", type=int, default=280)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    nyu_paths = evenly_spaced(sorted(args.nyu_input_root.glob("*.npy")), args.nyu_count)
    da2k_document = json.loads(args.da2k_manifest.read_text())
    if args.da2k_split not in da2k_document["splits"]:
        raise ValueError(f"missing DA-2K split {args.da2k_split!r}")
    da2k_records = evenly_spaced(
        da2k_document["splits"][args.da2k_split], args.da2k_count
    )

    samples = []
    expected_shape = (1, 3, args.size, args.size)
    for path in nyu_paths:
        value = np.load(path, allow_pickle=False)
        if value.shape != expected_shape:
            raise ValueError(f"unexpected NYU input shape {value.shape}: {path}")
        samples.append({
            "path": str(path.resolve()),
            "domain": "nyu",
            "sample_id": path.stem,
            "source_sha256": sha256(path),
        })

    da2k_output = args.output_dir / "inputs" / "da2k"
    da2k_output.mkdir(parents=True, exist_ok=True)
    for record in da2k_records:
        source = args.da2k_root / record["path"]
        target = da2k_output / f"{record['sample_id']}.npy"
        value = preprocess_image(source, args.size)
        np.save(target, value, allow_pickle=False)
        samples.append({
            "path": str(target.resolve()),
            "domain": "da2k",
            "sample_id": record["sample_id"],
            "scene": record["scene"],
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
        })

    report = {
        "schema": "depthanything-mixed-calibration-inputs-v1",
        "shape": list(expected_shape),
        "preprocessing": {
            "resize": "cv2.INTER_CUBIC square",
            "color": "BGR_to_RGB",
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        "selection": {
            "method": "deterministic evenly spaced",
            "nyu_count": args.nyu_count,
            "da2k_count": args.da2k_count,
            "da2k_split": args.da2k_split,
        },
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "calibration_inputs.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "samples": len(samples),
        "domains": {
            domain: sum(item["domain"] == domain for item in samples)
            for domain in ("nyu", "da2k")
        },
        "shape": list(expected_shape),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
