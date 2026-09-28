#!/usr/bin/env python3
"""Prepare a fixed DA-2K sample set and matching FP32 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text())
    indexed = {
        item["sample_id"]: {**item, "split": split}
        for split, records in selection["splits"].items()
        for item in records
    }
    missing = [sample for sample in args.samples if sample not in indexed]
    if missing:
        parser.error(f"samples absent from selection manifest: {missing}")

    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    input_dir = args.output_dir / "inputs"
    fp32_dir = args.output_dir / "fp32"
    input_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    with torch.inference_mode():
        for sample in args.samples:
            record = indexed[sample]
            source = args.dataset_root / record["path"]
            value = preprocess(source, args.input_size)
            prediction = model(torch.from_numpy(value).to(device)).float().cpu().numpy()
            input_path = input_dir / f"{sample}.npy"
            fp32_path = fp32_dir / f"{sample}.npy"
            np.save(input_path, value, allow_pickle=False)
            np.save(fp32_path, prediction, allow_pickle=False)
            evidence.append({
                "sample_id": sample,
                "split": record["split"],
                "scene": record["scene"],
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "input": str(input_path.resolve()),
                "input_sha256": sha256(input_path),
                "fp32": str(fp32_path.resolve()),
                "fp32_sha256": sha256(fp32_path),
            })
            print(json.dumps({"sample": sample, "shape": list(prediction.shape)}),
                  flush=True)

    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema": "depthanything-da2k-fixed-comparison-v1",
        "selection_source": str(args.selection_manifest.resolve()),
        "input_size": args.input_size,
        "input_shape": [1, 3, args.input_size, args.input_size],
        "preprocessing": {
            "resize": "cv2.INTER_CUBIC square", "color": "BGR_to_RGB",
            "mean": MEAN.tolist(), "std": STD.tolist(),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": str(device),
        "samples": evidence,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
