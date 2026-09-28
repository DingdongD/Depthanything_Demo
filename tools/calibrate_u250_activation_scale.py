#!/usr/bin/env python3
"""Search a symmetric A8 activation scale on captured runtime tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(paths: list[Path], key: str, scale: float) -> dict[str, float]:
    sse = reference_sq = 0.0
    clipped = elements = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            value = np.asarray(archive[key], dtype=np.float32)
        code = np.clip(np.rint(value / scale), -128, 127)
        restored = code * np.float32(scale)
        delta = restored.astype(np.float64) - value.astype(np.float64)
        sse += float(np.sum(delta * delta))
        reference = value.astype(np.float64)
        reference_sq += float(np.sum(reference * reference))
        clipped += int(np.count_nonzero((value < -128 * scale) | (value > 127 * scale)))
        elements += int(value.size)
    return {
        "relative_l2": float(np.sqrt(sse / max(reference_sq, 1e-30))),
        "clipping_fraction": float(clipped / max(elements, 1)),
        "elements": elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--current-scale", type=float, required=True)
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.trace_dir.glob("*/*.npz")) or sorted(
        args.trace_dir.glob("*.npz")
    )
    if len(paths) <= args.train_count:
        raise ValueError("need at least one held-out trace")
    scales = args.scales or np.geomspace(
        args.current_scale * 0.5, args.current_scale * 2.0, 33
    ).tolist()
    scales = sorted({float(value) for value in scales} | {args.current_scale})
    if any(value <= 0 for value in scales):
        raise ValueError("scales must be positive")
    splits = {"training": paths[:args.train_count], "validation": paths[args.train_count:]}
    candidates = []
    for scale in scales:
        candidates.append({
            "scale": scale,
            **{name: evaluate(split, args.key, scale) for name, split in splits.items()},
        })
    selected = min(candidates, key=lambda item: item["training"]["relative_l2"])
    report = {
        "schema": "depthanything-u250-activation-scale-calibration-v1",
        "trace_dir": str(args.trace_dir.resolve()),
        "key": args.key,
        "current_scale": args.current_scale,
        "training_samples": [str(path.relative_to(args.trace_dir)) for path in splits["training"]],
        "validation_samples": [str(path.relative_to(args.trace_dir)) for path in splits["validation"]],
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
