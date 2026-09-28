#!/usr/bin/env python3
"""Create deterministic, scene-stratified DA-2K calibration splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def allocate(total: int, capacities: dict[str, int]) -> dict[str, int]:
    available = sum(capacities.values())
    if not 0 <= total <= available:
        raise ValueError(f"cannot allocate {total} samples from {available}")
    exact = {key: total * value / available for key, value in capacities.items()}
    result = {key: min(capacities[key], int(exact[key])) for key in capacities}
    remainder = total - sum(result.values())
    order = sorted(capacities, key=lambda key: (exact[key] - int(exact[key]), key),
                   reverse=True)
    while remainder:
        progressed = False
        for key in order:
            if result[key] < capacities[key]:
                result[key] += 1
                remainder -= 1
                progressed = True
                if not remainder:
                    break
        if not progressed:
            raise RuntimeError("stratified allocation exhausted all capacities")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--tuning-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    annotation_path = root / "annotations.json"
    annotation_bytes = annotation_path.read_bytes()
    annotations = json.loads(annotation_bytes)
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError("annotations.json must contain a non-empty object")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for relative, pairs in sorted(annotations.items()):
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != "images":
            raise ValueError(f"unexpected DA-2K image path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        grouped[parts[1]].append({
            "path": relative,
            "scene": parts[1],
            "pair_annotations": len(pairs),
        })

    rng = random.Random(args.seed)
    for values in grouped.values():
        rng.shuffle(values)
    capacities = {key: len(value) for key, value in grouped.items()}
    calibration_counts = allocate(args.calibration_count, capacities)
    remaining = {key: capacities[key] - calibration_counts[key] for key in grouped}
    tuning_counts = allocate(args.tuning_count, remaining)

    splits = {"calibration": [], "tuning": [], "holdout": []}
    for scene in sorted(grouped):
        values = grouped[scene]
        c_end = calibration_counts[scene]
        t_end = c_end + tuning_counts[scene]
        splits["calibration"].extend(values[:c_end])
        splits["tuning"].extend(values[c_end:t_end])
        splits["holdout"].extend(values[t_end:])
    for split, values in splits.items():
        values.sort(key=lambda item: (item["scene"], item["path"]))
        for index, item in enumerate(values):
            item["sample_id"] = f"{split}_{index:04d}"

    report = {
        "schema": "depthanything-da2k-calibration-manifest-v1",
        "dataset_root": str(root),
        "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "seed": args.seed,
        "preprocessing_contract": {
            "shape": [1, 3, 518, 518],
            "resize": "cv2.INTER_CUBIC square",
            "color": "BGR_to_RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "splits": splits,
        "counts": {
            split: {
                "total": len(values),
                "by_scene": dict(sorted(Counter(
                    item["scene"] for item in values).items())),
                "pair_annotations": sum(item["pair_annotations"] for item in values),
            }
            for split, values in splits.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
