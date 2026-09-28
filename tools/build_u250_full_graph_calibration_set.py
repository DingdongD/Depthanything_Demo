#!/usr/bin/env python3
"""Build one immutable NYU + DA-2K calibration set for every input shape.

The source sample identity and train/validation split are shape independent.
Every calibration stage consumes the manifest fingerprint instead of selecting
its own files.  This prevents the historical per-layer/teacher-forced split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


SCHEMA = "depthanything-u250-full-graph-calibration-set-v1"
STAGES = (
    "1_norm1_to_qkv_input_a8",
    "2_per_head_qkv_a8",
    "3_softmax_dual_range_threshold",
    "4_dual_av_merge_boundary",
    "5_post_fc2_decoder_input_a8",
    "6_full_graph_final_depth_gate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_shapes(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not result or any(item <= 0 or item % 14 for item in result):
        raise argparse.ArgumentTypeError(
            "shapes must be positive multiples of the ViT patch size 14"
        )
    return result


def stable_order(paths: list[Path], seed: str, root: Path) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(root).as_posix()}".encode()
        ).digest(),
    )


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if count > len(paths):
        raise ValueError(f"requested {count} samples from only {len(paths)} files")
    if count == len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def split_stratum(paths: list[Path], validation_fraction: float,
                  seed: str, root: Path) -> dict[Path, str]:
    ordered = stable_order(paths, seed, root)
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation = set(ordered[:validation_count])
    return {path: "validation" if path in validation else "training"
            for path in paths}


def normalized_rgb(image: np.ndarray, shape: int) -> np.ndarray:
    resized = cv2.resize(image, (shape, shape), interpolation=cv2.INTER_CUBIC)
    value = resized.astype(np.float32) / np.float32(255.0)
    value = (value - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def load_source(path: Path, domain: str) -> np.ndarray:
    if domain == "nyu":
        with h5py.File(path, "r") as source:
            rgb = np.asarray(source["rgb"][:])
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"unexpected NYU RGB shape {rgb.shape}: {path}")
        return np.ascontiguousarray(rgb.transpose(1, 2, 0))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyu-root", type=Path, required=True)
    parser.add_argument("--da2k-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nyu-count", type=int, default=128)
    parser.add_argument("--da2k-count", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--shapes", type=parse_shapes, default=(280, 518))
    parser.add_argument("--seed", default="depthanything-u250-full-graph-v1")
    args = parser.parse_args()
    if args.nyu_count < 2 or args.da2k_count < 2:
        parser.error("each domain needs at least two samples")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be within (0, 0.5)")

    nyu_all = sorted(args.nyu_root.glob("*.h5"))
    nyu = select_evenly(nyu_all, args.nyu_count)
    nyu_split = split_stratum(
        nyu, args.validation_fraction, args.seed + ":nyu", args.nyu_root
    )

    scenes = {
        directory.name: sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )
        for directory in sorted(args.da2k_root.iterdir()) if directory.is_dir()
    }
    scenes = {name: paths for name, paths in scenes.items() if paths}
    da2k_all = [path for paths in scenes.values() for path in paths]
    if args.da2k_count > len(da2k_all):
        raise ValueError(
            f"requested {args.da2k_count} DA-2K samples from {len(da2k_all)} files"
        )
    # Round-robin across scenes keeps the small calibration set domain balanced.
    ordered_scenes = {
        name: stable_order(paths, args.seed + ":" + name, args.da2k_root)
        for name, paths in scenes.items()
    }
    da2k = []
    offset = 0
    while len(da2k) < args.da2k_count:
        progressed = False
        for name in sorted(ordered_scenes):
            paths = ordered_scenes[name]
            if offset < len(paths) and len(da2k) < args.da2k_count:
                da2k.append(paths[offset])
                progressed = True
        if not progressed:
            break
        offset += 1
    da2k_split = {}
    for name, scene_paths in scenes.items():
        selected = [path for path in da2k if path in set(scene_paths)]
        if selected:
            da2k_split.update(split_stratum(
                selected, args.validation_fraction,
                args.seed + ":da2k:" + name, args.da2k_root,
            ))

    selected = [("nyu", None, path, nyu_split[path]) for path in nyu]
    selected += [
        ("da2k", path.parent.name, path, da2k_split[path]) for path in da2k
    ]
    records = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for position, (domain, scene, source, split) in enumerate(selected, 1):
        relative = source.relative_to(
            args.nyu_root if domain == "nyu" else args.da2k_root
        )
        sample_id = (
            f"nyu/{source.stem}" if domain == "nyu"
            else f"da2k/{scene}/{source.stem}"
        )
        rgb = load_source(source, domain)
        tensors = {}
        for shape in args.shapes:
            output = args.output_root / "inputs" / str(shape) / Path(sample_id)
            output = output.with_suffix(".npy")
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, normalized_rgb(rgb, shape), allow_pickle=False)
            tensors[str(shape)] = {
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "shape": [1, 3, shape, shape],
            }
        record = {
            "sample_id": sample_id,
            "domain": domain,
            "scene": scene,
            "split": split,
            "source": str(source.resolve()),
            "source_relative": relative.as_posix(),
            "source_sha256": sha256_file(source),
            "tensors": tensors,
        }
        records.append(record)
        print(json.dumps({
            "prepared": position, "total": len(selected),
            "sample_id": sample_id, "split": split,
        }), flush=True)

    records.sort(key=lambda item: item["sample_id"])
    ids = {
        split: [item["sample_id"] for item in records if item["split"] == split]
        for split in ("training", "validation")
    }
    core = {
        "schema": SCHEMA,
        "seed": args.seed,
        "selection": {
            "nyu": "all/evenly-spaced from the dedicated NYU train pool",
            "da2k": "scene-stratified deterministic round-robin",
            "validation_fraction": args.validation_fraction,
        },
        "preprocessing": {
            "resize": "OpenCV INTER_CUBIC direct square resize",
            "rgb_range": "[0,1]",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "shapes": list(args.shapes),
        "sample_ids": ids,
        "samples": records,
        "calibration_policy": {
            "mode": "single-pass-full-graph-joint",
            "teacher_forcing": False,
            "sequential_layer_freeze": False,
            "av_output_gain": 1.0,
            "stages": [
                {"name": stage, "sample_ids": ids} for stage in STAGES
            ],
        },
    }
    core["manifest_sha256"] = canonical_sha256(core)
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(output.resolve()),
        "manifest_sha256": core["manifest_sha256"],
        "samples": len(records),
        "training": len(ids["training"]),
        "validation": len(ids["validation"]),
        "shapes": list(args.shapes),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
