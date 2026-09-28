#!/usr/bin/env python3
"""Render evidence-backed U250/FP32 depth comparison panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float32).squeeze()


def load_rgb(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32).squeeze(0)
    value = value.transpose(1, 2, 0) * STD + MEAN
    return np.clip(value, 0.0, 1.0)


def metrics(board: np.ndarray, teacher: np.ndarray) -> tuple[dict, np.ndarray]:
    x = board.astype(np.float64).ravel()
    y = teacher.astype(np.float64).ravel()
    design = np.stack((x, np.ones_like(x)), axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned = board * np.float32(scale) + np.float32(shift)
    denominator = np.linalg.norm(y)
    result = {
        "rel_l2": float(np.linalg.norm(x - y) / denominator),
        "mae": float(np.mean(np.abs(x - y))),
        "cosine": float(np.dot(x, y) / (np.linalg.norm(x) * denominator)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "affine_rel_l2": float(
            np.linalg.norm(aligned.astype(np.float64).ravel() - y) / denominator
        ),
        "affine_scale": float(scale),
        "affine_shift": float(shift),
    }
    return result, aligned


def render_row(axes, rgb, teacher, board, aligned, title, values):
    joint = np.concatenate((teacher.ravel(), board.ravel()))
    depth_lo, depth_hi = np.percentile(joint, [1.0, 99.0])
    raw_error = np.abs(board - teacher)
    aligned_error = np.abs(aligned - teacher)
    raw_hi = max(float(np.percentile(raw_error, 99.0)), 1.0e-6)
    aligned_hi = max(float(np.percentile(aligned_error, 99.0)), 1.0e-6)

    axes[0].imshow(rgb)
    axes[0].set_title(title, loc="left", fontsize=9, fontweight="bold")
    axes[1].imshow(teacher, cmap="turbo", vmin=depth_lo, vmax=depth_hi)
    axes[1].set_title(f"FP32 teacher\nshared range [{depth_lo:.2f}, {depth_hi:.2f}]", fontsize=8)
    axes[2].imshow(board, cmap="turbo", vmin=depth_lo, vmax=depth_hi)
    axes[2].set_title(
        f"U250 r80\nrel-L2 {values['rel_l2']:.3f}, cos {values['cosine']:.3f}",
        fontsize=8,
    )
    axes[3].imshow(raw_error, cmap="magma", vmin=0.0, vmax=raw_hi)
    axes[3].set_title(f"Raw |error|\nMAE {values['mae']:.3f}, p99 {raw_hi:.3f}", fontsize=8)
    axes[4].imshow(aligned_error, cmap="magma", vmin=0.0, vmax=aligned_hi)
    axes[4].set_title(
        f"Affine-aligned |error|\nrel-L2 {values['affine_rel_l2']:.3f}, p99 {aligned_hi:.3f}",
        fontsize=8,
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("tuning", "holdout"), default="holdout")
    args = parser.parse_args()

    manifest_path = args.artifact_dir / "board_subset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["splits"][args.split]
    # The manifest is scene-stratified and contains two samples per scene.
    selected = []
    seen = set()
    for record in records:
        if record["scene"] not in seen:
            selected.append(record)
            seen.add(record["scene"])

    input_root = args.artifact_dir / "board_subset_32" / "inputs" / args.split
    teacher_root = args.artifact_dir / "board_subset_32" / "teacher_depth" / args.split
    board_root = args.artifact_dir / "finalconv_selected_s0p350000000" / args.split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in selected:
        sample_id = record["sample_id"]
        input_path = input_root / f"{sample_id}.npy"
        teacher_path = teacher_root / f"{sample_id}.npy"
        board_path = board_root / f"{sample_id}.npz"
        rgb = load_rgb(input_path)
        teacher = load_depth(teacher_path)
        board = load_depth(board_path)
        if teacher.shape != board.shape:
            raise ValueError(f"shape mismatch for {sample_id}: {teacher.shape} vs {board.shape}")
        values, aligned = metrics(board, teacher)
        title = f"{record['scene']} / {sample_id}"
        rows.append((rgb, teacher, board, aligned, title, values))

        figure, axes = plt.subplots(1, 5, figsize=(16, 3.25), constrained_layout=True)
        render_row(axes, rgb, teacher, board, aligned, title, values)
        figure.savefig(individual_dir / f"{sample_id}_comparison.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(
        len(rows), 5, figsize=(16, 3.15 * len(rows)), constrained_layout=True
    )
    for row_axes, row in zip(axes, rows):
        render_row(row_axes, *row)
    figure.suptitle(
        "DepthAnything V2 — U250 r80 vs FP32 teacher (one holdout image per DA-2K scene)",
        fontsize=14,
    )
    montage_path = args.output_dir / f"r80_{args.split}_8scene_comparison.png"
    figure.savefig(montage_path, dpi=180)
    plt.close(figure)

    evidence = {
        "schema": "depthanything-u250-r80-visual-comparison-v1",
        "version": "r80_da2k_finalconv_s0p350000000",
        "split": args.split,
        "display": {
            "depth_range": "per-sample shared FP32/U250 p1-p99",
            "error_range": "per-map p99",
            "input": "inverse ImageNet normalization of stored board input",
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "montage": str(montage_path),
        "samples": [],
    }
    for record, row in zip(selected, rows):
        sample_id = record["sample_id"]
        input_path = input_root / f"{sample_id}.npy"
        teacher_path = teacher_root / f"{sample_id}.npy"
        board_path = board_root / f"{sample_id}.npz"
        evidence["samples"].append(
            {
                "sample_id": sample_id,
                "scene": record["scene"],
                "source_path": record["path"],
                "metrics": row[-1],
                "input_sha256": sha256(input_path),
                "teacher_sha256": sha256(teacher_path),
                "board_sha256": sha256(board_path),
                "visualization": str(individual_dir / f"{sample_id}_comparison.png"),
            }
        )
    (args.output_dir / "visualization_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
