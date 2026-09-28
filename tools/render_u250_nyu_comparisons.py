#!/usr/bin/env python3
"""Render NYU RGB/GT/FP32/U250 metric-depth comparison panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


ARIAL_FONT = Path("/root/demo/fonts/arial.ttf")
if not ARIAL_FONT.is_file():
    raise FileNotFoundError(f"required Arial font is missing: {ARIAL_FONT}")
font_manager.fontManager.addfont(ARIAL_FONT)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).squeeze()


def metric_depth(
    prediction: np.ndarray,
    scale: float,
    shift: float,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    inverse = np.clip(prediction * scale + shift, 1.0 / maximum, 1.0 / minimum)
    return 1.0 / inverse


def read_summary(path: Path) -> dict:
    prefix = "HYBRID_SUMMARY="
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise ValueError(f"missing HYBRID_SUMMARY in {path}")


def select_quantiles(samples: list[dict], count: int) -> list[tuple[str, dict]]:
    ordered = sorted(samples, key=lambda item: item["u250_vs_fp32_raw_rel_l2"])
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    result = []
    for position, index in enumerate(indices):
        if position == 0:
            label = "best raw alignment"
        elif position == len(indices) - 1:
            label = "worst raw alignment"
        else:
            percentile = int(round(100.0 * index / max(len(ordered) - 1, 1)))
            label = f"raw-error p{percentile}"
        result.append((label, ordered[index]))
    return result


def render_row(axes, item: dict) -> None:
    rgb = item["rgb"]
    gt = item["gt"]
    fp32 = item["fp32_depth"]
    board = item["board_depth"]
    valid = item["valid"]
    signed_delta = np.where(valid, board - fp32, np.nan)
    absolute_error = np.where(valid, np.abs(board - gt), np.nan)
    depth_lo, depth_hi = np.percentile(gt[valid], [1.0, 99.0])
    delta_hi = max(float(np.nanpercentile(np.abs(signed_delta), 99.0)), 1.0e-3)
    error_hi = max(float(np.nanpercentile(absolute_error, 99.0)), 1.0e-3)

    axes[0].imshow(rgb)
    axes[0].set_title(
        f"{item['label']}\n{item['sample']}\n"
        f"wall {item['wall_ms']:.0f} ms / NPU {item['npu_ms']:.0f} ms",
        loc="left", fontsize=8, fontweight="bold",
    )
    depth_image = axes[1].imshow(gt, cmap="turbo", vmin=depth_lo, vmax=depth_hi)
    axes[1].set_title(f"NYU GT depth\n{depth_lo:.2f}-{depth_hi:.2f} m", fontsize=8)
    axes[2].imshow(fp32, cmap="turbo", vmin=depth_lo, vmax=depth_hi)
    axes[2].set_title(
        f"FP32 metric depth\nRMSE {item['fp32_rmse']:.3f} m", fontsize=8
    )
    axes[3].imshow(board, cmap="turbo", vmin=depth_lo, vmax=depth_hi)
    axes[3].set_title(
        f"U250 r113, FP32 affine\nRMSE {item['board_rmse']:.3f} m", fontsize=8
    )
    delta_image = axes[4].imshow(
        signed_delta, cmap="coolwarm", vmin=-delta_hi, vmax=delta_hi
    )
    axes[4].set_title(
        f"U250 - FP32 depth\nraw rel-L2 {item['raw_rel_l2']:.3f}", fontsize=8
    )
    error_image = axes[5].imshow(
        absolute_error, cmap="magma", vmin=0.0, vmax=error_hi
    )
    axes[5].set_title(f"U250 |error| vs GT\np99 {error_hi:.3f} m", fontsize=8)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure = axes[0].figure
    figure.colorbar(depth_image, ax=list(axes[1:4]), shrink=0.72, label="depth (m)")
    figure.colorbar(delta_image, ax=axes[4], shrink=0.72, label="signed delta (m)")
    figure.colorbar(error_image, ax=axes[5], shrink=0.72, label="absolute error (m)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")

    manifest = json.loads(args.manifest.read_text())
    metrics_report = json.loads(args.metrics.read_text())
    records = {item["sample"]: item for item in manifest["samples"]}
    selected = select_quantiles(metrics_report["samples"], args.count)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for label, sample_metrics in selected:
        sample = sample_metrics["sample"]
        name = sample.split("/", 1)[1]
        record = records[sample]
        with h5py.File(record["source_h5"], "r") as source:
            rgb = source["rgb"][:].transpose(1, 2, 0)
            gt = source["depth"][:].astype(np.float64)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        fp32_path = args.fp32_root / f"{name}.npy"
        board_path = args.board_root / f"{name}.npz"
        fp32_raw = cv2.resize(
            load_prediction(fp32_path),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        board_raw = cv2.resize(
            load_prediction(board_path),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        affine = sample_metrics["fp32_inverse_affine"]
        fp32_depth = metric_depth(
            fp32_raw, affine["scale"], affine["shift"],
            args.min_depth, args.max_depth,
        )
        board_depth = metric_depth(
            board_raw, affine["scale"], affine["shift"],
            args.min_depth, args.max_depth,
        )
        valid = (
            np.isfinite(gt)
            & (gt > args.min_depth)
            & (gt <= args.max_depth)
        )
        summary = read_summary(board_path.with_suffix(".log"))
        item = {
            "label": label,
            "sample": sample,
            "rgb": rgb,
            "gt": np.where(valid, gt, np.nan),
            "fp32_depth": np.where(valid, fp32_depth, np.nan),
            "board_depth": np.where(valid, board_depth, np.nan),
            "valid": valid,
            "fp32_rmse": sample_metrics["metrics"]["fp32_independent_alignment"]["rmse"],
            "board_rmse": sample_metrics["metrics"]["u250_fp32_alignment"]["rmse"],
            "raw_rel_l2": sample_metrics["u250_vs_fp32_raw_rel_l2"],
            "raw_cosine": sample_metrics["u250_vs_fp32_raw_cosine"],
            "wall_ms": float(summary["wall_ms"]),
            "npu_ms": float(summary["npu_ms_total"]),
            "source_h5": record["source_h5"],
            "fp32_path": str(fp32_path.resolve()),
            "board_path": str(board_path.resolve()),
        }
        rows.append(item)

        figure, axes = plt.subplots(1, 6, figsize=(18, 3.2), constrained_layout=True)
        render_row(axes, item)
        figure.savefig(individual_dir / f"{name}_comparison.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(
        len(rows), 6, figsize=(18, 3.0 * len(rows)), constrained_layout=True
    )
    for row_axes, item in zip(axes, rows):
        render_row(row_axes, item)
    figure.suptitle(
        "DepthAnything V2 518x518 - U250 r113 board output vs FP32 and NYU GT",
        fontsize=14,
    )
    overview = args.output_dir / "nyu6_r113_board_vs_fp32_overview.png"
    figure.savefig(overview, dpi=170)
    plt.close(figure)

    evidence = {
        "schema": "depthanything-u250-r113-nyu-visualization-v1",
        "selection": "six evenly spaced quantiles of per-sample raw U250-vs-FP32 relative-L2",
        "metric_alignment": "FP32 per-image inverse-depth affine reused for U250",
        "depth_colormap": "turbo",
        "signed_delta_colormap": "coolwarm",
        "absolute_error_colormap": "magma",
        "colorbars": ["depth (m)", "signed delta (m)", "absolute error (m)"],
        "overview": str(overview.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "metrics_sha256": sha256(args.metrics),
        "font_actual": "Arial",
        "font_file": str(ARIAL_FONT),
        "font_sha256": sha256(ARIAL_FONT),
        "samples": [],
    }
    for item in rows:
        name = item["sample"].split("/", 1)[1]
        evidence["samples"].append({
            "label": item["label"],
            "sample": item["sample"],
            "fp32_rmse": item["fp32_rmse"],
            "board_rmse": item["board_rmse"],
            "raw_rel_l2": item["raw_rel_l2"],
            "raw_cosine": item["raw_cosine"],
            "wall_ms": item["wall_ms"],
            "npu_ms": item["npu_ms"],
            "source_h5": item["source_h5"],
            "fp32_path": item["fp32_path"],
            "board_path": item["board_path"],
            "board_sha256": sha256(Path(item["board_path"])),
            "image": str((individual_dir / f"{name}_comparison.png").resolve()),
        })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "overview": str(overview),
        "samples": [item["sample"] for item in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
