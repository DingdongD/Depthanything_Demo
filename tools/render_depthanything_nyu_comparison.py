#!/usr/bin/env python3
"""Render NYU RGB/GT/FP32/U250 depth comparisons with a fair shared alignment."""

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
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).squeeze()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_inverse_affine(prediction: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray) -> tuple[float, float]:
    target = 1.0 / gt[mask]
    design = np.stack((prediction[mask], np.ones(mask.sum())), axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(scale), float(shift)


def metric_depth(prediction: np.ndarray, scale: float, shift: float,
                 minimum: float = 0.1, maximum: float = 10.0) -> np.ndarray:
    inverse = np.clip(prediction * scale + shift, 1.0 / maximum, 1.0 / minimum)
    return 1.0 / inverse


def rmse(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction[mask] - gt[mask]) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--reference-selection", type=Path)
    args = parser.parse_args()

    if args.reference_selection is not None:
        reference = json.loads(args.reference_selection.read_text())
        expected = [Path(item["sample"]).name for item in reference["samples"]]
        if args.samples != expected:
            parser.error(
                "--samples must exactly match the reference selection and order: "
                + " ".join(expected)
            )

    records = {
        record["sample"].split("/", 1)[1]: record
        for record in json.loads(args.manifest.read_text())["samples"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    evidence = []

    for sample in args.samples:
        record = records[sample]
        with h5py.File(record["source_h5"], "r") as source:
            gt = source["depth"][:].astype(np.float64)
            rgb = np.moveaxis(source["rgb"][:], 0, -1)

        fp32_raw = load_prediction(args.fp32_root / f"{sample}.npy")
        board_raw = load_prediction(args.board_root / f"{sample}.npz")
        raw_rmse = float(np.sqrt(np.mean((board_raw - fp32_raw) ** 2)))
        fp32 = cv2.resize(fp32_raw, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        board = cv2.resize(board_raw, (gt.shape[1], gt.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        mask = np.isfinite(gt) & (gt > 0.1) & (gt <= 10.0)

        # Fit FP32 once and apply exactly the same affine calibration to U250.
        # This preserves the hardware drift rather than hiding it with a second fit.
        scale, shift = fit_inverse_affine(fp32, gt, mask)
        fp32_metric = metric_depth(fp32, scale, shift)
        board_metric = metric_depth(board, scale, shift)
        signed_delta = board_metric - fp32_metric
        board_gt_error = np.abs(board_metric - gt)

        valid_gt = gt[mask]
        depth_min, depth_max = np.percentile(valid_gt, [1.0, 99.0])
        delta_max = max(float(np.percentile(np.abs(signed_delta[mask]), 99.0)), 1e-6)
        error_max = max(float(np.percentile(board_gt_error[mask], 99.0)), 1e-6)
        figure, axes = plt.subplots(1, 6, figsize=(24, 4.15), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        depth_images = [
            axes[1].imshow(gt, cmap="turbo", vmin=depth_min, vmax=depth_max),
            axes[2].imshow(fp32_metric, cmap="turbo", vmin=depth_min, vmax=depth_max),
            axes[3].imshow(board_metric, cmap="turbo", vmin=depth_min, vmax=depth_max),
        ]
        axes[1].set_title("GT depth (m)")
        axes[2].set_title(f"FP32 · RMSE {rmse(fp32_metric, gt, mask):.3f} m")
        axes[3].set_title(
            f"U250 r120 · RMSE {rmse(board_metric, gt, mask):.3f} m\n"
            "using FP32 scale+shift"
        )
        delta_image = axes[4].imshow(
            np.where(mask, signed_delta, np.nan), cmap="coolwarm",
            vmin=-delta_max, vmax=delta_max,
        )
        axes[4].set_title(
            f"U250 − FP32 depth (m)\nraw-output RMSE {raw_rmse:.3f}"
        )
        error_image = axes[5].imshow(
            np.where(mask, board_gt_error, np.nan),
            cmap="magma", vmin=0.0, vmax=error_max,
        )
        axes[5].set_title(
            f"U250 |error| vs GT\np99 {error_max:.3f} m"
        )
        for axis in axes:
            axis.axis("off")
        figure.colorbar(depth_images[-1], ax=axes[1:4], shrink=0.72, label="depth (m)")
        figure.colorbar(delta_image, ax=axes[4], shrink=0.72, label="signed delta (m)")
        figure.colorbar(error_image, ax=axes[5], shrink=0.72, label="absolute error (m)")
        figure.suptitle(
            f"Depth Anything V2-S · 280×280 · NYU {sample} · shared FP32 alignment",
            fontsize=13,
        )
        output = args.output_dir / f"{sample}_gt_fp32_u250.png"
        figure.savefig(output, dpi=160, facecolor="white")
        plt.close(figure)
        image = cv2.imread(str(output), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read rendered output {output}")
        rendered.append(image)
        evidence.append({
            "sample": sample,
            "source_h5": record["source_h5"],
            "fp32": str(args.fp32_root / f"{sample}.npy"),
            "board": str(args.board_root / f"{sample}.npz"),
            "fp32_gt_rmse_m": rmse(fp32_metric, gt, mask),
            "u250_gt_rmse_with_fp32_alignment_m": rmse(board_metric, gt, mask),
            "u250_fp32_raw_output_rmse": raw_rmse,
            "fp32_inverse_affine": {"scale": scale, "shift": shift},
        })

    width = max(image.shape[1] for image in rendered)
    resized = [
        cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        if image.shape[1] != width else image
        for image in rendered
    ]
    contact_sheet = np.concatenate(resized, axis=0)
    cv2.imwrite(str(args.output_dir / "nyu_selected_gt_fp32_u250_contact_sheet.png"),
                contact_sheet)
    (args.output_dir / "visualization_evidence.json").write_text(json.dumps({
        "schema": "depthanything-u250-r120-nyu-visualization-v1",
        "input_shape": [1, 3, 280, 280],
        "depth_colormap": "turbo",
        "signed_delta_colormap": "coolwarm",
        "absolute_error_colormap": "magma",
        "colorbars": ["depth (m)", "signed delta (m)", "absolute error (m)"],
        "font_request": "Arial",
        "font_actual": "Arial",
        "font_file": str(ARIAL_FONT),
        "font_sha256": hashlib.sha256(ARIAL_FONT.read_bytes()).hexdigest(),
        "alignment": "FP32 inverse-depth scale+shift shared by FP32 and U250",
        "reference_selection": (
            None if args.reference_selection is None
            else str(args.reference_selection.resolve())
        ),
        "reference_selection_sha256": (
            None if args.reference_selection is None else sha256(args.reference_selection)
        ),
        "samples": evidence,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
