#!/usr/bin/env python3
"""Evaluate relative inverse-depth predictions on dense NYU metric GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).squeeze()


def fit_inverse_affine(prediction: np.ndarray, gt: np.ndarray,
                       mask: np.ndarray) -> tuple[float, float]:
    target = 1.0 / gt[mask]
    design = np.stack((prediction[mask], np.ones(mask.sum())), axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(scale), float(shift)


def metric_depth(prediction: np.ndarray, scale: float, shift: float,
                 minimum: float, maximum: float) -> np.ndarray:
    inverse = np.clip(prediction * scale + shift, 1.0 / maximum, 1.0 / minimum)
    return 1.0 / inverse


def accumulate(target: dict, prediction: np.ndarray, predicted_inverse: np.ndarray,
               gt: np.ndarray, mask: np.ndarray, minimum: float,
               maximum: float) -> dict:
    pred = prediction[mask]; truth = gt[mask]; error = pred - truth
    inverse_error = predicted_inverse[mask] - 1.0 / truth
    clipped = ((predicted_inverse[mask] < 1.0 / maximum)
               | (predicted_inverse[mask] > 1.0 / minimum))
    target["count"] += int(error.size)
    target["sum_sq"] += float(np.dot(error, error))
    target["sum_sq_inverse"] += float(np.dot(inverse_error, inverse_error))
    target["sum_abs_rel"] += float(np.sum(np.abs(error) / truth))
    target["clipped"] += int(np.count_nonzero(clipped))
    target["delta1"] += int(np.count_nonzero(
        np.maximum(pred / truth, truth / pred) < 1.25
    ))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "inverse_rmse": float(np.sqrt(np.mean(inverse_error * inverse_error))),
        "abs_rel": float(np.mean(np.abs(error) / truth)),
        "delta1": float(np.mean(np.maximum(pred / truth, truth / pred) < 1.25)),
        "inverse_clip_fraction": float(np.mean(clipped)),
    }


def finish(target: dict) -> dict:
    return {
        "pixels": target["count"],
        "rmse": float(np.sqrt(target["sum_sq"] / target["count"])),
        "inverse_rmse": float(np.sqrt(target["sum_sq_inverse"] / target["count"])),
        "abs_rel": target["sum_abs_rel"] / target["count"],
        "delta1": target["delta1"] / target["count"],
        "inverse_clip_fraction": target["clipped"] / target["count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=10.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    totals = {
        name: {"count": 0, "sum_sq": 0.0, "sum_sq_inverse": 0.0,
               "sum_abs_rel": 0.0, "delta1": 0, "clipped": 0}
        for name in ("fp32_independent_alignment",
                     "u250_independent_alignment", "u250_fp32_alignment")
    }
    raw_count = 0; raw_sum_sq = 0.0; raw_sum_abs = 0.0
    raw_fp32_sq = 0.0; raw_board_sq = 0.0; raw_dot = 0.0
    samples = []
    prediction_shape: tuple[int, ...] | None = None
    for record in manifest["samples"]:
        sample = record["sample"].split("/", 1)[1]
        with h5py.File(record["source_h5"], "r") as source:
            gt = source["depth"][:].astype(np.float64)
        fp32 = load_prediction(args.fp32_root / f"{sample}.npy")
        board = load_prediction(args.board_root / f"{sample}.npz")
        if fp32.shape != board.shape:
            raise ValueError(f"prediction shape mismatch for {sample}")
        if prediction_shape is None:
            prediction_shape = tuple(fp32.shape)
        elif tuple(fp32.shape) != prediction_shape:
            raise ValueError(f"inconsistent prediction shape for {sample}")
        raw_error = board - fp32
        raw_count += raw_error.size
        raw_sum_sq += float(np.dot(raw_error.ravel(), raw_error.ravel()))
        raw_sum_abs += float(np.abs(raw_error).sum())
        raw_fp32_sq += float(np.dot(fp32.ravel(), fp32.ravel()))
        raw_board_sq += float(np.dot(board.ravel(), board.ravel()))
        raw_dot += float(np.dot(board.ravel(), fp32.ravel()))
        fp32 = cv2.resize(fp32, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        board = cv2.resize(board, (gt.shape[1], gt.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        mask = np.isfinite(gt) & (gt > args.min_depth) & (gt <= args.max_depth)
        fp_scale, fp_shift = fit_inverse_affine(fp32, gt, mask)
        board_scale, board_shift = fit_inverse_affine(board, gt, mask)
        inverses = {
            "fp32_independent_alignment": fp32 * fp_scale + fp_shift,
            "u250_independent_alignment": board * board_scale + board_shift,
            "u250_fp32_alignment": board * fp_scale + fp_shift,
        }
        predictions = {
            name: metric_depth(value, 1.0, 0.0, args.min_depth, args.max_depth)
            for name, value in inverses.items()
        }
        metrics = {
            name: accumulate(totals[name], predictions[name], value, gt, mask,
                             args.min_depth, args.max_depth)
            for name, value in inverses.items()
        }
        samples.append({
            "sample": record["sample"], "valid_pixels": int(mask.sum()),
            "fp32_inverse_affine": {"scale": fp_scale, "shift": fp_shift},
            "u250_inverse_affine": {"scale": board_scale, "shift": board_shift},
            "metrics": metrics,
            "u250_vs_fp32_raw_rmse": float(np.sqrt(np.mean(raw_error * raw_error))),
            "u250_vs_fp32_raw_rel_l2": float(
                np.linalg.norm(raw_error) / np.linalg.norm(fp32)
            ),
            "u250_vs_fp32_raw_cosine": float(
                np.dot(board.ravel(), fp32.ravel())
                / (np.linalg.norm(board) * np.linalg.norm(fp32))
            ),
        })
    aggregate = {name: finish(value) for name, value in totals.items()}
    report = {
        "schema": "depthanything-u250-nyu32-metric-v1",
        "samples": samples, "sample_count": len(samples),
        "protocol": {
            "prediction_domain": "relative inverse depth",
            "alignment": "per-image least-squares scale+shift to inverse metric GT",
            "prediction_resize": (
                f"{prediction_shape[1]}x{prediction_shape[0]} to GT size with "
                "cv2.INTER_LINEAR"
            ),
            "valid_gt_meters": [args.min_depth, args.max_depth],
            "metric_depth_clamp_meters": [args.min_depth, args.max_depth],
        },
        "aggregate": aggregate,
        "u250_minus_fp32_independent_alignment": {
            key: (aggregate["u250_independent_alignment"][key]
                  - aggregate["fp32_independent_alignment"][key])
            for key in ("rmse", "inverse_rmse", "abs_rel", "delta1",
                        "inverse_clip_fraction")
        },
        "u250_vs_fp32_raw": {
            "pixels": raw_count, "rmse": float(np.sqrt(raw_sum_sq / raw_count)),
            "mae": raw_sum_abs / raw_count,
            "relative_l2": float(np.sqrt(raw_sum_sq / raw_fp32_sq)),
            "cosine": raw_dot / np.sqrt(raw_fp32_sq * raw_board_sq),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sample_count": report["sample_count"],
        "aggregate": aggregate,
        "u250_minus_fp32_independent_alignment":
            report["u250_minus_fp32_independent_alignment"],
        "u250_vs_fp32_raw": report["u250_vs_fp32_raw"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
