#!/usr/bin/env python3
"""Evaluate square U250 depth outputs against DA-2K FP32 teacher and pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).squeeze()


def continuous_metrics(board: np.ndarray, teacher: np.ndarray) -> dict:
    board = board.ravel(); teacher = teacher.ravel()
    denominator = np.linalg.norm(teacher)
    design = np.stack((board, np.ones_like(board)), axis=1)
    scale, shift = np.linalg.lstsq(design, teacher, rcond=None)[0]
    aligned = board * scale + shift
    return {
        "rel_l2": float(np.linalg.norm(board - teacher) / denominator),
        "mae": float(np.mean(np.abs(board - teacher))),
        "cosine": float(np.dot(board, teacher)
                        / (np.linalg.norm(board) * denominator)),
        "pearson": float(np.corrcoef(board, teacher)[0, 1]),
        "affine_rel_l2": float(np.linalg.norm(aligned - teacher) / denominator),
        "affine_scale": float(scale),
        "affine_shift": float(shift),
    }


def point_value(depth: np.ndarray, point: list[int], width: int, height: int) -> float:
    # DA-2K stores points as [row, column], despite the generic point name.
    row, column = point
    y = min(depth.shape[0] - 1, max(0, int(row * depth.shape[0] / height)))
    x = min(depth.shape[1] - 1, max(0, int(column * depth.shape[1] / width)))
    return float(depth[y, x])


def pair_metrics(board: np.ndarray, teacher: np.ndarray, entries: list[dict],
                 width: int, height: int) -> dict:
    board_ok = teacher_ok = agreement = 0
    for entry in entries:
        board_values = [point_value(board, entry[key], width, height)
                        for key in ("point1", "point2")]
        teacher_values = [point_value(teacher, entry[key], width, height)
                          for key in ("point1", "point2")]
        board_choice = "point1" if board_values[0] > board_values[1] else "point2"
        teacher_choice = "point1" if teacher_values[0] > teacher_values[1] else "point2"
        board_ok += board_choice == entry["closer_point"]
        teacher_ok += teacher_choice == entry["closer_point"]
        agreement += board_choice == teacher_choice
    count = len(entries)
    return {
        "pairs": count,
        "board_pair_correct": board_ok,
        "fp32_pair_correct": teacher_ok,
        "board_fp32_pair_agree": agreement,
    }


def aggregate(samples: list[dict]) -> dict:
    pairs = sum(item["pairs"] for item in samples)
    result = {
        key: float(np.mean([item[key] for item in samples]))
        for key in ("rel_l2", "mae", "cosine", "pearson", "affine_rel_l2")
    }
    result.update({"images": len(samples), "pairs": pairs})
    if pairs:
        result.update({
            "board_pair_accuracy": sum(item["board_pair_correct"] for item in samples) / pairs,
            "fp32_pair_accuracy": sum(item["fp32_pair_correct"] for item in samples) / pairs,
            "board_fp32_pair_agreement": sum(
                item["board_fp32_pair_agree"] for item in samples) / pairs,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dimensions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    dimensions = json.loads(args.dimensions.read_text())
    annotations = json.loads(args.annotations.read_text())
    samples = []
    for split, records in manifest["splits"].items():
        for record in records:
            sample_id = record["sample_id"]
            output_path = args.output_root / split / f"{sample_id}.npz"
            if not output_path.exists():
                continue
            board = load_depth(output_path)
            teacher = load_depth(args.teacher_root / split / f"{sample_id}.npy")
            if board.shape != teacher.shape or not np.isfinite(board).all():
                raise ValueError(f"invalid board output: {output_path}")
            dims = dimensions[sample_id]
            result = {
                "sample_id": sample_id, "split": split, "scene": record["scene"],
                "path": record["path"], **continuous_metrics(board, teacher),
                **pair_metrics(board, teacher, annotations[record["path"]],
                               dims["width"], dims["height"]),
            }
            result["board_pair_accuracy"] = result["board_pair_correct"] / result["pairs"]
            result["fp32_pair_accuracy"] = result["fp32_pair_correct"] / result["pairs"]
            result["board_fp32_pair_agreement"] = (
                result["board_fp32_pair_agree"] / result["pairs"])
            samples.append(result)

    by_split = defaultdict(list); by_scene = defaultdict(list)
    for sample in samples:
        by_split[sample["split"]].append(sample)
        by_scene[sample["scene"]].append(sample)
    report = {
        "schema": "depthanything-da2k-board-evaluation-v1",
        "all": aggregate(samples),
        "by_split": {key: aggregate(value) for key, value in sorted(by_split.items())},
        "by_scene": {key: aggregate(value) for key, value in sorted(by_scene.items())},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["all"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
