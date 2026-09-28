#!/usr/bin/env python3
"""Render current U250 and FP32-teacher comparisons for a DA-2K gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
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


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float32).squeeze()


def load_rgb(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32).squeeze(0)
    value = value.transpose(1, 2, 0) * STD + MEAN
    return np.clip(value, 0.0, 1.0)


def read_summary(path: Path) -> dict:
    prefix = "HYBRID_SUMMARY="
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise ValueError(f"missing HYBRID_SUMMARY in {path}")


def select_quantiles(samples: list[dict], count: int) -> list[tuple[str, dict]]:
    ordered = sorted(samples, key=lambda item: item["candidate"]["rel_l2"])
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    selected = []
    for position, index in enumerate(indices):
        if position == 0:
            label = "best raw alignment"
        elif position == len(indices) - 1:
            label = "worst raw alignment"
        else:
            percentile = int(round(100.0 * index / max(len(ordered) - 1, 1)))
            label = f"raw-error p{percentile}"
        selected.append((label, ordered[index]))
    return selected


def render_row(axes, item: dict) -> None:
    teacher = item["teacher"]
    baseline = item["baseline"]
    candidate = item["candidate"]
    aligned = item["aligned"]
    joint = np.concatenate((teacher.ravel(), baseline.ravel(), candidate.ravel()))
    lo, hi = np.percentile(joint, [1.0, 99.0])
    signed_delta = candidate - teacher
    delta_hi = max(float(np.percentile(np.abs(signed_delta), 99.0)), 1.0e-4)
    aligned_error = np.abs(aligned - teacher)
    error_hi = max(float(np.percentile(aligned_error, 99.0)), 1.0e-4)

    axes[0].imshow(item["rgb"])
    axes[0].set_title(
        f"{item['label']}\n{item['sample']}\n"
        f"wall {item['wall_ms']:.0f} ms / NPU {item['npu_ms']:.0f} ms",
        fontsize=8, loc="left", fontweight="bold",
    )
    depth_image = axes[1].imshow(teacher, cmap="turbo", vmin=lo, vmax=hi)
    axes[1].set_title(f"FP32 teacher\nshared raw range {lo:.2f}-{hi:.2f}", fontsize=8)
    axes[2].imshow(baseline, cmap="turbo", vmin=lo, vmax=hi)
    axes[2].set_title(
        f"U250 r111\nrel-L2 {item['baseline_rel']:.3f}", fontsize=8
    )
    axes[3].imshow(candidate, cmap="turbo", vmin=lo, vmax=hi)
    axes[3].set_title(
        f"U250 r113\nrel-L2 {item['candidate_rel']:.3f}, cos {item['cosine']:.4f}",
        fontsize=8,
    )
    delta_image = axes[4].imshow(
        signed_delta, cmap="coolwarm", vmin=-delta_hi, vmax=delta_hi
    )
    axes[4].set_title(f"r113 - FP32 raw\np99 |delta| {delta_hi:.3f}", fontsize=8)
    error_image = axes[5].imshow(
        aligned_error, cmap="magma", vmin=0.0, vmax=error_hi
    )
    axes[5].set_title(
        f"Affine-aligned |error|\nrel-L2 {item['affine_rel']:.3f}", fontsize=8
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure = axes[0].figure
    figure.colorbar(
        depth_image, ax=list(axes[1:4]), shrink=0.72,
        label="relative inverse depth",
    )
    figure.colorbar(delta_image, ax=axes[4], shrink=0.72, label="signed delta")
    figure.colorbar(
        error_image, ax=axes[5], shrink=0.72, label="absolute aligned error"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    if not 2 <= args.count <= 10:
        parser.error("--count must be within [2, 10]")

    comparison = json.loads(args.comparison.read_text())
    selected = select_quantiles(comparison["samples"], args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual = args.output_dir / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, record in selected:
        relative = Path(record["sample"])
        input_path = args.input_root / relative.with_suffix(".npy")
        teacher_path = args.teacher_root / relative.with_suffix(".npy")
        baseline_path = args.baseline_root / relative.with_suffix(".npz")
        candidate_path = args.candidate_root / relative.with_suffix(".npz")
        teacher = load_depth(teacher_path)
        baseline = load_depth(baseline_path)
        candidate = load_depth(candidate_path)
        if not (teacher.shape == baseline.shape == candidate.shape):
            raise ValueError(f"shape mismatch for {relative}")
        metrics = record["candidate"]
        aligned = candidate * metrics["affine_scale"] + metrics["affine_shift"]
        summary = read_summary(candidate_path.with_suffix(".log"))
        item = {
            "label": label,
            "sample": relative.as_posix(),
            "rgb": load_rgb(input_path),
            "teacher": teacher,
            "baseline": baseline,
            "candidate": candidate,
            "aligned": aligned,
            "baseline_rel": record["baseline"]["rel_l2"],
            "candidate_rel": metrics["rel_l2"],
            "cosine": metrics["cosine"],
            "affine_rel": metrics["affine_rel_l2"],
            "wall_ms": float(summary["wall_ms"]),
            "npu_ms": float(summary["npu_ms_total"]),
            "input_path": input_path,
            "teacher_path": teacher_path,
            "baseline_path": baseline_path,
            "candidate_path": candidate_path,
        }
        rows.append(item)
        figure, axes = plt.subplots(1, 6, figsize=(18, 3.2), constrained_layout=True)
        render_row(axes, item)
        figure.savefig(individual / f"{relative.name}_comparison.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(
        len(rows), 6, figsize=(18, 3.0 * len(rows)), constrained_layout=True
    )
    for row_axes, item in zip(axes, rows):
        render_row(row_axes, item)
    figure.suptitle(
        "DepthAnything V2 518x518 - DA-2K U250 r113 board output vs FP32 teacher",
        fontsize=14,
    )
    overview = args.output_dir / "da2k6_r113_board_vs_fp32_overview.png"
    figure.savefig(overview, dpi=170)
    plt.close(figure)

    evidence = {
        "schema": "depthanything-u250-r113-da2k-visualization-v1",
        "selection": "six quantiles of r113 raw relative-L2 over DA-2K10",
        "depth_colormap": "turbo",
        "signed_delta_colormap": "coolwarm",
        "absolute_error_colormap": "magma",
        "colorbars": [
            "relative inverse depth", "signed delta", "absolute aligned error"
        ],
        "comparison_sha256": sha256(args.comparison),
        "overview": str(overview.resolve()),
        "font_actual": "Arial",
        "font_file": str(ARIAL_FONT),
        "font_sha256": sha256(ARIAL_FONT),
        "samples": [],
    }
    for item in rows:
        evidence["samples"].append({
            "label": item["label"],
            "sample": item["sample"],
            "baseline_rel_l2": item["baseline_rel"],
            "candidate_rel_l2": item["candidate_rel"],
            "candidate_cosine": item["cosine"],
            "candidate_affine_rel_l2": item["affine_rel"],
            "wall_ms": item["wall_ms"],
            "npu_ms": item["npu_ms"],
            "input_sha256": sha256(item["input_path"]),
            "teacher_sha256": sha256(item["teacher_path"]),
            "candidate_sha256": sha256(item["candidate_path"]),
            "image": str(
                (individual / f"{Path(item['sample']).name}_comparison.png").resolve()
            ),
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
