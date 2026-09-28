#!/usr/bin/env python3
"""Render 280x280 DA-2K FP32/U250 comparisons in the 518 visual style."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
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


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float64).squeeze()
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).squeeze()


def resolve_depth(root: Path, stem: str) -> Path:
    matches = [path for suffix in (".npy", ".npz")
               if (path := root / f"{stem}{suffix}").is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one .npy/.npz prediction for {stem} under {root}, "
            f"found {matches}"
        )
    return matches[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def affine_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.stack((source.ravel(), np.ones(source.size)), axis=1)
    scale, shift = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
    return source * scale + shift, float(scale), float(shift)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True,
                        help="output stem, for example 01_train_da2k_0000")
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

    calibration = json.loads(args.calibration_manifest.read_text())
    sources = {
        item["sample_id"]: item for item in calibration["samples"]
        if item.get("domain", "da2k") == "da2k"
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    evidence = []

    for stem in args.samples:
        # Fixed comparison manifests use the canonical sample id directly.
        # Retain support for the older numbered mixed-calibration stems.
        sample_id = stem if stem in sources else "tuning_" + stem.rsplit("_", 1)[-1]
        record = sources[sample_id]
        rgb_bgr = cv2.imread(record["source"], cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"failed to read {record['source']}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        fp32_path = resolve_depth(args.fp32_root, stem)
        board_path = resolve_depth(args.board_root, stem)
        fp32 = load_depth(fp32_path)
        board = load_depth(board_path)
        if fp32.shape != board.shape:
            raise ValueError(f"prediction shape mismatch for {stem}")
        rgb = cv2.resize(rgb, (fp32.shape[1], fp32.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

        aligned, scale, shift = affine_align(board, fp32)
        signed_delta = board - fp32
        aligned_error = np.abs(aligned - fp32)
        joint = np.concatenate((fp32.ravel(), board.ravel()))
        depth_min, depth_max = np.percentile(joint, [1.0, 99.0])
        delta_max = max(float(np.percentile(np.abs(signed_delta), 99.0)), 1e-6)
        error_max = max(float(np.percentile(aligned_error, 99.0)), 1e-6)
        raw_rel_l2 = float(np.linalg.norm(signed_delta) / np.linalg.norm(fp32))
        raw_cosine = float(
            np.dot(board.ravel(), fp32.ravel())
            / (np.linalg.norm(board) * np.linalg.norm(fp32))
        )
        affine_rel_l2 = float(np.linalg.norm(aligned - fp32) / np.linalg.norm(fp32))

        figure, axes = plt.subplots(1, 5, figsize=(20, 4.1), constrained_layout=True)
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB · {record['scene']}")
        depth_images = [
            axes[1].imshow(fp32, cmap="turbo", vmin=depth_min, vmax=depth_max),
            axes[2].imshow(board, cmap="turbo", vmin=depth_min, vmax=depth_max),
        ]
        axes[1].set_title("FP32 teacher · relative inverse depth")
        axes[2].set_title(f"U250 r120\nrelL2 {raw_rel_l2:.4f} · cos {raw_cosine:.5f}")
        delta_image = axes[3].imshow(
            signed_delta, cmap="coolwarm", vmin=-delta_max, vmax=delta_max
        )
        axes[3].set_title(f"U250 − FP32\np99 |delta| {delta_max:.3f}")
        error_image = axes[4].imshow(
            aligned_error, cmap="magma", vmin=0.0, vmax=error_max
        )
        axes[4].set_title(f"Affine-aligned |error|\nrelL2 {affine_rel_l2:.4f}")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(depth_images[-1], ax=axes[1:3], shrink=0.72,
                        label="relative inverse depth")
        figure.colorbar(delta_image, ax=axes[3], shrink=0.72, label="signed delta")
        figure.colorbar(error_image, ax=axes[4], shrink=0.72,
                        label="absolute aligned error")
        figure.suptitle(
            f"Depth Anything V2-S · 280×280 · DA-2K {sample_id}", fontsize=13
        )
        output = args.output_dir / f"{sample_id}_fp32_u250.png"
        figure.savefig(output, dpi=160, facecolor="white")
        plt.close(figure)
        rendered_image = cv2.imread(str(output), cv2.IMREAD_COLOR)
        if rendered_image is None:
            raise RuntimeError(f"failed to read rendered output {output}")
        rendered.append(rendered_image)
        evidence.append({
            "sample": sample_id, "scene": record["scene"],
            "source": record["source"], "fp32": str(fp32_path),
            "board": str(board_path),
            "raw_relative_l2": raw_rel_l2, "raw_cosine": raw_cosine,
            "affine_scale": scale, "affine_shift": shift,
            "affine_relative_l2": affine_rel_l2,
        })

    width = max(image.shape[1] for image in rendered)
    rows = [
        cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        if image.shape[1] != width else image for image in rendered
    ]
    cv2.imwrite(str(args.output_dir / "da2k_selected_fp32_u250_contact_sheet.png"),
                np.concatenate(rows, axis=0))
    (args.output_dir / "visualization_evidence.json").write_text(json.dumps({
        "schema": "depthanything-u250-r120-da2k-visualization-v1",
        "input_shape": [1, 3, 280, 280],
        "depth_colormap": "turbo",
        "signed_delta_colormap": "coolwarm",
        "absolute_error_colormap": "magma",
        "colorbars": [
            "relative inverse depth", "signed delta", "absolute aligned error"
        ],
        "font_request": "Arial",
        "font_actual": "Arial",
        "font_file": str(ARIAL_FONT),
        "font_sha256": hashlib.sha256(ARIAL_FONT.read_bytes()).hexdigest(),
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
