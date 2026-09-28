#!/usr/bin/env python3
"""Render an RGB input beside a U250 Depth Anything relative-depth tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_depth(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        depth = loaded
    else:
        with loaded as archive:
            depth = archive["depth"] if "depth" in archive else archive[archive.files[0]]
    return np.asarray(depth, dtype=np.float32).squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-color-output", type=Path)
    args = parser.parse_args()

    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"failed to read {args.rgb}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = load_depth(args.depth)
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    low, high = np.percentile(depth[np.isfinite(depth)], [2.0, 98.0])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB input (demo05)")
    axes[0].axis("off")
    image = axes[1].imshow(depth, cmap="magma", vmin=low, vmax=high)
    axes[1].set_title("U250 r43 relative depth")
    axes[1].axis("off")
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02)
    colorbar.set_label("relative depth (2–98% display range)")
    figure.suptitle(
        f"Depth Anything V2-S on DS NPU · 518×518 · min={depth.min():.4f}, "
        f"max={depth.max():.4f}, mean={depth.mean():.4f}",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)

    if args.raw_color_output:
        normalized = np.clip((depth - low) / max(high - low, 1e-12), 0.0, 1.0)
        color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        args.raw_color_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.raw_color_output), color)


if __name__ == "__main__":
    main()
