#!/usr/bin/env python3
"""Analyze mixed-domain capture-and-replace outputs against FP32 depth traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_depth(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as archive:
        if "depth" in archive:
            return np.asarray(archive["depth"], dtype=np.float32)
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]], dtype=np.float32)
        raise ValueError(f"cannot select depth from {path}: {archive.files}")


def metrics(board: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    board64 = board.astype(np.float64).reshape(-1)
    ref64 = reference.astype(np.float64).reshape(-1)
    if board64.shape != ref64.shape:
        raise ValueError(f"shape mismatch {board.shape} != {reference.shape}")
    error = board64 - ref64
    denom = max(float(np.linalg.norm(ref64)), 1.0e-30)
    cosine_denom = max(
        float(np.linalg.norm(board64) * np.linalg.norm(ref64)), 1.0e-30
    )
    return {
        "relative_l2": float(np.linalg.norm(error) / denom),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "cosine": float(np.dot(board64, ref64) / cosine_denom),
    }


def evaluate(root: Path, trace_root: Path, samples: list[str]) -> dict:
    per_sample = []
    for sample in samples:
        board = load_depth(root / f"{sample}.npz")
        reference = load_depth(trace_root / f"{sample}.npz")
        per_sample.append({
            "sample": sample,
            "domain": sample.split("/", 1)[0],
            **metrics(board, reference),
        })
    domains = sorted({item["domain"] for item in per_sample})
    by_domain = {
        domain: {
            key: float(np.mean([
                item[key] for item in per_sample if item["domain"] == domain
            ]))
            for key in ("relative_l2", "rmse", "mae", "cosine")
        }
        for domain in domains
    }
    balanced = {
        key: float(np.mean([by_domain[domain][key] for domain in domains]))
        for key in ("relative_l2", "rmse", "mae", "cosine")
    }
    return {"balanced": balanced, "by_domain": by_domain, "samples": per_sample}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="skip intervention directories that do not yet contain every sample",
    )
    args = parser.parse_args()

    manifest = json.loads((args.trace_root / "manifest.json").read_text())
    samples = [item["sample"] for item in manifest["samples"]]
    baseline = evaluate(args.baseline_root, args.trace_root, samples)
    interventions = []
    skipped_incomplete = []
    for root in sorted(path for path in args.sweep_root.iterdir() if path.is_dir()):
        try:
            result = evaluate(root, args.trace_root, samples)
        except FileNotFoundError:
            if not args.allow_incomplete:
                raise
            skipped_incomplete.append(root.name)
            continue
        recovery = {
            key: baseline["balanced"][key] - result["balanced"][key]
            for key in ("relative_l2", "rmse", "mae")
        }
        recovery["cosine"] = (
            result["balanced"]["cosine"] - baseline["balanced"]["cosine"]
        )
        interventions.append({
            "target": root.name,
            "metrics": result,
            "balanced_recovery": recovery,
        })
    by_target = {item["target"]: item for item in interventions}
    for prefix, count in (("encoder_block", 12), ("decoder_conv", 32)):
        for index in range(1, count):
            current = by_target.get(f"{prefix}_{index:02d}")
            previous = by_target.get(f"{prefix}_{index - 1:02d}")
            if current is not None and previous is not None:
                current["incremental_recovery_vs_previous_boundary"] = {
                    key: (
                        current["balanced_recovery"][key]
                        - previous["balanced_recovery"][key]
                    )
                    for key in current["balanced_recovery"]
                }
    interventions.sort(
        key=lambda item: item["balanced_recovery"]["relative_l2"], reverse=True
    )
    report = {
        "schema": "depthanything-u250-mixed-capture-replace-v1",
        "method": (
            "Replace exactly one live NPU boundary with the matching FP32 trace, "
            "then execute the unchanged suffix; aggregate samples within each domain "
            "and weight NYU/DA-2K domains equally."
        ),
        "interpretation": (
            "Boundary recovery is causal but cumulative. Adjacent encoder boundary "
            "differences isolate the intervening block through the common suffix; "
            "decoder differences are descriptive because the decoder has skip branches."
        ),
        "sample_count": len(samples),
        "samples": samples,
        "baseline": baseline,
        "skipped_incomplete": skipped_incomplete,
        "interventions": interventions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "baseline": baseline["balanced"],
        "top": [
            {"target": item["target"], **item["balanced_recovery"]}
            for item in interventions[:10]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
