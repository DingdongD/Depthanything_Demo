#!/usr/bin/env python3
"""Compare decoder checkpoints from two U250 hybrid runtime archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != got.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {got.shape}")
    epsilon = np.finfo(np.float64).eps
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    dot = float(np.dot(ref, got))
    gain = dot / max(float(np.dot(got, got)), epsilon)
    scaled_error = np.linalg.norm(ref - gain * got) / max(ref_norm, epsilon)
    ref_centered = ref - np.mean(ref)
    got_centered = got - np.mean(got)
    affine_gain = float(np.dot(ref_centered, got_centered)) / max(
        float(np.dot(got_centered, got_centered)), epsilon
    )
    affine_bias = float(np.mean(ref) - affine_gain * np.mean(got))
    affine_error = np.linalg.norm(ref - (affine_gain * got + affine_bias)) / max(
        ref_norm, epsilon
    )
    return {
        "shape": list(reference.shape),
        "reference_mean": float(np.mean(ref)),
        "candidate_mean": float(np.mean(got)),
        "reference_std": float(np.std(ref)),
        "candidate_std": float(np.std(got)),
        "norm_ratio_candidate_over_reference": got_norm / max(ref_norm, epsilon),
        "cosine": dot / max(ref_norm * got_norm, epsilon),
        "relative_l2": float(np.linalg.norm(ref - got) / max(ref_norm, epsilon)),
        "best_scalar_gain": gain,
        "relative_l2_after_scalar": float(scaled_error),
        "best_affine_gain": affine_gain,
        "best_affine_bias": affine_bias,
        "relative_l2_after_affine": float(affine_error),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--prefix", default="decoder_")
    parser.add_argument("--key", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.reference, allow_pickle=False) as reference, np.load(
        args.candidate, allow_pickle=False
    ) as candidate:
        if args.key:
            keys = args.key
        else:
            keys = sorted(
                key for key in reference.files
                if key.startswith(args.prefix) and key in candidate.files
            )
        result = {key: metrics(reference[key], candidate[key]) for key in keys}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
