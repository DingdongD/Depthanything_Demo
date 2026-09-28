#!/usr/bin/env python3
"""Compare captured U250 encoder block boundaries with FP32 traces."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np


class Metric:
    def __init__(self) -> None:
        self.sse = 0.0
        self.reference_sq = 0.0
        self.actual_sq = 0.0
        self.dot = 0.0
        self.absolute_error = 0.0
        self.max_absolute_error = 0.0
        self.elements = 0

    def add(self, actual: np.ndarray, reference: np.ndarray) -> None:
        actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
        reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
        if actual64.shape != reference64.shape:
            raise ValueError(
                f"shape mismatch: {actual64.shape} != {reference64.shape}"
            )
        if not np.isfinite(actual64).all() or not np.isfinite(reference64).all():
            raise ValueError("non-finite encoder boundary")
        delta = actual64 - reference64
        self.sse += float(delta @ delta)
        self.reference_sq += float(reference64 @ reference64)
        self.actual_sq += float(actual64 @ actual64)
        self.dot += float(actual64 @ reference64)
        self.absolute_error += float(np.abs(delta).sum())
        self.max_absolute_error = max(
            self.max_absolute_error, float(np.max(np.abs(delta), initial=0.0))
        )
        self.elements += int(delta.size)

    def result(self) -> dict[str, float | int]:
        return {
            "relative_l2": float(
                np.sqrt(self.sse / max(self.reference_sq, 1e-30))
            ),
            "cosine": float(
                self.dot
                / np.sqrt(max(self.actual_sq * self.reference_sq, 1e-30))
            ),
            "gain": float(self.dot / max(self.reference_sq, 1e-30)),
            "mae": float(self.absolute_error / max(self.elements, 1)),
            "rmse": float(np.sqrt(self.sse / max(self.elements, 1))),
            "max_abs": self.max_absolute_error,
            "elements": self.elements,
        }


def paired_paths(
    trace_dir: Path, reference_dir: Path
) -> list[tuple[Path, Path, str]]:
    traces = sorted(trace_dir.glob("*/*.npz")) or sorted(trace_dir.glob("*.npz"))
    pairs = []
    for trace in traces:
        relative = trace.relative_to(trace_dir)
        reference = reference_dir / relative
        if not reference.is_file():
            reference = reference_dir / trace.name
        if not reference.is_file():
            raise FileNotFoundError(f"reference missing for {trace}")
        pairs.append((trace, reference, relative.with_suffix("").as_posix()))
    if not pairs:
        raise ValueError("no trace NPZ files found")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--block-reference-dir", type=Path,
        help="optional FP32 trace root containing block_lXX",
    )
    parser.add_argument(
        "--include-internal", action="store_true",
        help="also compare Norm, attention-branch, post, and MLP boundaries",
    )
    parser.add_argument(
        "--host-plan", type=Path,
        help="host plan used to unfold folded LayerNorm affine parameters",
    )
    parser.add_argument(
        "--host-params", type=Path,
        help="host parameter archive used with --host-plan",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="source checkpoint used to unfold compiler-folded LayerNorm affine",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer <= 11:
        parser.error("--layer must be within [0, 11]")
    if (args.host_plan is None) != (args.host_params is None):
        parser.error("--host-plan and --host-params must be used together")
    if args.checkpoint is not None and args.host_plan is not None:
        parser.error("--checkpoint and --host-plan are mutually exclusive")

    layer = args.layer
    boundaries = {
        "q": (f"q_l{layer:02d}", f"encoder_l{layer:02d}_q"),
        "k": (f"k_l{layer:02d}", f"encoder_l{layer:02d}_k"),
        "v": (f"v_l{layer:02d}", f"encoder_l{layer:02d}_v"),
        "raw_attention": (
            f"attention_l{layer:02d}", f"encoder_l{layer:02d}_attention"
        ),
        "block_output": (f"block_l{layer:02d}", f"block_l{layer:02d}"),
    }
    if args.include_internal:
        for name in (
            "norm1", "attention_branch", "post", "norm2",
            "fc1", "gelu", "fc2",
        ):
            key = f"encoder_l{layer:02d}_{name}"
            boundaries[name] = (key, key)
    norm_affine = {}
    if args.checkpoint is not None:
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for name in ("norm1", "norm2"):
            prefix = f"pretrained.blocks.{layer}.{name}"
            norm_affine[name] = (
                state[f"{prefix}.weight"].detach().float().cpu().numpy(),
                state[f"{prefix}.bias"].detach().float().cpu().numpy(),
            )
    elif args.host_plan is not None:
        plan = json.loads(args.host_plan.read_text())
        with np.load(args.host_params, allow_pickle=False) as parameters:
            for name in ("norm1", "norm2"):
                specification = plan["encoder"][layer][name]
                norm_affine[name] = (
                    np.asarray(parameters[specification["scale"]], np.float32),
                    np.asarray(parameters[specification["bias"]], np.float32),
                )
    aggregate = {name: Metric() for name in boundaries}
    samples = []
    for trace_path, reference_path, sample in paired_paths(
        args.trace_dir, args.reference_dir
    ):
        sample_metrics = {}
        with contextlib.ExitStack() as stack:
            trace = stack.enter_context(np.load(trace_path, allow_pickle=False))
            reference = stack.enter_context(
                np.load(reference_path, allow_pickle=False)
            )
            block_reference = reference
            if args.block_reference_dir is not None:
                relative = trace_path.relative_to(args.trace_dir)
                block_path = args.block_reference_dir / relative
                if not block_path.is_file():
                    block_path = args.block_reference_dir / trace_path.name
                if not block_path.is_file():
                    raise FileNotFoundError(
                        f"block reference missing for {trace_path}"
                    )
                block_reference = stack.enter_context(
                    np.load(block_path, allow_pickle=False)
                )
            for name, (trace_key, reference_key) in boundaries.items():
                reference_archive = (
                    block_reference if name == "block_output" else reference
                )
                if trace_key not in trace.files:
                    raise KeyError(f"{trace_path} does not contain {trace_key}")
                if reference_key not in reference_archive.files:
                    raise KeyError(
                        f"{reference_path} does not contain {reference_key}"
                    )
                metric = Metric()
                reference_value = reference_archive[reference_key]
                if name in norm_affine:
                    scale, bias = norm_affine[name]
                    reference_value = (reference_value - bias) / scale
                metric.add(trace[trace_key], reference_value)
                sample_metrics[name] = metric.result()
                aggregate[name].add(
                    trace[trace_key], reference_value
                )
        samples.append({"sample": sample, "boundaries": sample_metrics})

    report = {
        "schema": "depthanything-u250-encoder-block-boundary-error-v1",
        "layer": layer,
        "trace_dir": str(args.trace_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "block_reference_dir": (
            str(args.block_reference_dir.resolve())
            if args.block_reference_dir is not None else None
        ),
        "sample_count": len(samples),
        "layernorm_reference": (
            "affine-unfolded" if norm_affine else "as-captured"
        ),
        "aggregate": {name: metric.result() for name, metric in aggregate.items()},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
