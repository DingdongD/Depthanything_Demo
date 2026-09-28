#!/usr/bin/env python3
"""Summarize a replacement-free U250 encoder/decoder/final-depth gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _new_accumulator() -> dict[str, float]:
    return {
        "elements": 0,
        "sum_abs": 0.0,
        "sum_sq_error": 0.0,
        "sum_sq_candidate": 0.0,
        "sum_sq_reference": 0.0,
        "dot": 0.0,
        "max_abs": 0.0,
    }


def _update(acc: dict[str, float], candidate: np.ndarray, reference: np.ndarray) -> None:
    candidate = candidate.astype(np.float64, copy=False).ravel()
    reference = reference.astype(np.float64, copy=False).ravel()
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite tensor in full gate")
    error = candidate - reference
    acc["elements"] += candidate.size
    acc["sum_abs"] += float(np.abs(error).sum())
    acc["sum_sq_error"] += float(np.dot(error, error))
    acc["sum_sq_candidate"] += float(np.dot(candidate, candidate))
    acc["sum_sq_reference"] += float(np.dot(reference, reference))
    acc["dot"] += float(np.dot(candidate, reference))
    acc["max_abs"] = max(acc["max_abs"], float(np.abs(error).max(initial=0.0)))


def _finish(acc: dict[str, float]) -> dict[str, float | int]:
    count = int(acc["elements"])
    ref_norm = np.sqrt(acc["sum_sq_reference"])
    candidate_norm = np.sqrt(acc["sum_sq_candidate"])
    return {
        "elements": count,
        "relative_l2": float(np.sqrt(acc["sum_sq_error"]) / max(ref_norm, 1e-30)),
        "cosine": float(acc["dot"] / max(ref_norm * candidate_norm, 1e-30)),
        "mae": float(acc["sum_abs"] / max(count, 1)),
        "rmse": float(np.sqrt(acc["sum_sq_error"] / max(count, 1))),
        "max_abs": float(acc["max_abs"]),
    }


def _continuous(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate = candidate.astype(np.float64, copy=False).ravel()
    reference = reference.astype(np.float64, copy=False).ravel()
    denominator = np.linalg.norm(reference)
    design = np.stack((candidate, np.ones_like(candidate)), axis=1)
    scale, shift = np.linalg.lstsq(design, reference, rcond=None)[0]
    aligned = candidate * scale + shift
    return {
        "rel_l2": float(np.linalg.norm(candidate - reference) / denominator),
        "mae": float(np.mean(np.abs(candidate - reference))),
        "cosine": float(np.dot(candidate, reference) /
                        (np.linalg.norm(candidate) * denominator)),
        "pearson": float(np.corrcoef(candidate, reference)[0, 1]),
        "affine_rel_l2": float(np.linalg.norm(aligned - reference) / denominator),
        "affine_scale": float(scale),
        "affine_shift": float(shift),
    }


def _read_summary(path: Path) -> dict:
    prefix = "HYBRID_SUMMARY="
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise ValueError(f"missing HYBRID_SUMMARY in {path}")


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _summarize_latency(summaries: list[dict]) -> dict:
    latency_fields = (
        "wall_ms", "process_wall_ms", "npu_ms_total", "h2c_ms_total",
        "c2h_ms_total", "load_ms", "codec_pack_ms_total", "codec_unpack_ms_total",
    )
    result = {
        name: {
            "mean": _mean([float(summary[name]) for summary in summaries]),
            "min": float(min(float(summary[name]) for summary in summaries)),
            "max": float(max(float(summary[name]) for summary in summaries)),
        }
        for name in latency_fields
    }
    result["physical_npu_dispatches"] = {
        "mean": _mean([float(summary["cpp_runtime"]["physical_npu_dispatches"])
                       for summary in summaries])
    }
    result["static_reloads"] = {
        "max": max(int(summary["static_reloads"]) for summary in summaries)
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--latency-root", type=Path,
                        help="optional depth-only run root for production latency")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.trace_root.glob("*/*.npz"))
    if not paths:
        raise SystemExit("no full-gate traces found")
    keys = [f"block_l{layer:02d}" for layer in range(12)]
    keys += [f"decoder_conv_{layer:02d}" for layer in range(32)]
    keys += ["depth"]
    accumulators = {key: _new_accumulator() for key in keys}
    depth_samples = []
    baseline_depth_samples = []
    summaries = []

    for trace_path in paths:
        relative = trace_path.relative_to(args.trace_root)
        reference_path = args.reference_root / relative
        with np.load(trace_path, allow_pickle=False) as candidate, \
             np.load(reference_path, allow_pickle=False) as reference:
            for key in keys:
                if key not in candidate or key not in reference:
                    raise KeyError(f"missing {key} in {relative}")
                _update(accumulators[key], candidate[key], reference[key])
            depth_samples.append({
                "sample": str(relative.with_suffix("")),
                **_continuous(candidate["depth"], reference["depth"]),
            })
            if args.baseline_root:
                with np.load(args.baseline_root / relative, allow_pickle=False) as baseline:
                    baseline_depth_samples.append({
                        "sample": str(relative.with_suffix("")),
                        **_continuous(baseline["depth"], reference["depth"]),
                    })
        summaries.append(_read_summary(trace_path.with_suffix(".log")))

    def mean_metrics(samples: list[dict]) -> dict[str, float]:
        names = ("rel_l2", "mae", "cosine", "pearson", "affine_rel_l2")
        return {name: _mean([item[name] for item in samples]) for name in names}

    report = {
        "schema": "depthanything-u250-full-gate-v1",
        "samples": [str(path.relative_to(args.trace_root).with_suffix("")) for path in paths],
        "sample_count": len(paths),
        "encoder": {key: _finish(accumulators[key]) for key in keys if key.startswith("block_")},
        "decoder": {key: _finish(accumulators[key]) for key in keys if key.startswith("decoder_")},
        "final_depth": {
            "global": _finish(accumulators["depth"]),
            "mean_per_sample": mean_metrics(depth_samples),
            "samples": depth_samples,
        },
        "capture_mode_latency": _summarize_latency(summaries),
        "finite": all(bool(summary["finite"]) for summary in summaries),
    }
    if args.latency_root:
        latency_summaries = []
        depth_matches = True
        for trace_path in paths:
            relative = trace_path.relative_to(args.trace_root)
            latency_path = args.latency_root / relative
            with np.load(trace_path, allow_pickle=False) as traced, \
                 np.load(latency_path, allow_pickle=False) as depth_only:
                depth_matches &= np.array_equal(traced["depth"], depth_only["depth"])
            latency_summaries.append(_read_summary(latency_path.with_suffix(".log")))
        report["production_latency"] = _summarize_latency(latency_summaries)
        report["depth_only_matches_capture"] = bool(depth_matches)
    if baseline_depth_samples:
        baseline = mean_metrics(baseline_depth_samples)
        candidate = report["final_depth"]["mean_per_sample"]
        report["final_depth"]["baseline_mean_per_sample"] = baseline
        report["final_depth"]["delta_candidate_minus_baseline"] = {
            key: candidate[key] - baseline[key] for key in baseline
        }
        report["final_depth"]["rel_l2_samples_improved"] = sum(
            candidate_item["rel_l2"] < baseline_item["rel_l2"]
            for candidate_item, baseline_item in zip(depth_samples, baseline_depth_samples)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sample_count": report["sample_count"],
        "block_l11": report["encoder"]["block_l11"],
        "decoder_conv_31": report["decoder"]["decoder_conv_31"],
        "final_depth": report["final_depth"]["mean_per_sample"],
        "baseline_depth": report["final_depth"].get("baseline_mean_per_sample"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
