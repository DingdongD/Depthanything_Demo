#!/usr/bin/env python3
"""Compare unit-amplitude INT8 probability encodings for U250 attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_u250_attention_operator_error import (
    Metric,
    bf16,
    paired_paths,
    quantize,
    query_rows,
    softmax,
)


def largest_remainder_codes(probability: np.ndarray, scale: float) -> np.ndarray:
    """Project onto the capped integer simplex with row sum nearest to 1/scale."""
    normalized = np.asarray(probability, dtype=np.float32) / np.float32(scale)
    target = int(np.rint(1.0 / scale))
    if target > normalized.shape[1] * 127:
        raise ValueError(f"probability code sum {target} exceeds INT8 capacity")
    low = np.full((normalized.shape[0], 1), -127.0, dtype=np.float32)
    high = np.full((normalized.shape[0], 1), 127.0, dtype=np.float32)
    for _ in range(24):
        middle = (low + high) * np.float32(0.5)
        row_sum = np.sum(np.clip(normalized + middle, 0, 127), axis=1, keepdims=True)
        low = np.where(row_sum < target, middle, low)
        high = np.where(row_sum < target, high, middle)
    projected = np.clip(normalized + high, 0, 127)
    codes = np.floor(projected).astype(np.int16)
    fractions = projected - codes
    for row in range(codes.shape[0]):
        residual = target - int(np.sum(codes[row], dtype=np.int64))
        if residual:
            eligible = np.flatnonzero(codes[row] < 127)
            selected = eligible[
                np.argpartition(fractions[row, eligible], -residual)[-residual:]
            ]
            codes[row, selected] += 1
        if int(np.sum(codes[row], dtype=np.int64)) != target:
            raise AssertionError("fixed-sum probability quantization failed")
    return codes.astype(np.int32)


def probability_codes(
    probability: np.ndarray, scale: float, method: str
) -> np.ndarray:
    scaled = probability / np.float32(scale)
    if method == "floor":
        return np.clip(np.floor(scaled), 0, 127).astype(np.int32)
    if method == "nearest":
        return np.clip(np.rint(scaled), 0, 127).astype(np.int32)
    if method == "largest_remainder":
        return largest_remainder_codes(probability, scale)
    raise ValueError(method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    layer = contract["encoder"][args.layer]
    heads = layer["attention"]["heads"]
    host_heads = set(layer.get("host_attention_heads", []))
    pairs = paired_paths(args.trace_dir, args.reference_dir)
    rows = query_rows(1370, args.rows_per_chunk)
    methods = ("floor", "nearest", "largest_remainder", "floor_row_renorm")
    metrics = {
        method: {"probability": Metric(), "attention": Metric()}
        for method in methods
    }
    statistics = {
        method: {"elements": 0, "zeros": 0, "saturated": 0, "row_sums": []}
        for method in methods
    }

    for head, specification in enumerate(heads):
        if head in host_heads:
            continue
        begin, end = head * 64, (head + 1) * 64
        scales = specification["scales_bf16"]
        q_scale = float(scales["q"])
        k_scale = float(scales["k"])
        v_scale = float(scales["v"])
        probability_scale = float(scales["probability"])
        if float(scales.get("av_output_gain", 1.0)) != 1.0:
            raise ValueError(f"layer {args.layer} head {head}: forbidden AV gain")
        if float(scales.get("av_v", v_scale)) != v_scale:
            raise ValueError(f"layer {args.layer} head {head}: AV scale differs from V")
        for trace_path, reference_path in pairs:
            with np.load(trace_path) as trace, np.load(reference_path) as reference:
                q = trace[f"q_l{args.layer:02d}"][0, 0, :, begin:end]
                k = trace[f"k_l{args.layer:02d}"][0, 0, :, begin:end]
                v = trace[f"v_l{args.layer:02d}"][0, 0, :, begin:end]
                target = reference[f"encoder_l{args.layer:02d}_attention"][
                    0, 0, rows, begin:end
                ]
            qi = quantize(q[rows], q_scale).astype(np.int32)
            ki = quantize(k, k_scale).astype(np.int32)
            vi = quantize(v, v_scale).astype(np.int32)
            logits = bf16(
                (qi @ ki.T).astype(np.float32) * np.float32(q_scale * k_scale)
            )
            probability = bf16(softmax(logits))
            for method in methods:
                base_method = "floor" if method == "floor_row_renorm" else method
                codes = probability_codes(probability, probability_scale, base_method)
                dequantized = codes.astype(np.float32) * np.float32(probability_scale)
                if method == "floor_row_renorm":
                    row_sum = np.sum(dequantized, axis=-1, keepdims=True)
                    dequantized = np.divide(
                        dequantized,
                        row_sum,
                        out=np.zeros_like(dequantized),
                        where=row_sum != 0,
                    )
                    actual = bf16(dequantized @ (vi.astype(np.float32) * v_scale))
                else:
                    actual = bf16(
                        (codes @ vi).astype(np.float32)
                        * np.float32(probability_scale * v_scale)
                    )
                metrics[method]["probability"].add(dequantized, probability)
                metrics[method]["attention"].add(actual, target)
                stat = statistics[method]
                stat["elements"] += codes.size
                stat["zeros"] += int(np.count_nonzero(codes == 0))
                stat["saturated"] += int(np.count_nonzero(codes == 127))
                stat["row_sums"].extend(
                    np.sum(dequantized, axis=-1, dtype=np.float64).tolist()
                )

    result = {
        "schema_version": 1,
        "layer": args.layer,
        "samples": [str(path.relative_to(args.trace_dir)) for path, _ in pairs],
        "sampled_query_rows": rows.tolist(),
        "av_gain_policy": "unit-required",
        "methods": {},
    }
    for method in methods:
        stat = statistics[method]
        result["methods"][method] = {
            "probability": metrics[method]["probability"].result(),
            "attention": metrics[method]["attention"].result(),
            "zero_fraction": stat["zeros"] / stat["elements"],
            "saturation_fraction": stat["saturated"] / stat["elements"],
            "row_sum_mean": float(np.mean(stat["row_sums"])),
            "row_sum_min": float(np.min(stat["row_sums"])),
            "row_sum_max": float(np.max(stat["row_sums"])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["methods"], indent=2, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
