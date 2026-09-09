#!/usr/bin/env python3
"""Tune U250 Softmax/AV scales against FP32 attention on board-state QKV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DENOMINATORS = (127, 255, 511, 1023, 2047, 4095, 8191, 16383)


def bf16(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    shape = value.shape
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return np.asarray(
        rounded & np.uint32(0xFFFF0000), dtype=np.uint32
    ).reshape(-1).view(np.float32).reshape(shape)


def bf16_scalar(value: float) -> float:
    return float(bf16(value))


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=-1, keepdims=True)


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(bf16(value) / scale), -127, 127).astype(np.int32)


def row_indices(tokens: int, rows_per_chunk: int) -> np.ndarray:
    rows = []
    for start in range(0, tokens, 256):
        stop = min(start + 256, tokens)
        count = min(rows_per_chunk, stop - start)
        rows.extend(np.linspace(start, stop - 1, count).astype(np.int64))
    return np.asarray(rows, dtype=np.int64)


def metric(sse: float, reference_sq: float, dot: float, actual_sq: float) -> dict:
    return {
        "relative_l2": float(np.sqrt(sse / max(reference_sq, 1e-30))),
        "cosine": float(dot / np.sqrt(max(reference_sq * actual_sq, 1e-30))),
    }


def evaluate_head(
    pairs: list[tuple[Path, Path]], layer: int, head: int, scales: dict,
    rows: np.ndarray, denominator: int, av_v: float,
) -> tuple[float, float, float, float]:
    probability_scale = bf16_scalar(1.0 / denominator)
    sse = reference_sq = dot = actual_sq = 0.0
    begin, end = head * 64, (head + 1) * 64
    for trace_path, reference_path in pairs:
        with np.load(trace_path) as trace, np.load(reference_path) as reference:
            q = trace[f"q_l{layer:02d}"][0, 0, :, begin:end]
            k = trace[f"k_l{layer:02d}"][0, 0, :, begin:end]
            v = trace[f"v_l{layer:02d}"][0, 0, :, begin:end]
            target = reference[f"attention_l{layer:02d}"][0, 0, rows, begin:end]
        qi = quantize(q[rows], float(scales["q"]))
        ki = quantize(k, float(scales["k"]))
        vi = quantize(v, float(scales["v"]))
        logits = bf16((qi @ ki.T).astype(np.float32)
                      * np.float32(float(scales["q"]) * float(scales["k"])))
        probability = bf16(softmax(logits).astype(np.float32))
        probability_code = np.clip(
            np.floor(probability / probability_scale), 0, 127
        ).astype(np.int32)
        actual = bf16((probability_code @ vi).astype(np.float32)
                      * np.float32(probability_scale * av_v))
        a = actual.astype(np.float64).reshape(-1)
        r = target.astype(np.float64).reshape(-1)
        d = a - r
        sse += float(d @ d)
        reference_sq += float(r @ r)
        dot += float(a @ r)
        actual_sq += float(a @ a)
    return sse, reference_sq, dot, actual_sq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,1")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--gain-min", type=float, default=0.5)
    parser.add_argument("--gain-max", type=float, default=2.0)
    args = parser.parse_args()

    layers = [int(item) for item in args.layers.split(",")]
    trace_paths = sorted(args.trace_dir.glob("*.npz"))
    pairs = [(path, args.reference_dir / path.name) for path in trace_paths]
    if any(not reference.is_file() for _, reference in pairs):
        raise FileNotFoundError("one or more FP32 attention references are missing")
    required = args.train_count + args.validation_count
    if len(pairs) < required:
        raise ValueError(f"need {required} traces, found {len(pairs)}")
    train, validation = pairs[:args.train_count], pairs[args.train_count:required]
    contract = json.loads(args.contract.read_text())
    rows = row_indices(1370, args.rows_per_chunk)
    result = {
        "schema_version": 1,
        "objective": "FP32 attention output from board-state BF16 QKV",
        "probability_rounding": "floor",
        "candidate_denominators": list(DENOMINATORS),
        "train_samples": [path.name for path, _ in train],
        "validation_samples": [path.name for path, _ in validation],
        "sampled_query_rows": rows.tolist(),
        "layers": {},
    }

    for layer in layers:
        head_results = []
        for head, specification in enumerate(
                contract["encoder"][layer]["attention"]["heads"]):
            old = specification["scales_bf16"]
            candidates = {}
            for denominator in DENOMINATORS:
                # First obtain the continuous least-squares gain.
                base = evaluate_head(
                    train, layer, head, old, rows, denominator, float(old["v"])
                )
                _, _, dot, actual_sq = base
                gain = np.clip(
                    dot / max(actual_sq, 1e-30), args.gain_min, args.gain_max
                )
                gain = bf16_scalar(float(gain))
                av_v = bf16_scalar(float(old["v"]) * gain)
                train_values = evaluate_head(
                    train, layer, head, old, rows, denominator, av_v
                )
                validation_values = evaluate_head(
                    validation, layer, head, old, rows, denominator, av_v
                )
                candidates[str(denominator)] = {
                    "probability": bf16_scalar(1.0 / denominator),
                    "av_output_gain": gain,
                    "av_v": av_v,
                    "train": metric(*train_values),
                    "validation": metric(*validation_values),
                    "train_sse": train_values[0],
                }
            selected_denominator = min(
                DENOMINATORS,
                key=lambda value: candidates[str(value)]["train_sse"],
            )
            selected = candidates[str(selected_denominator)]
            old_probability = float(old["probability"])
            old_denominator = min(
                DENOMINATORS,
                key=lambda value: abs(bf16_scalar(1.0 / value) - old_probability),
            )
            old_gain = float(old.get("av_output_gain", 1.0))
            old_av_v = float(old.get("av_v", bf16_scalar(float(old["v"]) * old_gain)))
            old_train = evaluate_head(
                train, layer, head, old, rows, old_denominator, old_av_v
            )
            old_validation = evaluate_head(
                validation, layer, head, old, rows, old_denominator, old_av_v
            )
            head_results.append({
                "head": head,
                "old_scales_bf16": old,
                "selected_scales_bf16": {
                    "q": float(old["q"]), "k": float(old["k"]),
                    "v": float(old["v"]),
                    "probability": selected["probability"],
                    "av_output_gain": selected["av_output_gain"],
                    "av_v": selected["av_v"],
                },
                "old_denominator": old_denominator,
                "selected_denominator": selected_denominator,
                "old_train": metric(*old_train),
                "old_validation": metric(*old_validation),
                "selected_train": selected["train"],
                "selected_validation": selected["validation"],
                "candidates": candidates,
            })
        result["layers"][str(layer)] = {"heads": head_results}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for layer, record in result["layers"].items():
        print("layer", layer)
        for head in record["heads"]:
            print(head["head"], head["old_denominator"],
                  head["selected_denominator"],
                  f"val {head['old_validation']['relative_l2']:.6f} -> "
                  f"{head['selected_validation']['relative_l2']:.6f}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
