#!/usr/bin/env python3
"""Decompose U250 attention error into BF16, probability-A8 and hardware parts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16_round(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    return {
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * np.linalg.norm(expected))),
        "relative_l2": float(np.linalg.norm(actual - expected) /
                             np.linalg.norm(expected)),
        "rmse": float(np.sqrt(np.mean((actual - expected) ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probability-rounding", choices=("round", "floor"), default="round",
        help="quantization rule used by SPU Softmax output",
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    report = {"schema_version": 1,
              "probability_rounding": args.probability_rounding, "layers": []}
    with np.load(args.trace, allow_pickle=False) as trace:
        for layer, block in enumerate(contract["encoder"]):
            q = trace[f"q_l{layer:02d}"][0, 0].astype(np.int32)
            k = trace[f"k_l{layer:02d}"][0, 0].astype(np.int32)
            v = trace[f"v_l{layer:02d}"][0, 0].astype(np.int32)
            hardware = trace[f"attention_l{layer:02d}"][0, 0].astype(np.float32)
            q_scale = float(block["qkv"]["output_scales"]["q"])
            k_scale = float(block["qkv"]["output_scales"]["k"])
            v_scale = float(block["qkv"]["output_scales"]["v"])
            exact_heads = []
            bf16_heads = []
            probability_a8_heads = []
            for head, spec in enumerate(block["attention"]["heads"]):
                begin, end = head * 64, (head + 1) * 64
                qh, kh, vh = q[:, begin:end], k[:, begin:end], v[:, begin:end]
                accumulator = (qh @ kh.T).astype(np.float32)
                logits_fp32 = accumulator * np.float32(q_scale * k_scale)
                exact_probability = softmax(logits_fp32)
                exact_heads.append(exact_probability @
                                   (vh.astype(np.float32) * v_scale))

                logits_bf16 = bf16_round(logits_fp32)
                probability_bf16 = bf16_round(softmax(logits_bf16))
                bf16_heads.append(probability_bf16 @
                                  (vh.astype(np.float32) * v_scale))

                probability_scale = float(spec["scales_bf16"]["probability"])
                scaled_probability = probability_bf16 / probability_scale
                probability_code = (
                    np.floor(scaled_probability)
                    if args.probability_rounding == "floor"
                    else np.rint(scaled_probability)
                )
                probability_code = np.clip(
                    probability_code, 0, 127
                ).astype(np.int32)
                probability_a8_heads.append(bf16_round(
                    (probability_code @ vh).astype(np.float32) *
                    np.float32(probability_scale * v_scale)
                ))
            exact = np.concatenate(exact_heads, axis=-1)
            bf16 = np.concatenate(bf16_heads, axis=-1)
            probability_a8 = np.concatenate(probability_a8_heads, axis=-1)
            report["layers"].append({
                "layer": layer,
                "bf16_vs_fp32": metrics(bf16, exact),
                "probability_a8_vs_bf16": metrics(probability_a8, bf16),
                "modeled_pipeline_vs_fp32": metrics(probability_a8, exact),
                "hardware_vs_modeled_pipeline": metrics(hardware, probability_a8),
                "hardware_vs_fp32": metrics(hardware, exact),
            })
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for item in report["layers"]:
        print(item["layer"],
              f"bf16={item['bf16_vs_fp32']['relative_l2']:.6f}",
              f"p_a8={item['probability_a8_vs_bf16']['relative_l2']:.6f}",
              f"hw={item['hardware_vs_modeled_pipeline']['relative_l2']:.6f}",
              f"total={item['hardware_vs_fp32']['relative_l2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
