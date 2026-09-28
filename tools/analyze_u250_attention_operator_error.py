#!/usr/bin/env python3
"""Attribute U250 attention error to QK, Softmax, probability INT8, and AV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    result = np.exp(shifted, dtype=np.float32)
    return result / np.sum(result, axis=-1, keepdims=True, dtype=np.float32)


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(value / scale), -128, 127).astype(np.int8)


def query_rows(tokens: int, rows_per_chunk: int) -> np.ndarray:
    rows = []
    for start in range(0, tokens, 256):
        stop = min(tokens, start + 256)
        count = min(rows_per_chunk, stop - start)
        rows.extend(np.linspace(start, stop - 1, count).astype(np.int64))
    return np.asarray(rows, dtype=np.int64)


class Metric:
    def __init__(self) -> None:
        self.sse = self.reference_sq = self.dot = self.actual_sq = 0.0

    def add(self, actual: np.ndarray, reference: np.ndarray) -> None:
        a = np.asarray(actual, dtype=np.float64).reshape(-1)
        r = np.asarray(reference, dtype=np.float64).reshape(-1)
        delta = a - r
        self.sse += float(delta @ delta)
        self.reference_sq += float(r @ r)
        self.dot += float(a @ r)
        self.actual_sq += float(a @ a)

    def result(self) -> dict[str, float]:
        return {
            "relative_l2": float(np.sqrt(self.sse / max(self.reference_sq, 1e-30))),
            "cosine": float(
                self.dot / np.sqrt(max(self.actual_sq * self.reference_sq, 1e-30))
            ),
            "gain": float(self.dot / max(self.reference_sq, 1e-30)),
        }


def paired_paths(trace_dir: Path, reference_dir: Path) -> list[tuple[Path, Path]]:
    traces = sorted(trace_dir.glob("*/*.npz")) or sorted(trace_dir.glob("*.npz"))
    pairs = []
    for trace in traces:
        relative = trace.relative_to(trace_dir)
        reference = reference_dir / relative
        if not reference.is_file():
            reference = reference_dir / trace.name
        if not reference.is_file():
            raise FileNotFoundError(f"reference missing for {trace}")
        pairs.append((trace, reference))
    if not pairs:
        raise ValueError("no trace NPZ files found")
    return pairs


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
    rows = query_rows(1370, args.rows_per_chunk)
    pairs = paired_paths(args.trace_dir, args.reference_dir)
    result = {
        "schema_version": 1,
        "layer": args.layer,
        "samples": [str(path.relative_to(args.trace_dir)) for path, _ in pairs],
        "sampled_query_rows": rows.tolist(),
        "probability_rounding": "floor",
        "av_gain_policy": "unit-required",
        "heads": [],
    }

    aggregate_names = (
        "qkv_only_attention_output",
        "qk_int8_attention_output",
        "probability_int8_attention_output",
        "v_int8_attention_output",
        "av_bf16_attention_output",
        "board_attention_output",
        "board_vs_av_bf16_surrogate",
    )
    aggregate = {name: Metric() for name in aggregate_names}
    local_names = (
        "board_q_vs_fp32",
        "board_k_vs_fp32",
        "board_v_vs_fp32",
        "qk_int8_logits_vs_board_qk",
        "qk_bf16_vs_int8_logits",
        "softmax_from_qk_vs_fp32_probability",
        "spu_bf16_vs_float_softmax",
        "probability_int8_vs_spu_bf16",
    )
    aggregate_local = {name: Metric() for name in local_names}
    aggregate_probability_elements = 0
    aggregate_probability_zeros = 0
    aggregate_probability_saturated = 0
    aggregate_probability_row_sum = []
    for head_index, specification in enumerate(heads):
        begin, end = head_index * 64, (head_index + 1) * 64
        scales = specification["scales_bf16"]
        q_scale, k_scale, v_scale = (
            float(scales[name]) for name in ("q", "k", "v")
        )
        probability_scale = float(scales["probability"])
        if abs(float(scales.get("av_output_gain", 1.0)) - 1.0) > 1e-12:
            raise ValueError(f"head {head_index} has a forbidden AV gain")
        if abs(float(scales.get("av_v", v_scale)) - v_scale) > 1e-12:
            raise ValueError(f"head {head_index} AV and V scales differ")
        names = (
            "board_q_vs_fp32",
            "board_k_vs_fp32",
            "board_v_vs_fp32",
            "qkv_only_attention_output",
            "qk_int8_logits_vs_board_qk",
            "qk_bf16_vs_int8_logits",
            "softmax_from_qk_vs_fp32_probability",
            "spu_bf16_vs_float_softmax",
            "probability_int8_vs_spu_bf16",
            "qk_int8_attention_output",
            "probability_int8_attention_output",
            "v_int8_attention_output",
            "av_bf16_attention_output",
            "board_attention_output",
            "board_vs_av_bf16_surrogate",
        )
        metrics = {name: Metric() for name in names}
        probability_elements = probability_zeros = probability_saturated = 0
        probability_row_sum = []
        for trace_path, reference_path in pairs:
            with np.load(trace_path) as trace, np.load(reference_path) as reference:
                q = trace[f"q_l{args.layer:02d}"][0, 0, :, begin:end]
                k = trace[f"k_l{args.layer:02d}"][0, 0, :, begin:end]
                v = trace[f"v_l{args.layer:02d}"][0, 0, :, begin:end]
                board = trace[f"attention_l{args.layer:02d}"][
                    0, 0, rows, begin:end
                ]
                q_ref = reference[f"encoder_l{args.layer:02d}_q"][
                    0, 0, :, begin:end
                ]
                k_ref = reference[f"encoder_l{args.layer:02d}_k"][
                    0, 0, :, begin:end
                ]
                v_ref = reference[f"encoder_l{args.layer:02d}_v"][
                    0, 0, :, begin:end
                ]
                attention_ref = reference[f"encoder_l{args.layer:02d}_attention"][
                    0, 0, rows, begin:end
                ]

            metrics["board_q_vs_fp32"].add(q, q_ref)
            metrics["board_k_vs_fp32"].add(k, k_ref)
            metrics["board_v_vs_fp32"].add(v, v_ref)
            logits_ref = q_ref[rows] @ k_ref.T
            probability_ref = softmax(logits_ref)
            qkv_only = softmax(q[rows] @ k.T) @ v
            metrics["qkv_only_attention_output"].add(qkv_only, attention_ref)

            qi = quantize(q[rows], q_scale).astype(np.int32)
            ki = quantize(k, k_scale).astype(np.int32)
            vi = quantize(v, v_scale).astype(np.int32)
            logits_board = q[rows] @ k.T
            logits_int8 = (qi @ ki.T).astype(np.float32) * np.float32(
                q_scale * k_scale
            )
            logits_bf16 = bf16(logits_int8)
            metrics["qk_int8_logits_vs_board_qk"].add(logits_int8, logits_board)
            metrics["qk_bf16_vs_int8_logits"].add(logits_bf16, logits_int8)
            probability_float = softmax(logits_bf16)
            probability_bf16 = bf16(probability_float)
            metrics["softmax_from_qk_vs_fp32_probability"].add(
                probability_float, probability_ref
            )
            metrics["spu_bf16_vs_float_softmax"].add(
                probability_bf16, probability_float
            )
            probability_code = np.clip(
                np.floor(probability_bf16 / probability_scale), 0, 127
            ).astype(np.int32)
            probability_dequant = probability_code.astype(np.float32) * np.float32(
                probability_scale
            )
            metrics["probability_int8_vs_spu_bf16"].add(
                probability_dequant, probability_bf16
            )
            probability_elements += probability_code.size
            probability_zeros += int(np.count_nonzero(probability_code == 0))
            probability_saturated += int(np.count_nonzero(probability_code == 127))
            probability_row_sum.extend(
                np.sum(probability_dequant, axis=-1, dtype=np.float64).tolist()
            )

            qk_output = probability_bf16 @ v
            probability_output = probability_dequant @ v
            v_dequant = vi.astype(np.float32) * np.float32(v_scale)
            v_int8_output = probability_dequant @ v_dequant
            av_bf16 = bf16(
                (probability_code @ vi).astype(np.float32)
                * np.float32(probability_scale * v_scale)
            )
            for name, value in (
                ("qk_int8_attention_output", qk_output),
                ("probability_int8_attention_output", probability_output),
                ("v_int8_attention_output", v_int8_output),
                ("av_bf16_attention_output", av_bf16),
                ("board_attention_output", board),
            ):
                metrics[name].add(value, attention_ref)
            metrics["board_vs_av_bf16_surrogate"].add(board, av_bf16)
            if head_index not in host_heads:
                for name, actual, reference_value in (
                    ("board_q_vs_fp32", q, q_ref),
                    ("board_k_vs_fp32", k, k_ref),
                    ("board_v_vs_fp32", v, v_ref),
                    (
                        "qk_int8_logits_vs_board_qk",
                        logits_int8,
                        logits_board,
                    ),
                    ("qk_bf16_vs_int8_logits", logits_bf16, logits_int8),
                    (
                        "softmax_from_qk_vs_fp32_probability",
                        probability_float,
                        probability_ref,
                    ),
                    (
                        "spu_bf16_vs_float_softmax",
                        probability_bf16,
                        probability_float,
                    ),
                    (
                        "probability_int8_vs_spu_bf16",
                        probability_dequant,
                        probability_bf16,
                    ),
                ):
                    aggregate_local[name].add(actual, reference_value)
                aggregate_probability_elements += probability_code.size
                aggregate_probability_zeros += int(
                    np.count_nonzero(probability_code == 0)
                )
                aggregate_probability_saturated += int(
                    np.count_nonzero(probability_code == 127)
                )
                aggregate_probability_row_sum.extend(
                    np.sum(probability_dequant, axis=-1, dtype=np.float64).tolist()
                )
                aggregate["qkv_only_attention_output"].add(
                    qkv_only, attention_ref
                )
                for name, value in (
                    ("qk_int8_attention_output", qk_output),
                    ("probability_int8_attention_output", probability_output),
                    ("v_int8_attention_output", v_int8_output),
                    ("av_bf16_attention_output", av_bf16),
                    ("board_attention_output", board),
                ):
                    aggregate[name].add(value, attention_ref)
                aggregate["board_vs_av_bf16_surrogate"].add(board, av_bf16)

        head_result = {
            "head": head_index,
            "backend": "host_fp32" if head_index in host_heads else "npu_int8",
            "scales_bf16": scales,
            "metrics": {name: value.result() for name, value in metrics.items()},
            "probability_int8": {
                "zero_fraction": probability_zeros / probability_elements,
                "saturation_fraction": probability_saturated / probability_elements,
                "row_sum_mean": float(np.mean(probability_row_sum)),
                "row_sum_min": float(np.min(probability_row_sum)),
                "row_sum_max": float(np.max(probability_row_sum)),
            },
        }
        result["heads"].append(head_result)

    result["aggregate_npu_attention_output"] = {
        name: value.result() for name, value in aggregate.items()
    }
    result["aggregate_npu_local_operator"] = {
        name: value.result() for name, value in aggregate_local.items()
    }
    result["aggregate_npu_probability_int8"] = {
        "zero_fraction": (
            aggregate_probability_zeros / aggregate_probability_elements
        ),
        "saturation_fraction": (
            aggregate_probability_saturated / aggregate_probability_elements
        ),
        "row_sum_mean": float(np.mean(aggregate_probability_row_sum)),
        "row_sum_min": float(np.min(aggregate_probability_row_sum)),
        "row_sum_max": float(np.max(aggregate_probability_row_sum)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate_npu_attention_output"], sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
