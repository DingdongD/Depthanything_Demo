#!/usr/bin/env python3
"""Evaluate mathematically additive multi-A8 probability representations.

No scheme in this file changes V, fits an output gain, or rescales a completed
attention result.  Query chunks select an independent static probability
quantizer.  Key partitions are disjoint and their AV partials are summed.
Radix routes encode the non-negative quantization residual and sum their AV
partials, which is an accuracy ceiling until the bitstream can form that
residual on device.
"""

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


DENOMINATORS = (127, 255, 511, 1023, 2047, 4095, 8191, 16383)
QUERY_CHUNKS = (
    (0, 256),
    (256, 512),
    (512, 768),
    (768, 1024),
    (1024, 1280),
    (1280, 1370),
)


def bf16_scalar(value: float) -> float:
    return float(bf16(np.asarray([value], dtype=np.float32))[0])


def candidate_scales() -> dict[int, float]:
    return {value: bf16_scalar(1.0 / value) for value in DENOMINATORS}


def floor_codes(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.floor(value / np.float32(scale)), 0, 127).astype(np.int32)


def dynamic_row_max_codes(
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match SPU DQU: one BF16 max/127 scale and nearest A8 per row."""
    scales = bf16(
        np.maximum(
            np.max(probability, axis=-1, keepdims=True) / np.float32(127.0),
            np.float32(2.0 ** -24),
        )
    )
    codes = np.clip(np.rint(probability / scales), 0, 127).astype(np.int32)
    return codes, scales


def radix_codes(
    probability: np.ndarray, routes: int
) -> tuple[list[np.ndarray], list[float]]:
    """Encode P as base-127 non-negative digits without duplicate counting."""
    if routes < 1:
        raise ValueError("routes must be positive")
    scales = [bf16_scalar(1.0 / 127.0)]
    for _ in range(1, routes):
        scales.append(bf16_scalar(scales[-1] / 127.0))
    residual = probability.astype(np.float32, copy=True)
    codes = []
    for scale in scales:
        code = floor_codes(residual, scale)
        codes.append(code)
        residual = np.maximum(
            residual - code.astype(np.float32) * np.float32(scale), 0.0
        )
    return codes, scales


def chunk_id_for_rows(rows: np.ndarray) -> np.ndarray:
    result = np.empty(rows.shape, dtype=np.int64)
    for index, (start, stop) in enumerate(QUERY_CHUNKS):
        result[(rows >= start) & (rows < stop)] = index
    return result


def key_ranges(tokens: int, parts: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, tokens, parts + 1).astype(np.int64)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(parts)]


def add_probability_stats(target: dict, codes: list[np.ndarray],
                          dequantized: np.ndarray) -> None:
    target["route_elements"] += sum(code.size for code in codes)
    target["route_zeros"] += sum(int(np.count_nonzero(code == 0)) for code in codes)
    target["route_saturated"] += sum(
        int(np.count_nonzero(code == 127)) for code in codes
    )
    target["logical_elements"] += dequantized.size
    target["logical_zeros"] += int(np.count_nonzero(dequantized == 0))
    target["row_sums"].extend(
        np.sum(dequantized, axis=-1, dtype=np.float64).tolist()
    )


def empty_stats() -> dict:
    return {
        "route_elements": 0,
        "route_zeros": 0,
        "route_saturated": 0,
        "logical_elements": 0,
        "logical_zeros": 0,
        "row_sums": [],
    }


def finish_stats(value: dict) -> dict:
    return {
        "route_zero_fraction": value["route_zeros"] / value["route_elements"],
        "route_saturation_fraction": (
            value["route_saturated"] / value["route_elements"]
        ),
        "logical_zero_fraction": (
            value["logical_zeros"] / value["logical_elements"]
        ),
        "row_sum_mean": float(np.mean(value["row_sums"])),
        "row_sum_min": float(np.min(value["row_sums"])),
        "row_sum_max": float(np.max(value["row_sums"])),
    }


def load_head(
    trace_path: Path,
    reference_path: Path,
    layer: int,
    head: int,
    rows: np.ndarray,
    scales: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    begin, end = head * 64, (head + 1) * 64
    with np.load(trace_path) as trace, np.load(reference_path) as reference:
        q = trace[f"q_l{layer:02d}"][0, 0, :, begin:end]
        k = trace[f"k_l{layer:02d}"][0, 0, :, begin:end]
        v = trace[f"v_l{layer:02d}"][0, 0, :, begin:end]
        key = f"encoder_l{layer:02d}_attention"
        if key not in reference:
            key = f"attention_l{layer:02d}"
        target = reference[key][0, 0, rows, begin:end]
    qi = quantize(q[rows], float(scales["q"])).astype(np.int32)
    ki = quantize(k, float(scales["k"])).astype(np.int32)
    vi = quantize(v, float(scales["v"])).astype(np.int32)
    logits = bf16(
        (qi @ ki.T).astype(np.float32)
        * np.float32(float(scales["q"]) * float(scales["k"]))
    )
    probability = bf16(softmax(logits))
    v_dequantized = vi.astype(np.float32) * np.float32(float(scales["v"]))
    return probability, vi, v_dequantized, target


def select_probability_scales(
    pairs: list[tuple[Path, Path]],
    contract: dict,
    layer: int,
    rows: np.ndarray,
    query_chunk_ids: np.ndarray,
) -> tuple[dict[int, list[float]], dict[int, dict[int, list[float]]]]:
    """Select scales by additive AV reconstruction, never output amplitude."""
    scales = candidate_scales()
    heads = contract["encoder"][layer]["attention"]["heads"]
    host_heads = set(contract["encoder"][layer].get("host_attention_heads", []))
    query_sse = {
        head: np.zeros((len(QUERY_CHUNKS), len(scales)), dtype=np.float64)
        for head in range(len(heads)) if head not in host_heads
    }
    partition_sse = {
        parts: {
            head: np.zeros((parts, len(scales)), dtype=np.float64)
            for head in range(len(heads)) if head not in host_heads
        }
        for parts in (2, 4)
    }
    for head, specification in enumerate(heads):
        if head in host_heads:
            continue
        for trace_path, reference_path in pairs:
            probability, vi, v_dequantized, _ = load_head(
                trace_path, reference_path, layer, head, rows,
                specification["scales_bf16"],
            )
            v_scale = float(specification["scales_bf16"]["v"])
            exact = bf16(probability @ v_dequantized)
            for candidate_index, scale in enumerate(scales.values()):
                code = floor_codes(probability, scale)
                actual = bf16(
                    (code @ vi).astype(np.float32)
                    * np.float32(scale * v_scale)
                )
                delta_sq = np.square(actual - exact, dtype=np.float32).astype(
                    np.float64
                )
                for chunk in range(len(QUERY_CHUNKS)):
                    query_sse[head][chunk, candidate_index] += float(
                        np.sum(delta_sq[query_chunk_ids == chunk])
                    )
            for parts in (2, 4):
                for part, (start, stop) in enumerate(
                    key_ranges(probability.shape[1], parts)
                ):
                    exact_partial = bf16(
                        probability[:, start:stop] @ v_dequantized[start:stop]
                    )
                    for candidate_index, scale in enumerate(scales.values()):
                        code = floor_codes(probability[:, start:stop], scale)
                        actual_partial = bf16(
                            (code @ vi[start:stop]).astype(np.float32)
                            * np.float32(scale * v_scale)
                        )
                        delta = actual_partial.astype(np.float64) - exact_partial
                        partition_sse[parts][head][part, candidate_index] += float(
                            np.sum(delta * delta)
                        )
    scale_values = list(scales.values())
    selected_query = {
        head: [scale_values[int(np.argmin(row))] for row in values]
        for head, values in query_sse.items()
    }
    selected_partition = {
        parts: {
            head: [scale_values[int(np.argmin(row))] for row in values]
            for head, values in heads_sse.items()
        }
        for parts, heads_sse in partition_sse.items()
    }
    return selected_query, selected_partition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = paired_paths(args.trace_dir, args.reference_dir)
    required = args.train_count + args.validation_count
    if len(pairs) < required:
        raise ValueError(f"need {required} trace pairs, found {len(pairs)}")
    training = pairs[:args.train_count]
    validation = pairs[args.train_count:required]
    contract = json.loads(args.contract.read_text())
    layer_record = contract["encoder"][args.layer]
    heads = layer_record["attention"]["heads"]
    host_heads = set(layer_record.get("host_attention_heads", []))
    rows = query_rows(1370, args.rows_per_chunk)
    query_chunk_ids = chunk_id_for_rows(rows)
    selected_query, selected_partition = select_probability_scales(
        training, contract, args.layer, rows, query_chunk_ids
    )

    scheme_names = (
        "current_single_a8",
        "dynamic_row_max_a8",
        "query_chunk_single_a8",
        "key_partition_2way_a8",
        "key_partition_4way_a8",
        "radix_2way_a8",
        "radix_3way_a8",
    )
    metrics = {
        split: {
            scheme: {
                "probability": Metric(),
                "attention_vs_quantized_boundary": Metric(),
                "attention_vs_fp32": Metric(),
                "stats": empty_stats(),
            }
            for scheme in scheme_names
        }
        for split in ("training", "validation")
    }

    for split, split_pairs in (("training", training), ("validation", validation)):
        for head, specification in enumerate(heads):
            if head in host_heads:
                continue
            scales = specification["scales_bf16"]
            if float(scales.get("av_output_gain", 1.0)) != 1.0:
                raise ValueError(f"layer {args.layer} head {head}: forbidden AV gain")
            if float(scales.get("av_v", scales["v"])) != float(scales["v"]):
                raise ValueError(f"layer {args.layer} head {head}: AV scale differs from V")
            for trace_path, reference_path in split_pairs:
                probability, vi, v_dequantized, target = load_head(
                    trace_path, reference_path, args.layer, head, rows, scales
                )
                boundary_target = bf16(probability @ v_dequantized)
                encodings: dict[str, tuple[list[np.ndarray], list[float], np.ndarray]] = {}

                current_scale = float(scales["probability"])
                current_code = floor_codes(probability, current_scale)
                encodings["current_single_a8"] = (
                    [current_code], [current_scale],
                    current_code.astype(np.float32) * np.float32(current_scale),
                )

                dynamic_code, dynamic_scales = dynamic_row_max_codes(probability)
                encodings["dynamic_row_max_a8"] = (
                    [dynamic_code], [dynamic_scales],
                    dynamic_code.astype(np.float32) * dynamic_scales,
                )

                query_dequantized = np.zeros_like(probability)
                query_codes = []
                for chunk, scale in enumerate(selected_query[head]):
                    mask = query_chunk_ids == chunk
                    code = floor_codes(probability[mask], scale)
                    query_codes.append(code)
                    query_dequantized[mask] = (
                        code.astype(np.float32) * np.float32(scale)
                    )
                encodings["query_chunk_single_a8"] = (
                    query_codes, selected_query[head], query_dequantized
                )

                for parts in (2, 4):
                    partition_dequantized = np.zeros_like(probability)
                    partition_codes = []
                    partition_scales = selected_partition[parts][head]
                    for part, (start, stop) in enumerate(
                        key_ranges(probability.shape[1], parts)
                    ):
                        scale = partition_scales[part]
                        code = floor_codes(probability[:, start:stop], scale)
                        partition_codes.append(code)
                        partition_dequantized[:, start:stop] = (
                            code.astype(np.float32) * np.float32(scale)
                        )
                    encodings[f"key_partition_{parts}way_a8"] = (
                        partition_codes, partition_scales, partition_dequantized
                    )

                for routes in (2, 3):
                    route_codes, route_scales = radix_codes(probability, routes)
                    route_dequantized = sum(
                        code.astype(np.float32) * np.float32(scale)
                        for code, scale in zip(route_codes, route_scales)
                    )
                    encodings[f"radix_{routes}way_a8"] = (
                        route_codes, route_scales, route_dequantized
                    )

                for scheme, (codes, route_scales, dequantized) in encodings.items():
                    record = metrics[split][scheme]
                    record["probability"].add(dequantized, probability)
                    add_probability_stats(record["stats"], codes, dequantized)
                    if scheme.startswith("key_partition_"):
                        parts = len(codes)
                        partials = []
                        for (start, stop), code, scale in zip(
                            key_ranges(probability.shape[1], parts),
                            codes,
                            route_scales,
                        ):
                            partials.append(bf16(
                                (code @ vi[start:stop]).astype(np.float32)
                                * np.float32(scale * float(scales["v"]))
                            ))
                        actual = bf16(sum(partials))
                    elif scheme.startswith("radix_"):
                        partials = [
                            bf16(
                                (code @ vi).astype(np.float32)
                                * np.float32(scale * float(scales["v"]))
                            )
                            for code, scale in zip(codes, route_scales)
                        ]
                        actual = bf16(sum(partials))
                    else:
                        actual = bf16(dequantized @ v_dequantized)
                    record["attention_vs_quantized_boundary"].add(
                        actual, boundary_target
                    )
                    record["attention_vs_fp32"].add(actual, target)

    result = {
        "schema_version": 1,
        "layer": args.layer,
        "objective": (
            "additive AV reconstruction; V unchanged; unit AV gain; "
            "non-overlapping or residual-additive routes only"
        ),
        "probability_rounding": (
            "static/radix routes use floor; SPU dynamic row-max uses "
            "round-to-nearest as verified against U250"
        ),
        "candidate_denominators": list(DENOMINATORS),
        "sampled_query_rows": rows.tolist(),
        "training_samples": [str(path.relative_to(args.trace_dir)) for path, _ in training],
        "validation_samples": [str(path.relative_to(args.trace_dir)) for path, _ in validation],
        "host_heads_excluded": sorted(host_heads),
        "selected_query_chunk_scales": selected_query,
        "selected_key_partition_scales": selected_partition,
        "schemes": {},
        "hardware_status": {
            "current_single_a8": "implemented",
            "dynamic_row_max_a8": (
                "implemented by SPU dynamic INT8 and instruction-level U250 qualified"
            ),
            "query_chunk_single_a8": "compiler-compatible; requires per-call kernel variants",
            "key_partition_2way_a8": (
                "compiler-compatible with two zero-masked V inputs and additive "
                "AV accumulation; approximately 2x QK/Softmax/AV"
            ),
            "key_partition_4way_a8": (
                "compiler-compatible with four zero-masked V inputs and additive "
                "AV accumulation; approximately 4x QK/Softmax/AV"
            ),
            "radix_2way_a8": "accuracy ceiling; requires on-device residual quantization",
            "radix_3way_a8": "accuracy ceiling; requires on-device residual quantization",
        },
    }
    for split in ("training", "validation"):
        result["schemes"][split] = {}
        for scheme in scheme_names:
            record = metrics[split][scheme]
            result["schemes"][split][scheme] = {
                "probability": record["probability"].result(),
                "attention_vs_quantized_boundary": (
                    record["attention_vs_quantized_boundary"].result()
                ),
                "attention_vs_fp32": record["attention_vs_fp32"].result(),
                **finish_stats(record["stats"]),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for scheme, record in result["schemes"]["validation"].items():
        print(
            f"{scheme:28s} "
            f"P={record['probability']['relative_l2']:.6f} "
            f"AV-boundary={record['attention_vs_quantized_boundary']['relative_l2']:.6f} "
            f"AV-fp32={record['attention_vs_fp32']['relative_l2']:.6f} "
            f"zero={record['logical_zero_fraction']:.4f}"
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
