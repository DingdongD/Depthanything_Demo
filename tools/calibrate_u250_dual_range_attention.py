#!/usr/bin/env python3
"""Calibrate the deployable fine-plus-residual A8 attention encoding.

The search preserves the mathematical amplitude: V is quantized with its
existing scale, AV gain is one, and the two AV results are added directly.
Training and validation samples are kept separate in the emitted report.
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


DEFAULT_THRESHOLD_DENOMINATORS = (
    64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512,
    640, 768, 896, 1024,
)
DEFAULT_RESIDUAL_MULTIPLIERS = (0.5, 1.0, 2.0)


def parse_csv(value: str, cast) -> tuple:
    result = tuple(cast(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def dual_range_context(
    probability: np.ndarray,
    v_code: np.ndarray,
    v_scale: float,
    threshold: float,
    residual_step: float,
) -> tuple[np.ndarray, dict[str, float]]:
    fine_step = threshold / 127.0
    fine_code = np.clip(
        np.floor(probability / np.float32(fine_step)), 0, 127
    ).astype(np.int32)
    residual_code = np.maximum(
        np.clip(
            np.rint((probability - np.float32(threshold)) / residual_step),
            -128,
            127,
        ).astype(np.int32),
        0,
    )
    fine_context = bf16(
        (fine_code @ v_code).astype(np.float32)
        * np.float32(fine_step * v_scale)
    )
    residual_context = bf16(
        (residual_code @ v_code).astype(np.float32)
        * np.float32(residual_step * v_scale)
    )
    represented = (
        fine_code.astype(np.float32) * np.float32(fine_step)
        + residual_code.astype(np.float32) * np.float32(residual_step)
    )
    return bf16(fine_context + residual_context), {
        "row_sum_mean": float(np.mean(np.sum(represented, axis=-1))),
        "row_sum_min": float(np.min(np.sum(represented, axis=-1))),
        "row_sum_max": float(np.max(np.sum(represented, axis=-1))),
        "fine_zero_fraction": float(np.mean(fine_code == 0)),
        "fine_saturation_fraction": float(np.mean(fine_code == 127)),
        "residual_zero_fraction": float(np.mean(residual_code == 0)),
        "residual_saturation_fraction": float(np.mean(residual_code == 127)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument(
        "--threshold-denominators",
        type=lambda value: parse_csv(value, int),
        default=DEFAULT_THRESHOLD_DENOMINATORS,
    )
    parser.add_argument(
        "--residual-multipliers",
        type=lambda value: parse_csv(value, float),
        default=DEFAULT_RESIDUAL_MULTIPLIERS,
    )
    parser.add_argument(
        "--objective", choices=("raw_attention", "probability_boundary"),
        default="raw_attention",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = paired_paths(args.trace_dir, args.reference_dir)
    required = args.train_count + args.validation_count
    if len(pairs) < required:
        raise ValueError(f"need {required} trace pairs, found {len(pairs)}")
    splits = {
        "training": pairs[:args.train_count],
        "validation": pairs[args.train_count:required],
    }
    contract = json.loads(args.contract.read_text())
    layer = contract["encoder"][args.layer]
    heads = layer["attention"]["heads"]
    host_heads = set(layer.get("host_attention_heads", []))
    first_trace, _ = splits["training"][0]
    with np.load(first_trace, allow_pickle=False) as trace:
        token_count = int(trace[f"q_l{args.layer:02d}"].shape[-2])
    rows = query_rows(token_count, args.rows_per_chunk)
    candidates = [
        {
            "threshold_denominator": denominator,
            "threshold": 1.0 / denominator,
            "fine_step": 1.0 / (127.0 * denominator),
            "residual_multiplier": multiplier,
            "residual_step": multiplier / denominator,
        }
        for denominator in args.threshold_denominators
        for multiplier in args.residual_multipliers
    ]
    # Always include the exact deployed carrier.  Shape-specialized packages
    # may use 127/16384 rather than a reciprocal-integer threshold, so using
    # only the search grid would make the baseline comparison approximate.
    for specification in heads:
        deployed = specification.get("scales_bf16", {}).get("probability")
        if not deployed:
            continue
        exact = {
            "threshold_denominator": None,
            "threshold": float(deployed["threshold"]),
            "fine_step": float(deployed["fine"]),
            "residual_multiplier": None,
            "residual_step": float(deployed["residual"]),
        }
        if not any(
            np.isclose(item["threshold"], exact["threshold"], rtol=0, atol=0)
            and np.isclose(item["residual_step"], exact["residual_step"], rtol=0, atol=0)
            for item in candidates
        ):
            candidates.append(exact)
    metrics = {
        split: {
            head: [
                {
                    "raw_attention": Metric(),
                    "probability_boundary": Metric(),
                    "stats": [],
                }
                for _ in candidates
            ]
            for head in range(len(heads)) if head not in host_heads
        }
        for split in splits
    }

    for split, split_pairs in splits.items():
        for head, specification in enumerate(heads):
            if head in host_heads:
                continue
            scales = specification["scales_bf16"]
            v_scale = float(scales["v"])
            if float(scales.get("av_output_gain", 1.0)) != 1.0:
                raise ValueError(f"layer {args.layer} head {head}: forbidden AV gain")
            if float(scales.get("av_v", v_scale)) != v_scale:
                raise ValueError(f"layer {args.layer} head {head}: AV scale differs from V")
            begin, end = 64 * head, 64 * (head + 1)
            for trace_path, reference_path in split_pairs:
                with np.load(trace_path) as trace, np.load(reference_path) as reference:
                    q = trace[f"q_l{args.layer:02d}"][0, 0, :, begin:end]
                    k = trace[f"k_l{args.layer:02d}"][0, 0, :, begin:end]
                    v = trace[f"v_l{args.layer:02d}"][0, 0, :, begin:end]
                    target = reference[f"encoder_l{args.layer:02d}_attention"][
                        0, 0, rows, begin:end
                    ]
                q_code = quantize(q[rows], float(scales["q"])).astype(np.int32)
                k_code = quantize(k, float(scales["k"])).astype(np.int32)
                v_code = quantize(v, v_scale).astype(np.int32)
                logits = bf16(
                    (q_code @ k_code.T).astype(np.float32)
                    * np.float32(float(scales["q"]) * float(scales["k"]))
                )
                probability = bf16(softmax(logits))
                boundary = bf16(
                    probability @ (v_code.astype(np.float32) * np.float32(v_scale))
                )
                for index, candidate in enumerate(candidates):
                    actual, stats = dual_range_context(
                        probability, v_code, v_scale,
                        candidate["threshold"], candidate["residual_step"],
                    )
                    metrics[split][head][index]["raw_attention"].add(actual, target)
                    metrics[split][head][index]["probability_boundary"].add(
                        actual, boundary
                    )
                    metrics[split][head][index]["stats"].append(stats)

    report_heads = []
    for head in sorted(metrics["training"]):
        records = []
        for index, candidate in enumerate(candidates):
            record = dict(candidate)
            for split in splits:
                entry = metrics[split][head][index]
                record[split] = {
                    "raw_attention": entry["raw_attention"].result(),
                    "probability_boundary": entry["probability_boundary"].result(),
                    "statistics": {
                        key: float(np.mean([item[key] for item in entry["stats"]]))
                        for key in entry["stats"][0]
                    },
                }
            records.append(record)
        selected = min(
            records,
            key=lambda item: item["training"][args.objective]["relative_l2"],
        )
        report_heads.append({"head": head, "selected": selected, "candidates": records})

    result = {
        "schema_version": 1,
        "layer": args.layer,
        "objective": args.objective,
        "v_unchanged": True,
        "av_output_gain": 1.0,
        "sampled_query_rows": rows.tolist(),
        "token_count": token_count,
        "training_samples": [str(path.relative_to(args.trace_dir)) for path, _ in splits["training"]],
        "validation_samples": [str(path.relative_to(args.trace_dir)) for path, _ in splits["validation"]],
        "heads": report_heads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for item in report_heads:
        chosen = item["selected"]
        print(
            f"head {item['head']}: threshold={chosen['threshold']:.9g} "
            f"residual={chosen['residual_step']:.9g} "
            f"train={chosen['training'][args.objective]['relative_l2']:.6f} "
            f"validation={chosen['validation'][args.objective]['relative_l2']:.6f}"
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
