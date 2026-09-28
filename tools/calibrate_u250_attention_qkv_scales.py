#!/usr/bin/env python3
"""Coordinate-search per-head Q/K/V A8 scales with unit-gain dual attention."""

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
from calibrate_u250_dual_range_attention import dual_range_context


def bf16_scalar(value: float) -> float:
    return float(bf16(np.asarray([value], dtype=np.float32))[0])


def candidates(current: float, count: int) -> list[float]:
    return sorted({
        bf16_scalar(value)
        for value in np.geomspace(current * 0.5, current * 2.0, count)
    } | {float(current)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=25)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = paired_paths(args.trace_dir, args.reference_dir)
    required = args.train_count + args.validation_count
    if len(pairs) < required:
        raise ValueError(f"need {required} trace pairs, found {len(pairs)}")
    split_pairs = {
        "training": pairs[:args.train_count],
        "validation": pairs[args.train_count:required],
    }
    contract = json.loads(args.contract.read_text())
    attention = contract["encoder"][args.layer]["attention"]
    probability = attention.get("dual_range_probability")
    if probability is not None and probability.get("av_output_gain") != 1.0:
        raise ValueError("calibration requires unit-gain dual-range attention")
    first_trace, _ = split_pairs["training"][0]
    with np.load(first_trace, allow_pickle=False) as trace:
        token_count = int(trace[f"q_l{args.layer:02d}"].shape[-2])
    rows = query_rows(token_count, args.rows_per_chunk)
    head_reports = []
    for head, specification in enumerate(attention["heads"]):
        scales = specification["scales_bf16"]
        if float(scales.get("av_output_gain", 1.0)) != 1.0:
            raise ValueError(f"head {head}: forbidden AV gain")
        if float(scales.get("av_v", scales["v"])) != float(scales["v"]):
            raise ValueError(f"head {head}: AV scale differs from V")
        if probability is not None:
            params = probability["heads"][str(head)]
            threshold = float(params["threshold"])
            residual_step = float(params["residual_step"])
        else:
            # Newer shape-specialized contracts keep the dual-range
            # probability parameters beside each head's Q/K/V scales.
            params = scales.get("probability")
            if not params:
                raise ValueError(f"head {head}: missing dual-range probability")
            threshold = float(params["threshold"])
            residual_step = float(params.get("residual_step", params["residual"]))
        begin, end = 64 * head, 64 * (head + 1)
        cache = {}
        for split, entries in split_pairs.items():
            values = []
            for trace_path, reference_path in entries:
                with np.load(trace_path, allow_pickle=False) as trace, np.load(
                    reference_path, allow_pickle=False
                ) as reference:
                    values.append({
                        "q": np.asarray(
                            trace[f"q_l{args.layer:02d}"][0, 0, rows, begin:end],
                            dtype=np.float32,
                        ),
                        "k": np.asarray(
                            trace[f"k_l{args.layer:02d}"][0, 0, :, begin:end],
                            dtype=np.float32,
                        ),
                        "v": np.asarray(
                            trace[f"v_l{args.layer:02d}"][0, 0, :, begin:end],
                            dtype=np.float32,
                        ),
                        "target": np.asarray(
                            reference[f"encoder_l{args.layer:02d}_attention"][
                                0, 0, rows, begin:end
                            ],
                            dtype=np.float32,
                        ),
                    })
            cache[split] = values

        def evaluate(candidate_scales: dict[str, float], split: str) -> dict:
            metric = Metric()
            for sample in cache[split]:
                q_code = quantize(sample["q"], candidate_scales["q"]).astype(np.int32)
                k_code = quantize(sample["k"], candidate_scales["k"]).astype(np.int32)
                v_code = quantize(sample["v"], candidate_scales["v"]).astype(np.int32)
                logits = bf16(
                    (q_code @ k_code.T).astype(np.float32)
                    * np.float32(candidate_scales["q"] * candidate_scales["k"])
                )
                context, _ = dual_range_context(
                    bf16(softmax(logits)), v_code, candidate_scales["v"],
                    threshold, residual_step,
                )
                metric.add(context, sample["target"])
            return metric.result()

        selected = {name: float(scales[name]) for name in ("q", "k", "v")}
        baseline = {
            split: evaluate(selected, split) for split in split_pairs
        }
        steps = []
        for pass_index in range(args.passes):
            for kind in ("q", "k", "v"):
                records = []
                for scale in candidates(float(scales[kind]), args.candidate_count):
                    trial = {**selected, kind: scale}
                    records.append({
                        "scale": scale,
                        "training": evaluate(trial, "training"),
                    })
                choice = min(
                    records,
                    key=lambda item: item["training"]["relative_l2"],
                )
                selected[kind] = float(choice["scale"])
                steps.append({
                    "pass": pass_index,
                    "kind": kind,
                    "selected_scale": selected[kind],
                    "training": choice["training"],
                    "validation": evaluate(selected, "validation"),
                })
        final = {split: evaluate(selected, split) for split in split_pairs}
        head_reports.append({
            "head": head,
            "old_scales_bf16": {
                name: float(scales[name]) for name in ("q", "k", "v")
            },
            "selected_scales_bf16": selected,
            "baseline": baseline,
            "selected": final,
            "steps": steps,
        })
        print(
            f"head {head}: {baseline['validation']['relative_l2']:.6f} -> "
            f"{final['validation']['relative_l2']:.6f} "
            f"q={selected['q']:.9g} k={selected['k']:.9g} "
            f"v={selected['v']:.9g}"
        )

    report = {
        "schema": "depthanything-u250-attention-qkv-scale-calibration-v1",
        "layer": args.layer,
        "objective": "FP32 raw attention under unit-gain dual-range AV",
        "av_output_gain": 1.0,
        "v_unchanged": True,
        "training_samples": [
            str(path.relative_to(args.trace_dir))
            for path, _ in split_pairs["training"]
        ],
        "validation_samples": [
            str(path.relative_to(args.trace_dir))
            for path, _ in split_pairs["validation"]
        ],
        "sampled_query_rows": rows.tolist(),
        "token_count": token_count,
        "heads": head_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
