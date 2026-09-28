#!/usr/bin/env python3
"""Retune per-layer/head INT8 Softmax scales from a real U250 trace."""

import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


DENOMINATORS = (127, 255, 511, 1023, 2047, 4095, 8191, 16383)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path, action="append", required=True,
        help="U250 output NPZ containing q_lXX/k_lXX/v_lXX; repeat for a set",
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--attention-manifest", type=Path, required=True)
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        help="also evaluate these fixed probability scales/gains without selecting them",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--probability-rounding", choices=("round", "floor"), default="round",
        help="match the SPU's output quantization rule; U250 probes use floor",
    )
    parser.add_argument(
        "--selection-objective", choices=("raw_sse", "compensated_sse"),
        default="raw_sse",
        help=("select probability scale directly, or jointly select it with a "
              "least-squares AV output gain"),
    )
    parser.add_argument(
        "--gain-min", type=float, default=0.5,
        help="lower bound for the folded AV output gain",
    )
    parser.add_argument(
        "--gain-max", type=float, default=2.0,
        help="upper bound for the folded AV output gain",
    )
    return parser.parse_args()


def bf16_round(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return np.asarray(rounded & np.uint32(0xFFFF0000), dtype=np.uint32).reshape(
        -1).view(np.float32).reshape(value.shape)


def bf16_scalar(value):
    return float(bf16_round(np.asarray(value, dtype=np.float32)))


def softmax(value):
    value = value - np.max(value, axis=-1, keepdims=True)
    result = np.exp(value)
    return result / np.sum(result, axis=-1, keepdims=True)


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    return {
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * np.linalg.norm(expected))),
        "rel_l2": float(np.linalg.norm(actual - expected) /
                        np.linalg.norm(expected)),
    }


def optimal_gain(actual, expected, lower, upper):
    """Return a BF16-representable least-squares gain for actual -> expected."""
    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected64 = np.asarray(expected, dtype=np.float64).reshape(-1)
    denominator = float(np.dot(actual64, actual64))
    gain = 1.0 if denominator == 0.0 else float(
        np.dot(actual64, expected64) / denominator)
    return bf16_scalar(np.clip(gain, lower, upper))


def main():
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.attention_manifest.read_text())
    evaluation_manifest = (json.loads(args.evaluation_manifest.read_text())
                           if args.evaluation_manifest else None)
    output_manifest = copy.deepcopy(manifest)
    scales = {denominator: bf16_scalar(1.0 / denominator)
              for denominator in DENOMINATORS}
    report = {
        "schema_version": 1,
        "source_traces": [str(path.resolve()) for path in args.trace],
        "candidate_denominators": list(DENOMINATORS),
        "selection": args.selection_objective,
        "probability_rounding": args.probability_rounding,
        "gain_bounds": [args.gain_min, args.gain_max],
        "layers": [],
    }

    with ExitStack() as stack:
        traces = [stack.enter_context(np.load(path)) for path in args.trace]
        for layer, block in enumerate(contract["encoder"]):
            output_layer = output_manifest["layers"][layer]
            qkv_scales = block["qkv"]["output_scales"]
            layer_report = {"layer": layer, "heads": []}
            exact_heads = []
            old_heads = []
            selected_heads = []
            evaluated_heads = []
            for head, head_spec in enumerate(block["attention"]["heads"]):
                begin = head * 64
                end = begin + 64
                q_scale = float(qkv_scales["q"])
                k_scale = float(qkv_scales["k"])
                v_scale = float(qkv_scales["v"])
                exact_samples = []
                candidate_samples = {denominator: [] for denominator in DENOMINATORS}
                saturation_samples = {denominator: [] for denominator in DENOMINATORS}
                probability_max = 0.0
                for trace in traces:
                    q_code = trace[f"q_l{layer:02d}"][0, 0, :, begin:end].astype(np.int32)
                    k_code = trace[f"k_l{layer:02d}"][0, 0, :, begin:end].astype(np.int32)
                    v_code = trace[f"v_l{layer:02d}"][0, 0, :, begin:end].astype(np.int32)
                    logits = bf16_round((q_code @ k_code.T).astype(np.float32) *
                                        np.float32(q_scale * k_scale))
                    probability = bf16_round(softmax(logits).astype(np.float32))
                    probability_max = max(probability_max, float(probability.max()))
                    exact_samples.append(
                        probability @ (v_code.astype(np.float32) * v_scale))
                    for denominator, probability_scale in scales.items():
                        scaled_probability = probability / probability_scale
                        if args.probability_rounding == "floor":
                            probability_code = np.floor(scaled_probability)
                        else:
                            probability_code = np.rint(scaled_probability)
                        probability_code = np.clip(
                            probability_code, 0, 127).astype(np.int32)
                        got = bf16_round(
                            (probability_code @ v_code).astype(np.float32) *
                            np.float32(probability_scale * v_scale))
                        candidate_samples[denominator].append(got)
                        saturation_samples[denominator].append(
                            float(np.mean(probability_code >= 127) * 100.0))
                exact = np.concatenate(exact_samples, axis=0)
                candidates = {}
                candidate_outputs = {}
                for denominator, probability_scale in scales.items():
                    got = np.concatenate(candidate_samples[denominator], axis=0)
                    gain = optimal_gain(got, exact, args.gain_min, args.gain_max)
                    compensated = bf16_round(got * np.float32(gain))
                    candidate_outputs[denominator] = got
                    candidates[str(denominator)] = {
                        **metrics(got, exact),
                        "squared_error": float(np.sum((got.astype(np.float64) - exact) ** 2)),
                        "av_output_gain_bf16": gain,
                        "av_value_accumulator_scale_bf16": bf16_scalar(v_scale * gain),
                        "compensated": metrics(compensated, exact),
                        "compensated_squared_error": float(np.sum(
                            (compensated.astype(np.float64) - exact) ** 2)),
                        "probability_scale_bf16": probability_scale,
                        "probability_saturation_percent": float(
                            np.mean(saturation_samples[denominator])),
                    }
                old_scale = float(head_spec["scales_bf16"]["probability"])
                old_denominator = min(DENOMINATORS,
                                      key=lambda item: abs(scales[item] - old_scale))
                objective_key = ("compensated_squared_error"
                                 if args.selection_objective == "compensated_sse"
                                 else "squared_error")
                selected_denominator = min(
                    DENOMINATORS, key=lambda item: candidates[str(item)][objective_key])
                selected_scale = scales[selected_denominator]
                selected_gain = (candidates[str(selected_denominator)]["av_output_gain_bf16"]
                                 if args.selection_objective == "compensated_sse" else 1.0)
                for triplet in output_layer["triplets"]:
                    if int(triplet["head"]) == head:
                        triplet["probability_scale"] = selected_scale
                        triplet["av_output_gain"] = selected_gain
                        triplet["av_value_accumulator_scale"] = bf16_scalar(
                            v_scale * selected_gain)
                exact_heads.append(exact)
                old_heads.append(candidate_outputs[old_denominator])
                selected_heads.append(bf16_round(
                    candidate_outputs[selected_denominator] * np.float32(selected_gain)))
                if evaluation_manifest is not None:
                    evaluation_triplet = evaluation_manifest["layers"][layer]["triplets"][head * 6]
                    evaluation_scale = float(evaluation_triplet["probability_scale"])
                    evaluation_denominator = min(
                        DENOMINATORS, key=lambda item: abs(scales[item] - evaluation_scale))
                    evaluation_gain = float(evaluation_triplet.get("av_output_gain", 1.0))
                    evaluated_heads.append(bf16_round(
                        candidate_outputs[evaluation_denominator]
                        * np.float32(evaluation_gain)))
                layer_report["heads"].append({
                    "head": head,
                    "old_denominator": old_denominator,
                    "selected_denominator": selected_denominator,
                    "old_probability_scale": scales[old_denominator],
                    "selected_probability_scale": selected_scale,
                    "selected_av_output_gain": selected_gain,
                    "probability_max": probability_max,
                    "candidates": candidates,
                })
            exact = np.concatenate(exact_heads, axis=-1)
            old = np.concatenate(old_heads, axis=-1)
            selected = np.concatenate(selected_heads, axis=-1)
            layer_report["old"] = metrics(old, exact)
            layer_report["selected"] = metrics(selected, exact)
            if evaluation_manifest is not None:
                evaluated = np.concatenate(evaluated_heads, axis=-1)
                layer_report["evaluation_manifest"] = metrics(evaluated, exact)
            report["layers"].append(layer_report)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("layer old_denominators selected_denominators old_cos selected_cos old_rel selected_rel")
    for layer in report["layers"]:
        old = [head["old_denominator"] for head in layer["heads"]]
        new = [head["selected_denominator"] for head in layer["heads"]]
        print(f"{layer['layer']:02d} {old} {new} "
              f"{layer['old']['cosine']:.6f} {layer['selected']['cosine']:.6f} "
              f"{layer['old']['rel_l2']:.6f} {layer['selected']['rel_l2']:.6f}")
    print(args.output_manifest)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
