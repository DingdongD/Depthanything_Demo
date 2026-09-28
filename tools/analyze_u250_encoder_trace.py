#!/usr/bin/env python3
"""Attribute U250 encoder error to QKV, attention, and projection stages."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--float-trace", type=Path, required=True)
    parser.add_argument("--float-posts", type=Path, required=True)
    parser.add_argument("--float-blocks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--host-params", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metrics(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    expected_norm = np.linalg.norm(expected)
    return {
        "cosine": float(np.dot(actual, expected) /
                        (np.linalg.norm(actual) * expected_norm)),
        "rel_l2": float(np.linalg.norm(actual - expected) / expected_norm),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
    }


def layer_norm(value, scale, bias, axis, epsilon):
    axes = tuple(range(axis if axis >= 0 else value.ndim + axis, value.ndim))
    mean = np.mean(value, axis=axes, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=axes, keepdims=True)
    return ((value - mean) / np.sqrt(variance + epsilon) * scale + bias).astype(np.float32)


def softmax(value):
    value = value - np.max(value, axis=-1, keepdims=True)
    result = np.exp(value)
    return result / np.sum(result, axis=-1, keepdims=True)


def main():
    args = parse_args()
    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
    model = onnx.load(str(args.model), load_external_data=True)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    nodes = {node.name: node for node in model.graph.node}
    with np.load(args.host_params) as archive:
        host_params = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.input) as archive:
        model_input = np.asarray(archive[archive.files[0]], dtype=np.float32)

    report = {"layers": []}
    with (np.load(args.trace) as trace,
          np.load(args.float_trace) as float_trace,
          np.load(args.float_posts) as float_posts,
          np.load(args.float_blocks) as float_blocks):
        for layer, (block, norm_specs) in enumerate(zip(contract["encoder"], plan["encoder"])):
            x = model_input if layer == 0 else np.asarray(trace[f"block_l{layer - 1:02d}"], dtype=np.float32)
            norm1 = norm_specs["norm1"]
            normalized = layer_norm(x, host_params[norm1["scale"]],
                                    host_params[norm1["bias"]], norm1["axis"],
                                    norm1["epsilon"])
            qkv_local = {}
            qkv_dequant = {}
            qkv_metrics = {}
            for short in ("q", "k", "v"):
                matmul = nodes[f"/blocks.{layer}/attn/qkv/{short}/MatMul"]
                add = nodes[f"/blocks.{layer}/attn/qkv/{short}/Add"]
                expected = normalized @ initializers[matmul.input[1]] + initializers[add.input[0]]
                scale = float(block["qkv"]["output_scales"][short])
                code = np.asarray(trace[f"{short}_l{layer:02d}"])
                actual = code[:, 0].astype(np.float32) * scale
                qkv_local[short] = expected
                qkv_dequant[short] = actual
                qkv_metrics[short] = {
                    "local": metrics(actual, expected),
                    "end_to_end": metrics(actual, float_trace[f"{short}_l{layer:02d}"]),
                    "saturation_percent": float(np.mean((code <= -128) | (code >= 127)) * 100.0),
                }

            exact_heads = []
            quantized_heads = []
            head_reports = []
            for head, head_spec in enumerate(block["attention"]["heads"]):
                begin = head * 64
                end = begin + 64
                q = qkv_dequant["q"][0, :, begin:end]
                k = qkv_dequant["k"][0, :, begin:end]
                v = qkv_dequant["v"][0, :, begin:end]
                # The original 1/sqrt(64) factor is already folded into the Q
                # projection weights by fuse_attention_projection_a8.py.
                probability = softmax(q @ k.T).astype(np.float32)
                exact_heads.append(probability @ v)
                probability_scale = float(head_spec["scales_bf16"]["probability"])
                probability_code = np.clip(np.rint(probability / probability_scale), 0, 127)
                quantized_heads.append((probability_code * probability_scale) @ v)
                head_reports.append({
                    "head": head,
                    "probability_scale": probability_scale,
                    "probability_max": float(probability.max()),
                    "probability_saturation_percent": float(np.mean(probability_code >= 127) * 100.0),
                })
            exact_attention = np.concatenate(exact_heads, axis=-1)[None]
            quantized_attention = np.concatenate(quantized_heads, axis=-1)[None]
            board_attention = np.asarray(trace[f"attention_l{layer:02d}"])[:, 0]

            projection = nodes[f"/blocks.{layer}/attn/proj/MatMul"]
            projection_add = nodes[f"/blocks.{layer}/attn/proj/Add"]
            layer_scale = nodes[f"/blocks.{layer}/ls1/Mul"]
            projected = (board_attention @ initializers[projection.input[1]] +
                         initializers[projection_add.input[0]])
            expected_post = projected * initializers[layer_scale.input[1]] + x
            board_post = np.asarray(trace[f"post_l{layer:02d}"])
            row = {
                "layer": layer,
                "qkv": qkv_metrics,
                "attention_vs_local_exact": metrics(board_attention, exact_attention),
                "attention_vs_probability_int8_simulation": metrics(board_attention, quantized_attention),
                "attention_end_to_end": metrics(board_attention, float_trace[f"attention_l{layer:02d}"]),
                "post_projection_local": metrics(board_post, expected_post),
                "post_end_to_end": metrics(board_post, float_posts[f"post_l{layer:02d}"]),
                "block_end_to_end": metrics(trace[f"block_l{layer:02d}"],
                                            float_blocks[f"block_l{layer:02d}"]),
                "heads": head_reports,
            }
            report["layers"].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("layer qkv_local_cos(q/k/v) att_local_cos att_sim_cos post_local_cos post_e2e_cos block_e2e_cos")
    for row in report["layers"]:
        qkv = "/".join(f"{row['qkv'][key]['local']['cosine']:.5f}" for key in ("q", "k", "v"))
        print(f"{row['layer']:02d} {qkv} "
              f"{row['attention_vs_local_exact']['cosine']:.5f} "
              f"{row['attention_vs_probability_int8_simulation']['cosine']:.5f} "
              f"{row['post_projection_local']['cosine']:.5f} "
              f"{row['post_end_to_end']['cosine']:.5f} "
              f"{row['block_end_to_end']['cosine']:.5f}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
