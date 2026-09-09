#!/usr/bin/env python3
"""Offline precision gate for fused decoder LayerNorm/layout/project stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


LAYERS = (2, 5, 8, 11)


def attr(node: onnx.NodeProto, name: str) -> object:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    raise KeyError(f"{node.name}: missing {name}")


def bf16(value: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(value, dtype=np.float32)
    bits = data.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def quantize(value: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    return np.clip(np.rint(value / scale), -128, 127).astype(np.int8)


def quantized_linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                     input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    qx = quantize(value, input_scale).astype(np.int32)
    qw = quantize(weight, weight_scales[:, None]).astype(np.int32)
    accum = qx @ qw.T
    return (accum.astype(np.float32)
            * (np.float32(input_scale) * weight_scales[None, :])
            + bias[None, :])


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    left = candidate.astype(np.float64).reshape(-1)
    right = reference.astype(np.float64).reshape(-1)
    delta = left - right
    denom = max(float(np.linalg.norm(right)), np.finfo(np.float64).tiny)
    cosine_denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)),
                       np.finfo(np.float64).tiny)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_l2": float(np.linalg.norm(delta) / denom),
        "cosine": float(np.dot(left, right) / cosine_denom),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stem-manifest", type=Path, required=True)
    parser.add_argument("--capture-npz", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--max-degradation-ratio", type=float, default=1.25)
    args = parser.parse_args()

    model = onnx.load(str(args.model), load_external_data=True)
    nodes = {node.name: node for node in model.graph.node}
    values = {item.name: numpy_helper.to_array(item).astype(np.float32)
              for item in model.graph.initializer}
    manifest = json.loads(args.stem_manifest.read_text())
    affine_mode = manifest.get("affine_mode", "fold")
    stem_input = manifest.get("stem_input", "capture")
    stem_scales = {}
    for item in manifest["kernels"]:
        stem_scales.setdefault(int(item["project_index"]),
                               float(item["project_input_scale"]))

    cases = []
    qualified = True
    for capture_path in args.capture_npz:
        with np.load(capture_path, allow_pickle=False) as archive:
            captures = {layer: np.asarray(archive[f"capture_l{layer:02d}"],
                                          dtype=np.float32)
                        for layer in LAYERS}
        for index, layer in enumerate(LAYERS):
            x = captures[layer]
            mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
            variance = np.mean(np.square(x - mean), axis=-1,
                               keepdims=True, dtype=np.float32)
            core = (x - mean) / np.sqrt(variance + np.float32(1.0e-6))
            gamma = values["pretrained.norm.weight"]
            beta = values["pretrained.norm.bias"]
            affine = core * gamma + beta
            conv = nodes[f"/depth_head/projects.{index}/Conv"]
            weight4 = values[conv.input[1]]
            weight = weight4[:, :, 0, 0]
            bias = values[conv.input[2]]
            original_scales = np.asarray(attr(conv, "weight_ch_scales"),
                                         dtype=np.float32)
            legacy = quantized_linear(
                affine[:, 1:].reshape(-1, affine.shape[-1]), weight, bias,
                float(attr(conv, "input_scales")[0]), original_scales,
            )
            if stem_input in ("normalized", "patches"):
                fused = quantized_linear(
                    bf16(affine)[:, 1:].reshape(-1, core.shape[-1]),
                    weight, bias, stem_scales[index], original_scales,
                )
            elif affine_mode == "epu":
                epu_affine = bf16(bf16(core) * bf16(gamma))
                epu_affine = bf16(epu_affine + bf16(beta))
                fused = quantized_linear(
                    epu_affine[:, 1:].reshape(-1, core.shape[-1]),
                    weight, bias, stem_scales[index], original_scales,
                )
            else:
                folded_weight = weight * gamma[None, :]
                folded_bias = bias + weight @ beta
                folded_scales = np.maximum(
                    np.max(np.abs(folded_weight), axis=1) / 127.0,
                    np.finfo(np.float32).tiny,
                )
                fused = quantized_linear(
                    bf16(core)[:, 1:].reshape(-1, core.shape[-1]),
                    folded_weight, folded_bias, stem_scales[index], folded_scales,
                )
            reference = (affine[:, 1:].reshape(-1, affine.shape[-1])
                         @ weight.T + bias)
            legacy_metrics = metrics(legacy, reference)
            fused_metrics = metrics(fused, reference)
            ratio = fused_metrics["relative_l2"] / max(
                legacy_metrics["relative_l2"], np.finfo(np.float64).tiny)
            passed = (fused_metrics["relative_l2"] <= args.max_relative_l2
                      and ratio <= args.max_degradation_ratio)
            qualified &= passed
            cases.append({
                "capture": capture_path.name, "layer": layer,
                "project_index": index, "legacy": legacy_metrics,
                "fused": fused_metrics, "relative_l2_degradation_ratio": ratio,
                "qualified": passed,
            })
    report = {
        "schema_version": 1, "model": str(args.model.resolve()),
        "stem_manifest": str(args.stem_manifest.resolve()),
        "affine_mode": affine_mode,
        "stem_input": stem_input,
        "captures": [str(path.resolve()) for path in args.capture_npz],
        "bf16_layernorm_core_simulated": stem_input == "capture",
        "limits": {"max_relative_l2": args.max_relative_l2,
                   "max_degradation_ratio": args.max_degradation_ratio},
        "qualified": bool(qualified), "cases": cases,
        "worst_fused_relative_l2": max(x["fused"]["relative_l2"] for x in cases),
        "worst_degradation_ratio": max(x["relative_l2_degradation_ratio"]
                                        for x in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "qualified", "worst_fused_relative_l2", "worst_degradation_ratio"
    )}, sort_keys=True))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
