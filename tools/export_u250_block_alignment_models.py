#!/usr/bin/env python3
"""Restore one encoder block's FP32 QKV/attention amplitude contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def replace_initializer(
    model: onnx.ModelProto, name: str, value: np.ndarray
) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        replacement = numpy_helper.from_array(
            np.ascontiguousarray(value, dtype=np.float32), name=name
        )
        model.graph.initializer[index].CopyFrom(replacement)
        return
    raise ValueError(f"initializer not found: {name}")


def initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    for item in model.graph.initializer:
        if item.name == name:
            return numpy_helper.to_array(item).astype(np.float32, copy=False)
    raise ValueError(f"initializer not found: {name}")


def folded_qkv(
    state: dict[str, torch.Tensor], layer: int, heads: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prefix = f"pretrained.blocks.{layer}"
    gamma = state[f"{prefix}.norm1.weight"].cpu().numpy()
    beta = state[f"{prefix}.norm1.bias"].cpu().numpy()
    weight = state[f"{prefix}.attn.qkv.weight"].cpu().numpy()
    bias = state[f"{prefix}.attn.qkv.bias"].cpu().numpy()
    channels = gamma.size
    if channels % heads:
        raise ValueError("embedding channels are not divisible by head count")
    # DINOv2 applies the scaled-dot-product factor to Q.  The divisor is the
    # per-head width, not the full embedding width.
    q_scale = (channels // heads) ** -0.5
    result = {}
    for index, name in enumerate(("q", "k", "v")):
        begin, end = index * channels, (index + 1) * channels
        source_weight = weight[begin:end]
        factor = q_scale if name == "q" else 1.0
        # ONNX MatMul stores [input, output], while PyTorch Linear stores
        # [output, input].  LayerNorm gamma/beta are folded into the kernel.
        folded_weight = (source_weight * gamma[None, :] * factor).T
        folded_bias = (bias[begin:end] + source_weight @ beta) * factor
        result[name] = (
            np.ascontiguousarray(folded_weight, dtype=np.float32),
            np.ascontiguousarray(folded_bias, dtype=np.float32),
        )
    return result


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual.astype(np.float64) - expected.astype(np.float64))
        / max(np.linalg.norm(expected.astype(np.float64)), 1e-30)
    )


def head_gains(actual: np.ndarray, expected: np.ndarray, heads: int) -> list[float]:
    channels = expected.shape[1]
    if channels % heads:
        raise ValueError("QKV output channels are not divisible by head count")
    width = channels // heads
    gains = []
    for head in range(heads):
        begin, end = head * width, (head + 1) * width
        reference = expected[:, begin:end].astype(np.float64).reshape(-1)
        value = actual[:, begin:end].astype(np.float64).reshape(-1)
        gains.append(float(value @ reference / max(reference @ reference, 1e-30)))
    return gains


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qkv-model", type=Path, required=True)
    parser.add_argument("--attention-model-dir", type=Path, required=True)
    parser.add_argument("--attention-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    attention_heads = contract["encoder"][args.layer]["attention"]["heads"]
    heads = len(attention_heads)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = folded_qkv(state, args.layer, heads)
    qkv = onnx.load(str(args.qkv_model), load_external_data=True)
    prefix = f"/blocks.{args.layer}/attn/qkv"
    audit: dict[str, object] = {"layer": args.layer}
    for name in ("q", "k"):
        expected_weight, expected_bias = expected[name]
        weight_error = relative_l2(
            initializer(qkv, f"{prefix}/{name}/weight"), expected_weight
        )
        bias_error = relative_l2(
            initializer(qkv, f"{prefix}/{name}/bias"), expected_bias
        )
        if weight_error > 1e-6 or bias_error > 1e-6:
            raise ValueError(
                f"{name.upper()} is not the reference folded projection: "
                f"weight={weight_error}, bias={bias_error}"
            )
        audit[f"{name}_weight_relative_l2_before"] = weight_error
        audit[f"{name}_bias_relative_l2_before"] = bias_error

    v_weight, v_bias = expected["v"]
    old_v_weight = initializer(qkv, f"{prefix}/v/weight")
    old_v_bias = initializer(qkv, f"{prefix}/v/bias")
    audit.update({
        "v_weight_relative_l2_before": relative_l2(old_v_weight, v_weight),
        "v_bias_relative_l2_before": relative_l2(old_v_bias, v_bias),
        "v_weight_head_gains_before": head_gains(old_v_weight, v_weight, heads),
    })
    replace_initializer(qkv, f"{prefix}/v/weight", v_weight)
    replace_initializer(qkv, f"{prefix}/v/bias", v_bias)

    qkv_dir = args.output_dir / "qkv"
    attention_dir = args.output_dir / "attention"
    qkv_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
    qkv_path = qkv_dir / args.qkv_model.name
    onnx.save(qkv, qkv_path)

    manifest = json.loads(args.attention_manifest.read_text())
    records = [
        item for item in manifest["kernels"]
        if int(item["layer"]) == args.layer
    ]
    if len(records) != heads:
        raise ValueError(f"expected {heads} attention records, found {len(records)}")
    output_records = []
    old_av_gains = []
    for record in sorted(records, key=lambda item: int(item["head"])):
        head = int(record["head"])
        specification = attention_heads[head]
        scales = copy.deepcopy(specification["scales_bf16"])
        v_scale = float(scales["v"])
        old_av_gains.append(float(scales.get("av_output_gain", 1.0)))
        scales["av_output_gain"] = 1.0
        scales["av_v"] = v_scale
        model = onnx.load(str(args.attention_model_dir / record["onnx"]))
        changed = 0
        for node in model.graph.node:
            if node.op_type == "MatMul" and node.name.endswith("/AV"):
                replace_attribute(node, "B_scales", [v_scale])
                changed += 1
        if changed != 2:
            raise ValueError(f"{record['name']}: expected two AV MatMuls, got {changed}")
        output_path = attention_dir / record["onnx"]
        onnx.save(model, output_path)
        output_record = copy.deepcopy(record)
        output_record["scales_bf16"] = scales
        output_records.append(output_record)
        specification["scales_bf16"] = copy.deepcopy(scales)

    audit["attention_av_gains_before"] = old_av_gains
    audit["attention_av_gains_after"] = [1.0] * heads
    audit["v_weight_relative_l2_after"] = relative_l2(
        initializer(qkv, f"{prefix}/v/weight"), v_weight
    )
    audit["policy"] = "restore-reference-v-and-remove-av-amplitude-compensation"
    (qkv_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "kernel": f"qkv_projection_l{args.layer:02d}",
        "onnx": qkv_path.name,
        "audit": audit,
    }, indent=2, sort_keys=True) + "\n")
    (attention_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "strategy": "reference-aligned-unit-av-gain",
        "kernels": output_records,
        "kernels_total": len(output_records),
        "audit": audit,
    }, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "depthanything_u250_runtime_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
