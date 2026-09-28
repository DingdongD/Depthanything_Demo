#!/usr/bin/env python3
"""Make Q/K/V projection convolutions produce attention A8 tensors directly.

U250 cannot numerically convert a resident BF16 activation to A8 with the
current Transform/Stick instructions.  The projection MatMul+Add is lowered to
a CTC Conv2D, whose accumulator requantization is supported.  This pass:

* folds Q's shared attention scale into Q weights and bias;
* folds each head's calibrated AV gain into its V output channels;
* gives Q, K and V one conservative (maximum-head) A8 scale per layer;
* bypasses the now-folded Q/V Mul nodes and updates attention MatMul scales.

Weights and their per-channel scales are multiplied by the same factors, so
their A8 integer codes do not change.  The standard floating-point graph also
retains the exported model's semantics after the folded Mul nodes are bypassed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import helper, numpy_helper


BLOCK = re.compile(r"^/blocks\.(\d+)/attn/")


def get_attr(node: onnx.NodeProto, name: str):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    raise KeyError(f"{node.name}: missing {name}")


def set_attr(node: onnx.NodeProto, name: str, value) -> None:
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def constant_scalar(
    tensor: str,
    producer: dict[str, onnx.NodeProto],
    arrays: dict[str, np.ndarray],
) -> float:
    if tensor in arrays:
        return float(np.asarray(arrays[tensor]).reshape(()))
    node = producer.get(tensor)
    if node is None or node.op_type != "Constant":
        raise ValueError(f"{tensor!r} is not a scalar constant")
    value = get_attr(node, "value")
    if isinstance(value, onnx.TensorProto):
        value = numpy_helper.to_array(value)
    return float(np.asarray(value).reshape(()))


def projection_parts(
    branch: str,
    block: int,
    nodes: dict[str, onnx.NodeProto],
    arrays: dict[str, np.ndarray],
) -> tuple[onnx.NodeProto, onnx.NodeProto, str, str]:
    prefix = f"/blocks.{block}/attn/qkv/{branch}"
    add = nodes[prefix + "/Add"]
    matmul = nodes[prefix + "/MatMul"]
    weight_name = matmul.input[1]
    bias_names = [name for name in add.input if name in arrays]
    if weight_name not in arrays or len(bias_names) != 1:
        raise ValueError(f"{prefix}: malformed projection constants")
    return add, matmul, weight_name, bias_names[0]


def replace_array(
    model: onnx.ModelProto,
    index_by_name: dict[str, int],
    name: str,
    value: np.ndarray,
) -> None:
    replacement = numpy_helper.from_array(np.ascontiguousarray(value), name=name)
    model.graph.initializer[index_by_name[name]].CopyFrom(replacement)


def rewrite(model: onnx.ModelProto) -> dict:
    producer = {output: node for node in model.graph.node for output in node.output}
    nodes = {node.name: node for node in model.graph.node}
    arrays = {
        value.name: numpy_helper.to_array(value).copy()
        for value in model.graph.initializer
    }
    index_by_name = {
        value.name: index for index, value in enumerate(model.graph.initializer)
    }

    blocks: dict[int, dict[str, list]] = {}
    for node in model.graph.node:
        match = BLOCK.match(node.name)
        if match is None or node.op_type != "MatMul" or len(node.input) != 2:
            continue
        block = int(match.group(1))
        lhs = producer.get(node.input[0])
        rhs = producer.get(node.input[1])
        record = blocks.setdefault(block, {"qk": [], "av": []})
        if lhs is not None and rhs is not None and lhs.op_type == "Slice" and rhs.op_type == "Transpose":
            q_head = producer.get(lhs.input[0])
            if q_head is None or q_head.op_type != "Slice":
                raise ValueError(f"{node.name}: malformed Q slice chain")
            q_mul = producer.get(q_head.input[0])
            if q_mul is None or q_mul.op_type != "Mul":
                raise ValueError(f"{node.name}: missing shared Q scale Mul")
            alpha_inputs = [name for name in q_mul.input if name != q_mul.input[0]]
            if len(alpha_inputs) != 1:
                raise ValueError(f"{q_mul.name}: malformed scalar Mul")
            alpha = constant_scalar(alpha_inputs[0], producer, arrays)
            record["qk"].append((node, q_head, q_mul, alpha))
        elif lhs is not None and rhs is not None and lhs.op_type == "Softmax" and rhs.op_type == "Mul":
            constant_inputs = [
                name
                for name in rhs.input
                if name in arrays or (producer.get(name) is not None and producer[name].op_type == "Constant")
            ]
            data_inputs = [name for name in rhs.input if name not in constant_inputs]
            if len(constant_inputs) != 1 or len(data_inputs) != 1:
                raise ValueError(f"{rhs.name}: malformed V gain Mul")
            gain = constant_scalar(constant_inputs[0], producer, arrays)
            record["av"].append((node, rhs, data_inputs[0], gain))

    report = {"blocks": {}, "attention_blocks": len(blocks)}
    for block, record in sorted(blocks.items()):
        qk = record["qk"]
        av = record["av"]
        if not qk or not av:
            raise ValueError(f"block {block}: incomplete attention triplets")

        unique_q_heads: list[tuple[onnx.NodeProto, onnx.NodeProto, float]] = []
        seen_q: set[str] = set()
        for _, q_head, q_mul, alpha in qk:
            if q_head.name not in seen_q:
                unique_q_heads.append((q_head, q_mul, alpha))
                seen_q.add(q_head.name)
        unique_v_heads: list[tuple[onnx.NodeProto, str, float]] = []
        seen_v: set[str] = set()
        for _, v_mul, data_input, gain in av:
            if v_mul.name not in seen_v:
                unique_v_heads.append((v_mul, data_input, gain))
                seen_v.add(v_mul.name)
        if len(unique_q_heads) != len(unique_v_heads):
            raise ValueError(f"block {block}: Q/V head count mismatch")
        heads = len(unique_q_heads)

        q_scales = [float(get_attr(item[0], "A_scales")[0]) for item in qk]
        k_scales = [float(get_attr(item[0], "B_scales")[0]) for item in qk]
        v_effective_scales = [float(get_attr(item[0], "B_scales")[0]) for item in av]
        q_scale = max(q_scales)
        k_scale = max(k_scales)
        v_scale = max(v_effective_scales)

        q_add, q_mm, q_weight_name, q_bias_name = projection_parts("q", block, nodes, arrays)
        k_add, _, _, _ = projection_parts("k", block, nodes, arrays)
        v_add, v_mm, v_weight_name, v_bias_name = projection_parts("v", block, nodes, arrays)

        q_alphas = {alpha for _, _, alpha in unique_q_heads}
        if len(q_alphas) != 1:
            raise ValueError(f"block {block}: inconsistent Q scale Mul")
        q_alpha = q_alphas.pop()
        arrays[q_weight_name] = arrays[q_weight_name] * np.float32(q_alpha)
        arrays[q_bias_name] = arrays[q_bias_name] * np.float32(q_alpha)
        q_weight_scales = np.asarray(get_attr(q_mm, "B_scales"), dtype=np.float32)
        set_attr(q_mm, "B_scales", (q_weight_scales * np.float32(q_alpha)).tolist())

        gains = np.asarray([gain for _, _, gain in unique_v_heads], dtype=np.float32)
        width = arrays[v_weight_name].shape[1]
        if width % heads or arrays[v_bias_name].shape != (width,):
            raise ValueError(f"block {block}: V projection/head shape mismatch")
        channel_gain = np.repeat(gains, width // heads)
        arrays[v_weight_name] = arrays[v_weight_name] * channel_gain.reshape(1, -1)
        arrays[v_bias_name] = arrays[v_bias_name] * channel_gain
        v_weight_scales = np.asarray(get_attr(v_mm, "B_scales"), dtype=np.float32)
        set_attr(v_mm, "B_scales", (v_weight_scales * channel_gain).tolist())

        for name in (q_weight_name, q_bias_name, v_weight_name, v_bias_name):
            replace_array(model, index_by_name, name, arrays[name])
        for add, scale in ((q_add, q_scale), (k_add, k_scale), (v_add, v_scale)):
            set_attr(add, "output_bitdepth", 8)
            set_attr(add, "output_scale", scale)

        for node, _, _, _ in qk:
            set_attr(node, "A_scales", [q_scale])
            set_attr(node, "B_scales", [k_scale])
        for q_head, q_mul, _ in unique_q_heads:
            q_head.input[0] = q_mul.input[0]
        for node, _, data_input, _ in av:
            node.input[1] = data_input
            set_attr(node, "B_scales", [v_scale])

        report["blocks"][str(block)] = {
            "heads": heads,
            "q_fold_alpha": q_alpha,
            "v_head_gains": gains.tolist(),
            "q_projection_scale": q_scale,
            "k_projection_scale": k_scale,
            "v_projection_scale": v_scale,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.input.resolve()), load_external_data=True)
    report = rewrite(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    external_name = args.output.name + ".data"
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=1024,
    )
    print(report)
    print(args.output)


if __name__ == "__main__":
    main()
