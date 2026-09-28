#!/usr/bin/env python3
"""Fuse static-attention A8 conversion into executable Transform producers.

The compiler's fallback BF16 ``Unstick -> Stick(A8)`` changes layout metadata
but does not execute a numeric quantizer on U250.  An identity Mul is optimized
to that same invalid sequence, and K's terminal transpose is folded into the
MatMul.  Force real, non-identity transforms while preserving QK exactly:

* remove the shared Q ``* 0.125`` and apply ``* 0.25`` per Q chunk;
* apply ``* 0.5`` after each K head transpose;
* double Q's A8 scale and halve K's A8 scale.

The quantized integers and the product of the Q/K scales are therefore
unchanged.  V already has a non-identity gain Mul and can emit A8 directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import helper


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


def set_a8(node: onnx.NodeProto, scale: float) -> None:
    set_attr(node, "output_bitdepth", 8)
    set_attr(node, "output_scale", float(scale))


def scalar_initializer(model: onnx.ModelProto, name: str, value: float) -> None:
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)
    )


def rewrite(model: onnx.ModelProto) -> dict[str, int]:
    producer = {output: node for node in model.graph.node for output in node.output}
    counts = {
        "q_shared_bypass": 0,
        "q_transform": 0,
        "k_transform": 0,
        "v_transform": 0,
    }
    bridge_cache: dict[tuple[str, float], str] = {}
    bypassed_q_heads: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    for node in list(model.graph.node):
        if node.op_type != "MatMul" or len(node.input) != 2:
            new_nodes.append(node)
            continue
        lhs = producer.get(node.input[0])
        rhs = producer.get(node.input[1])
        if lhs is None or rhs is None:
            new_nodes.append(node)
            continue
        if lhs.op_type == "Slice" and rhs.op_type == "Transpose":
            q_scale = float(get_attr(node, "A_scales")[0])
            k_scale = float(get_attr(node, "B_scales")[0])

            # The final Q chunk Slice consumes a head Slice, which consumes the
            # shared scale Mul.  Bypass the shared *0.125 once per head.  The
            # per-chunk *0.25 below supplies twice the value with twice the A8
            # scale, hence exactly the same integer codes.
            q_head = producer.get(lhs.input[0])
            if q_head is None or q_head.op_type != "Slice":
                raise RuntimeError(f"{node.name}: unexpected Q producer chain")
            if q_head.name not in bypassed_q_heads:
                q_shared_mul = producer.get(q_head.input[0])
                if q_shared_mul is None or q_shared_mul.op_type != "Mul":
                    raise RuntimeError(f"{node.name}: missing shared Q scale Mul")
                q_head.input[0] = q_shared_mul.input[0]
                bypassed_q_heads.add(q_head.name)
                counts["q_shared_bypass"] += 1

            q_new_scale = 2.0 * q_scale
            k_new_scale = 0.5 * k_scale
            set_attr(node, "A_scales", [q_new_scale])
            set_attr(node, "B_scales", [k_new_scale])

            for index, (source, scale, alpha, kind) in enumerate(
                (
                    (node.input[0], q_new_scale, 0.25, "q"),
                    (node.input[1], k_new_scale, 0.5, "k"),
                )
            ):
                key = (source, alpha)
                bridge_output = bridge_cache.get(key)
                if bridge_output is None:
                    stem = node.name.strip("/").replace("/", "_") or "MatMul_0"
                    constant_name = f"/{stem}/{kind}_a8_transform_alpha"
                    bridge_output = f"/{stem}/{kind}_a8_transform_output_0"
                    scalar_initializer(model, constant_name, alpha)
                    bridge = helper.make_node(
                        "Mul",
                        [source, constant_name],
                        [bridge_output],
                        name=f"/{stem}/{kind}_a8_transform",
                        weight_bitdepth=16,
                        weight_scale=-1.0,
                        output_bitdepth=8,
                        output_scale=scale,
                    )
                    new_nodes.append(bridge)
                    producer[bridge_output] = bridge
                    bridge_cache[key] = bridge_output
                    counts[kind + "_transform"] += 1
                node.input[index] = bridge_output
        elif lhs.op_type == "Softmax" and rhs.op_type == "Mul":
            v_scale = float(get_attr(node, "B_scales")[0])
            if int(get_attr(rhs, "output_bitdepth")) != 8:
                counts["v_transform"] += 1
            set_a8(rhs, v_scale)
        new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.input), load_external_data=True)
    counts = rewrite(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(str(args.output) + ".data")
    for old in (args.output, data_path):
        if old.exists():
            old.unlink()
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
    )
    print(counts)
    print(args.output)


if __name__ == "__main__":
    main()
