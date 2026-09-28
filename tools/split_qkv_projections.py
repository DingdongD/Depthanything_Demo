#!/usr/bin/env python3
"""Split fused ViT QKV projections into three hardware-sized projections.

The U250 CTC path is reliable for 384-channel A8 x B8 projections, while the
fused 1152-channel QKV output can leave the output DMA incomplete.  This tool
rewrites each ``qkv/MatMul`` + ``qkv/Add`` pair as three independent linear
projections followed by a channel-axis Concat.  Run DS quantization after this
rewrite so every branch receives its own weight scales.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}


def split_qkv(model: onnx.ModelProto) -> int:
    arrays = _initializer_arrays(model)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    replacements: dict[int, list[onnx.NodeProto]] = {}
    removed: set[int] = set()
    new_initializers: list[onnx.TensorProto] = []
    split_count = 0

    for index, matmul in enumerate(model.graph.node):
        if matmul.op_type != "MatMul" or "/qkv/MatMul" not in matmul.name:
            continue
        if len(matmul.input) != 2 or matmul.input[1] not in arrays:
            raise ValueError(f"{matmul.name}: expected a constant RHS weight")
        users = consumers.get(matmul.output[0], [])
        if len(users) != 1 or users[0].op_type != "Add":
            raise ValueError(f"{matmul.name}: expected one bias Add consumer")
        add = users[0]
        add_index = next(i for i, node in enumerate(model.graph.node) if node is add)
        bias_names = [name for name in add.input if name in arrays]
        if len(bias_names) != 1:
            raise ValueError(f"{add.name}: expected one constant bias")

        weight = arrays[matmul.input[1]]
        bias = arrays[bias_names[0]]
        if weight.ndim != 2 or weight.shape[1] % 3 or bias.shape != (weight.shape[1],):
            raise ValueError(
                f"{matmul.name}: invalid QKV shapes weight={weight.shape}, bias={bias.shape}"
            )
        width = weight.shape[1] // 3
        branch_outputs = []
        nodes = []
        for branch_index, label in enumerate(("q", "k", "v")):
            begin, end = branch_index * width, (branch_index + 1) * width
            base = f"{matmul.name.rsplit('/MatMul', 1)[0]}/{label}"
            weight_name = base + "/weight"
            bias_name = base + "/bias"
            mm_output = base + "/MatMul_output_0"
            branch_output = base + "/Add_output_0"
            new_initializers.extend(
                [
                    numpy_helper.from_array(np.ascontiguousarray(weight[:, begin:end]), weight_name),
                    numpy_helper.from_array(np.ascontiguousarray(bias[begin:end]), bias_name),
                ]
            )
            branch_matmul = helper.make_node(
                "MatMul", [matmul.input[0], weight_name], [mm_output], name=base + "/MatMul"
            )
            branch_add = helper.make_node(
                "Add", [bias_name, mm_output], [branch_output], name=base + "/Add"
            )
            # The bias Add owns the BF16 output contract in DS-quantized
            # graphs.  Losing it makes ACPC silently expose/propagate INT8
            # even though the projection MatMul requests a BF16 output.
            branch_add.attribute.extend(copy.deepcopy(add.attribute))
            nodes.extend([branch_matmul, branch_add])
            branch_outputs.append(branch_output)
        concat = helper.make_node(
            "Concat", branch_outputs, list(add.output), name=add.name + "/QKVSplit", axis=-1
        )
        concat.attribute.extend(
            copy.deepcopy(
                [attr for attr in add.attribute if attr.name.startswith("output_")]
            )
        )
        nodes.append(concat)
        replacements[index] = nodes
        removed.add(add_index)
        split_count += 1

    if not split_count:
        raise ValueError("no /qkv/MatMul + Add pairs found")
    rewritten = []
    for index, node in enumerate(model.graph.node):
        if index in replacements:
            rewritten.extend(replacements[index])
        elif index not in removed:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(new_initializers)
    used_initializers = {name for node in model.graph.node for name in node.input}
    kept_initializers = [
        value for value in model.graph.initializer if value.name in used_initializers
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    return split_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input.resolve()), load_external_data=True)
    count = split_qkv(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(str(args.output) + ".data")
    if args.output.exists():
        args.output.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
    )
    print(f"split_qkv_count={count}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
