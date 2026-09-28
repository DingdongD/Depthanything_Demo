#!/usr/bin/env python3
"""Replace the ViT /14 patch Conv with exact SpaceToDepth + 1x1 Conv.

The DS backend lowers a stride-14 Conv to a stride-2 Conv followed by a
hardware DownSample instruction.  The U250 image used by this project does
not complete that DownSample instruction.  For a 518x518 input, block-14
SpaceToDepth followed by a reshaped 1x1 kernel is algebraically identical to
the original non-overlapping 14x14 patch projection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PATCH_OUTPUT = "/patch_embed/proj/Conv_output_0"


def _attr(node: onnx.NodeProto, name: str):
    for value in node.attribute:
        if value.name == name:
            return helper.get_attribute_value(value)
    return None


def _replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def _replace_initializer(graph: onnx.GraphProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(value), name=name)
            )
            return
    raise KeyError(f"initializer {name!r} not found")


def _set_value_info_shape(graph: onnx.GraphProto, name: str, dims: list[int]) -> None:
    values = list(graph.input) + list(graph.output) + list(graph.value_info)
    for value in values:
        if value.name == name:
            shape = value.type.tensor_type.shape
            del shape.dim[:]
            for dim in dims:
                shape.dim.add().dim_value = dim
            return
    graph.value_info.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
    )


def rewrite(model: onnx.ModelProto, lowering: str = "space-to-depth") -> onnx.ModelProto:
    graph = model.graph
    concat_indices = [
        index
        for index, node in enumerate(graph.node)
        if node.op_type == "Concat" and PATCH_OUTPUT in node.output
    ]
    if len(concat_indices) != 1:
        raise ValueError(
            f"expected one split patch Concat producing {PATCH_OUTPUT}, "
            f"found {len(concat_indices)}"
        )
    concat_index = concat_indices[0]
    concat = graph.node[concat_index]
    chunk_names = set(concat.input)
    chunk_convs = [
        node
        for node in graph.node[:concat_index]
        if node.op_type == "Conv" and any(name in chunk_names for name in node.output)
    ]
    if not chunk_convs:
        raise ValueError("split patch Conv nodes were not found")
    first_conv = chunk_convs[0]
    kernel_shape = list(_attr(first_conv, "kernel_shape") or [])
    strides = list(_attr(first_conv, "strides") or [])
    if kernel_shape != [14, 14] or strides != [14, 14]:
        raise ValueError(f"unexpected patch Conv attributes: {kernel_shape}, {strides}")

    weight_name = first_conv.input[1]
    initializer_by_name = {value.name: value for value in graph.initializer}
    weight = numpy_helper.to_array(initializer_by_name[weight_name]).copy()
    if weight.ndim != 4 or tuple(weight.shape[2:]) != (14, 14):
        raise ValueError(f"unexpected patch weight shape {weight.shape}")
    output_channels, input_channels, _, _ = weight.shape
    # ONNX SpaceToDepth orders output channels as (kernel_y, kernel_x,
    # input_channel), whereas Conv stores weights as (output_channel,
    # input_channel, kernel_y, kernel_x).
    flattened_weight = weight.transpose(0, 2, 3, 1).reshape(
        output_channels, input_channels * 196, 1, 1
    )
    _replace_initializer(graph, weight_name, flattened_weight)

    s2d_output = "ds_patch_space_to_depth_output"
    calibrated = _attr(first_conv, "weight_bitdepth") is not None
    if lowering == "space-to-depth":
        rearrange_nodes = [
            helper.make_node(
                "SpaceToDepth",
                [graph.input[0].name],
                [s2d_output],
                blocksize=14,
                name="/ds_patch_space_to_depth/SpaceToDepth",
            )
        ]
    elif lowering == "reshape":
        shape1_name = "ds_patch_rearrange_shape1"
        shape2_name = "ds_patch_rearrange_shape2"
        graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.asarray([1, input_channels, 37, 14, 37, 14], dtype=np.int64),
                    name=shape1_name,
                ),
                numpy_helper.from_array(
                    np.asarray([1, input_channels * 196, 37, 37], dtype=np.int64),
                    name=shape2_name,
                ),
            ]
        )
        reshape1 = helper.make_node(
            "Reshape",
            [graph.input[0].name, shape1_name],
            ["ds_patch_rearrange_6d"],
            allowzero=0,
            name="/ds_patch_rearrange/Reshape6D",
        )
        transpose = helper.make_node(
            "Transpose",
            ["ds_patch_rearrange_6d"],
            ["ds_patch_rearrange_transposed"],
            perm=[0, 3, 5, 1, 2, 4],
            name="/ds_patch_rearrange/Transpose",
        )
        reshape2 = helper.make_node(
            "Reshape",
            ["ds_patch_rearrange_transposed", shape2_name],
            [s2d_output],
            allowzero=0,
            name="/ds_patch_rearrange/Reshape4D",
        )
        rearrange_nodes = [reshape1, transpose, reshape2]
    elif lowering == "reshape4d":
        # Equivalent pixel-unshuffle using only rank-3/rank-4 transforms.
        # The backend layout pass does not support the natural rank-6 form.
        shape_values = (
            ("ds_patch_r4_shape1", [input_channels * 518, 37, 14]),
            ("ds_patch_r4_shape2", [input_channels, 518, 14, 37]),
            ("ds_patch_r4_shape3", [14 * input_channels, 37, 14, 37]),
            ("ds_patch_r4_shape4", [1, input_channels * 196, 37, 37]),
        )
        graph.initializer.extend(
            numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)
            for name, value in shape_values
        )
        specs = (
            ("Reshape", graph.input[0].name, "ds_patch_r4_a", "ds_patch_r4_shape1", None),
            ("Transpose", "ds_patch_r4_a", "ds_patch_r4_b", None, [0, 2, 1]),
            ("Reshape", "ds_patch_r4_b", "ds_patch_r4_c", "ds_patch_r4_shape2", None),
            ("Transpose", "ds_patch_r4_c", "ds_patch_r4_d", None, [2, 0, 1, 3]),
            ("Reshape", "ds_patch_r4_d", "ds_patch_r4_e", "ds_patch_r4_shape3", None),
            ("Transpose", "ds_patch_r4_e", "ds_patch_r4_f", None, [2, 0, 1, 3]),
            ("Reshape", "ds_patch_r4_f", s2d_output, "ds_patch_r4_shape4", None),
        )
        rearrange_nodes = []
        for index, (op_type, input_name, output_name, shape_name, perm) in enumerate(specs):
            inputs = [input_name] + ([shape_name] if shape_name else [])
            kwargs = {"allowzero": 0} if op_type == "Reshape" else {"perm": perm}
            rearrange_nodes.append(
                helper.make_node(
                    op_type,
                    inputs,
                    [output_name],
                    name=f"/ds_patch_rearrange4d/{index}_{op_type}",
                    **kwargs,
                )
            )
    else:
        raise ValueError(f"unsupported lowering {lowering!r}")
    # Compiler-private quantization attributes are legal on the calibrated
    # source, but intentionally omitted for a normal FP ONNX model.
    if calibrated:
        for node in rearrange_nodes:
            if node.op_type == "SpaceToDepth":
                node.attribute.extend(
                    [
                        helper.make_attribute("input_bitdepth", 16),
                        helper.make_attribute("input_scale", -1.0),
                        helper.make_attribute("output_bitdepth", 16),
                        helper.make_attribute("output_scale", -1.0),
                    ]
                )
            else:
                node.attribute.append(helper.make_attribute("output_bitdepth", 16))

    conv = onnx.NodeProto()
    conv.CopyFrom(first_conv)
    conv.name = "/ds_patch_space_to_depth/Conv1x1"
    conv.input[0] = s2d_output
    del conv.output[:]
    conv.output.extend([PATCH_OUTPUT])
    for name, value in (
        ("kernel_shape", [1, 1]),
        ("strides", [1, 1]),
        ("dilations", [1, 1]),
        ("pads", [0, 0, 0, 0]),
    ):
        _replace_attr(conv, name, value)

    removable_outputs = set(chunk_names)
    removable_outputs.update(
        output
        for node in graph.node[:concat_index]
        if node.name.startswith("/ds_patch_rows/")
        for output in node.output
    )
    kept_nodes = [
        node
        for index, node in enumerate(graph.node)
        if index > concat_index or not node.name.startswith("/ds_patch_rows/")
    ]
    del graph.node[:]
    graph.node.extend(rearrange_nodes + [conv] + kept_nodes)

    _set_value_info_shape(graph, s2d_output, [1, input_channels * 196, 37, 37])
    _set_value_info_shape(graph, PATCH_OUTPUT, [1, output_channels, 37, 37])

    # Remove only initializers that belonged exclusively to the deleted Slice
    # nodes.  Keeping unrelated calibrated constants preserves all later ops.
    used = {name for node in graph.node for name in node.input}
    stale = {
        value.name
        for value in graph.initializer
        if value.name.startswith("ds_patch_rows_") and value.name not in used
    }
    if stale:
        retained = [value for value in graph.initializer if value.name not in stale]
        del graph.initializer[:]
        graph.initializer.extend(retained)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--lowering",
        choices=("space-to-depth", "reshape", "reshape4d"),
        default="space-to-depth",
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".data").exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    model = onnx.load(str(args.input), load_external_data=True)
    rewrite(model, lowering=args.lowering)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
