#!/usr/bin/env python3
"""Split a large-stride patch embedding Conv into exact row chunks.

DS lowers stride > 2 to stride-2 Conv plus DownSample.  On a 518x518 input,
lowering the ViT-S/14 patch projection as one operation creates a transient
384x253x253 BF16 tensor.  Splitting output rows first keeps every transient
below the U250 feature-map ring capacity without changing any arithmetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def _attr(node: onnx.NodeProto, name: str):
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-per-chunk", type=int, default=2)
    args = parser.parse_args()
    if args.rows_per_chunk <= 0:
        raise SystemExit("--rows-per-chunk must be positive")

    model = onnx.load(str(args.input), load_external_data=True)
    candidates = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Conv"
        and list(_attr(node, "kernel_shape") or []) == [14, 14]
        and list(_attr(node, "strides") or []) == [14, 14]
    ]
    if len(candidates) != 1:
        raise SystemExit(f"expected one 14x14/stride14 Conv, found {len(candidates)}")
    index, conv = candidates[0]

    input_shape = None
    for value in list(model.graph.input) + list(model.graph.value_info):
        if value.name == conv.input[0]:
            dims = value.type.tensor_type.shape.dim
            if all(dim.HasField("dim_value") for dim in dims):
                input_shape = [dim.dim_value for dim in dims]
            break
    if not input_shape or len(input_shape) != 4:
        raise SystemExit(f"static NCHW shape missing for {conv.input[0]}")
    height = input_shape[2]
    kernel = 14
    output_rows = (height - kernel) // kernel + 1

    new_nodes: list[onnx.NodeProto] = []
    chunk_outputs: list[str] = []
    prefix = "ds_patch_rows"
    for chunk_index, row0 in enumerate(range(0, output_rows, args.rows_per_chunk)):
        rows = min(args.rows_per_chunk, output_rows - row0)
        y0 = row0 * kernel
        y1 = y0 + rows * kernel
        names = {}
        for suffix, values in (
            ("starts", [y0]),
            ("ends", [y1]),
            ("axes", [2]),
            ("steps", [1]),
        ):
            name = f"{prefix}_{chunk_index}_{suffix}"
            names[suffix] = name
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
            )
        slice_output = f"{prefix}_{chunk_index}_slice"
        conv_output = f"{prefix}_{chunk_index}_conv"
        new_nodes.append(
            helper.make_node(
                "Slice",
                [conv.input[0], names["starts"], names["ends"], names["axes"], names["steps"]],
                [slice_output],
                name=f"/{prefix}/{chunk_index}/Slice",
            )
        )
        cloned = onnx.NodeProto()
        cloned.CopyFrom(conv)
        cloned.name = f"/{prefix}/{chunk_index}/Conv"
        cloned.input[0] = slice_output
        cloned.output[0] = conv_output
        new_nodes.append(cloned)
        chunk_outputs.append(conv_output)

    new_nodes.append(
        helper.make_node(
            "Concat",
            chunk_outputs,
            list(conv.output),
            axis=2,
            name=f"/{prefix}/Concat",
        )
    )
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes[:index] + new_nodes + nodes[index + 1 :])
    # The calibrated DS model intentionally carries compiler-private Conv
    # attributes (for example weight_ch_scales) that the stock ONNX checker
    # does not recognize.  Structural validation is performed by loading the
    # saved model and by ORT/compiler smoke tests instead.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    location = args.output.name + ".data"
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"split patch embedding into {len(chunk_outputs)} chunks; output={args.output}")


if __name__ == "__main__":
    main()
