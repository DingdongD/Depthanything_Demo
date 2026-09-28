#!/usr/bin/env python3
"""Build a six-output U250 attention-head kernel with explicit A8 Q/K/V inputs.

The QKV projection is deliberately outside this graph.  A layer projects Q, K
and V once and leaves the three tensors in DDR; each head kernel consumes its
64-channel views and emits the six query chunks independently.  This keeps the
attention graph comfortably below the U250 4 MiB feature-memory limit while
reducing six launches per head to one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CHUNKS = ((0, 256), (256, 512), (512, 768), (768, 1024),
          (1024, 1280), (1280, 1370))


def scalar_i64(name: str, value: int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name)


def make_model(q_scale: float, k_scale: float, probability_scale: float,
               v_scale: float, chunk_count: int,
               softmax_output_bf16: bool = False,
               av_v_scale: float | None = None) -> onnx.ModelProto:
    if av_v_scale is None:
        av_v_scale = v_scale
    chunks = CHUNKS[:chunk_count]
    inputs = [
        helper.make_tensor_value_info(f"input{index}", TensorProto.FLOAT,
                                      [1, stop - start, 64])
        for index, (start, stop) in enumerate(chunks)
    ]
    inputs.extend([
        helper.make_tensor_value_info(f"input{chunk_count}", TensorProto.FLOAT,
                                      [1, 64, 1370]),
        helper.make_tensor_value_info(f"input{chunk_count + 1}", TensorProto.FLOAT,
                                      [1, 1370, 64]),
    ])
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    for index, (start, stop) in enumerate(chunks):
        prefix = f"/chunk{index}"
        logits = prefix + "/logits"
        probability = prefix + "/probability"
        av = prefix + "/av"
        output = f"output{index}"
        nodes.extend([
            helper.make_node(
                "MatMul", [f"input{index}", f"input{chunk_count}"], [logits],
                name=prefix + "/QK",
                A_bitdepth=8, A_scales=[q_scale],
                B_bitdepth=8, B_scales=[k_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
            helper.make_node(
                "Softmax", [logits], [probability], name=prefix + "/Softmax",
                axis=-1, input_bitdepth=16, input_scale=-1.0,
                **({"output_bitdepth": 16, "output_scale": -1.0}
                   if softmax_output_bf16 else
                   {"output_bitdepth": 8,
                    "output_scales": [probability_scale]}),
            ),
            helper.make_node(
                "MatMul", [probability, f"input{chunk_count + 1}"], [av],
                name=prefix + "/AV",
                A_bitdepth=8, A_scales=[probability_scale],
                B_bitdepth=8, B_scales=[av_v_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
        ])
        # A real downstream consumer prevents the AV result from being exposed
        # directly from the accelerator's internal tiled matrix layout.
        shape = prefix + "/output_shape"
        rows = stop - start
        initializers.append(numpy_helper.from_array(
            np.asarray([1, rows, 64], dtype=np.int64), name=shape))
        nodes.append(helper.make_node(
            "Reshape", [av, shape], [output], name=prefix + "/OutputReshape"))
        outputs.append(helper.make_tensor_value_info(
            output, TensorProto.FLOAT, [1, rows, 64]))

    graph = helper.make_graph(nodes, "u250_attention_head_multioutput", inputs,
                              outputs, initializers)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid("", 13)])
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-scale", type=float, required=True)
    parser.add_argument("--k-scale", type=float, required=True)
    parser.add_argument("--probability-scale", type=float, required=True)
    parser.add_argument("--v-scale", type=float, required=True)
    parser.add_argument("--chunks", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--softmax-output-bf16", action="store_true",
                        help="insert the A8 boundary at AV instead of Softmax")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(args.q_scale, args.k_scale,
                         args.probability_scale, args.v_scale, args.chunks,
                         args.softmax_output_bf16),
              args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
