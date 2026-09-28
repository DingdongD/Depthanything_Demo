#!/usr/bin/env python3
"""Export a diagnostic QK -> fine/residual probability kernel for U250."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--fine-step", type=float, default=1.0 / 16384.0)
    parser.add_argument("--residual-step", type=float, default=1.0 / 128.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fine_step <= 0 or args.residual_step <= 0:
        raise ValueError("probability scales must be positive")
    threshold = 127.0 * args.fine_step
    contract = json.loads(args.contract.read_text())
    scales = contract["encoder"][0]["attention"]["heads"][args.head]["scales_bf16"]
    q_scale, k_scale = float(scales["q"]), float(scales["k"])
    negative_threshold = "dual_range_negative_threshold"
    nodes = [
        helper.make_node(
            "MatMul", ["input0", "input1"], ["logits"], name="/QK",
            A_bitdepth=8, A_scales=[q_scale], B_bitdepth=8, B_scales=[k_scale],
            output_bitdepth=16, output_scale=-1.0,
        ),
        helper.make_node(
            "Softmax", ["logits"], ["fine_probability"],
            name="/DualRangeFineSoftmax", axis=-1,
            input_bitdepth=16, input_scale=-1.0, input_scales=[-1.0],
            output_bitdepth=8, output_scale=args.fine_step,
            output_scales=[args.fine_step],
        ),
        helper.make_node(
            "Softmax", ["logits"], ["residual_bf16"],
            name="/DualRangeResidualSoftmax", axis=-1,
            input_bitdepth=16, input_scale=-1.0, input_scales=[-1.0],
            output_bitdepth=16, output_scale=-1.0, output_scales=[-1.0],
        ),
        helper.make_node(
            "Add", ["residual_bf16", negative_threshold], ["residual_shifted"],
            name="/DualRangeResidualSubtract",
            left_scale=-1.0, left_scales=[-1.0], left_bitdepth=16,
            input_scale=-1.0, input_scales=[-1.0], input_bitdepth=16,
            const_scale=1.0 / args.residual_step, const_bitdepth=16,
            output_scale=args.residual_step, output_scales=[args.residual_step],
            output_bitdepth=8,
        ),
        helper.make_node(
            "Relu", ["residual_shifted"], ["residual_probability"],
            name="/DualRangeResidualRelu",
            input_scale=args.residual_step, input_scales=[args.residual_step],
            input_bitdepth=8, output_scale=args.residual_step,
            output_scales=[args.residual_step], output_bitdepth=8,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "u250_dual_range_probability_probe",
        [
            helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 256, 64]),
            helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64, 1370]),
        ],
        [
            helper.make_tensor_value_info(
                "fine_probability", TensorProto.FLOAT, [1, 256, 1370]
            ),
            helper.make_tensor_value_info(
                "residual_bf16", TensorProto.FLOAT, [1, 256, 1370]
            ),
            helper.make_tensor_value_info(
                "residual_probability", TensorProto.FLOAT, [1, 256, 1370]
            ),
        ],
        [
            numpy_helper.from_array(
                np.asarray(-threshold, np.float32), name=negative_threshold
            )
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]),
        args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
