#!/usr/bin/env python3
"""Export two-chunk attention kernels with SPU per-row dynamic A8 scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_model(q_scale: float, k_scale: float, v_scale: float) -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 256, 64]),
        helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 256, 64]),
        helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 64, 1370]),
        helper.make_tensor_value_info("input3", TensorProto.FLOAT, [1, 1370, 64]),
    ]
    nodes = []
    initializers = []
    outputs = []
    for index in range(2):
        prefix = f"/chunk{index}"
        logits = prefix + "/logits"
        probability = prefix + "/probability"
        av = prefix + "/av"
        output = f"output{index}"
        nodes.extend([
            helper.make_node(
                "MatMul", [f"input{index}", "input2"], [logits],
                name=prefix + "/QK",
                A_bitdepth=8, A_scales=[q_scale],
                B_bitdepth=8, B_scales=[k_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
            helper.make_node(
                "Softmax", [logits], [probability], name=prefix + "/Softmax",
                axis=-1, input_bitdepth=16, input_scale=-1.0,
                output_bitdepth=8, output_scale=-1.0,
                output_scales=[], output_quant_dim=[1],
            ),
            helper.make_node(
                "MatMul", [probability, "input3"], [av],
                name=prefix + "/AV",
                A_bitdepth=8, A_scale=-1.0,
                A_scales=[], A_quant_dim=[1],
                B_bitdepth=8, B_scales=[v_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
        ])
        shape = prefix + "/output_shape"
        initializers.append(numpy_helper.from_array(
            np.asarray([1, 256, 64], dtype=np.int64), name=shape
        ))
        nodes.append(helper.make_node(
            "Reshape", [av, shape], [output], name=prefix + "/OutputReshape"
        ))
        outputs.append(helper.make_tensor_value_info(
            output, TensorProto.FLOAT, [1, 256, 64]
        ))
    graph = helper.make_graph(
        nodes, "u250_attention_dynamic_rowmax_2chunk", inputs, outputs, initializers
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    layer = contract["encoder"][args.layer]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = []
    for head, specification in enumerate(layer["attention"]["heads"]):
        scales = specification["scales_bf16"]
        if float(scales.get("av_output_gain", 1.0)) != 1.0:
            raise ValueError(f"head {head}: forbidden AV gain")
        if float(scales.get("av_v", scales["v"])) != float(scales["v"]):
            raise ValueError(f"head {head}: AV scale differs from V")
        name = f"attention2_l{args.layer:02d}_h{head:02d}"
        output = args.output_dir / f"{name}.onnx"
        onnx.save(make_model(
            float(scales["q"]), float(scales["k"]), float(scales["v"])
        ), output)
        kernels.append({
            "layer": args.layer,
            "head": head,
            "name": name,
            "onnx": output.name,
            "probability_quantization": "dynamic-row-max-a8",
            "probability_rounding": "spu-dqu-round-to-nearest",
            "calls_per_head": 3,
            "query_groups": [[0, 1], [2, 3], [4, 5]],
            "scales_bf16": {
                "q": float(scales["q"]),
                "k": float(scales["k"]),
                "v": float(scales["v"]),
                "av_v": float(scales["v"]),
                "av_output_gain": 1.0,
                "probability": "dynamic-row-max/127",
            },
        })
        print(output)
    manifest = {
        "schema_version": 1,
        "strategy": "two chunks per launch with SPU dynamic row scales",
        "probability_rounding": "spu-dqu-round-to-nearest",
        "layer": args.layer,
        "kernels": kernels,
        "kernels_total": len(kernels),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
