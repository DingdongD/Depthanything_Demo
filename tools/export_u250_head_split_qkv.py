#!/usr/bin/env python3
"""Export one QKV projection as 18 calibrated INT8 head outputs.

The resident attention kernels consume one static-INT8 Q, K and V tensor per
head.  Emitting those tensors directly from QKV is the compiler-side half of a
device-resident QKV-to-attention boundary: no empirical gain is applied and
every output scale comes from the existing qualified runtime contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        value.name: numpy_helper.to_array(value).astype(np.float32, copy=False)
        for value in model.graph.initializer
    }


def source_nodes(
    model: onnx.ModelProto, layer: int
) -> dict[str, tuple[onnx.NodeProto, onnx.NodeProto]]:
    prefix = f"/blocks.{layer}/attn/qkv"
    result = {}
    for branch in ("q", "k", "v"):
        matmul_name = f"{prefix}/{branch}/MatMul"
        add_name = f"{prefix}/{branch}/Add"
        matmul = next((node for node in model.graph.node
                       if node.name == matmul_name), None)
        add = next((node for node in model.graph.node
                    if node.name == add_name), None)
        if matmul is None or add is None:
            raise ValueError(f"source does not contain {matmul_name}/{add_name}")
        result[branch] = (matmul, add)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer <= 11:
        parser.error("--layer must be in 0..11")

    contract = json.loads(args.contract.read_text())
    block = contract["encoder"][args.layer]
    heads = block["attention"]["heads"]
    if [int(item["head"]) for item in heads] != list(range(len(heads))):
        raise ValueError("attention heads must be densely ordered from zero")
    model = onnx.load(str(args.source), load_external_data=True)
    values = initializer_map(model)
    nodes = source_nodes(model, args.layer)
    input_scale = float(block["qkv"]["input_quantization"]["scale"])
    input_dims = [
        int(dim.dim_value)
        for dim in model.graph.input[0].type.tensor_type.shape.dim
    ]
    if len(input_dims) != 3 or any(value <= 0 for value in input_dims):
        raise ValueError(f"expected one static BWC input, got {input_dims}")
    batch, tokens, hidden = input_dims
    if hidden % len(heads):
        raise ValueError("hidden size must be divisible by attention head count")
    head_width = hidden // len(heads)

    exported_nodes = []
    exported_initializers = []
    exported_outputs = []
    records = []
    prefix = f"/blocks.{args.layer}/attn/qkv_head_split"
    for branch in ("q", "k", "v"):
        source_matmul, source_add = nodes[branch]
        weight = values[source_matmul.input[1]]
        bias = values[source_add.input[0]]
        if weight.shape != (hidden, hidden) or bias.shape != (hidden,):
            raise ValueError(f"{branch}: unexpected weight/bias shapes")
        for head, head_contract in enumerate(heads):
            begin, end = head * head_width, (head + 1) * head_width
            head_weight = np.ascontiguousarray(weight[:, begin:end])
            head_bias = np.ascontiguousarray(bias[begin:end])
            scale_name = "v" if branch == "v" else branch
            output_scale = float(head_contract["scales_bf16"][scale_name])
            if branch == "v":
                av_scale = float(head_contract["scales_bf16"]["av_v"])
                gain = float(head_contract["scales_bf16"].get(
                    "av_output_gain", 1.0
                ))
                if av_scale != output_scale or gain != 1.0:
                    raise ValueError(
                        f"head {head}: V/AV scale mismatch or non-unit gain"
                    )
            stem = f"{prefix}/{branch}_h{head:02d}"
            weight_name = stem + "/weight"
            bias_name = stem + "/bias"
            matmul_output = stem + "/MatMul_output_0"
            output_name = stem + "/Add_output_0"
            weight_scales = np.maximum(
                np.max(np.abs(head_weight), axis=0) / 127.0,
                np.finfo(np.float32).tiny,
            )
            exported_nodes.extend([
                helper.make_node(
                    "MatMul", ["input0", weight_name], [matmul_output],
                    name=stem + "/MatMul", A_bitdepth=8, B_bitdepth=8,
                    A_scales=[input_scale],
                    B_scales=weight_scales.astype(np.float32).tolist(),
                    B_quant_dim=[1], output_bitdepth=16,
                    output_scale=-1.0,
                ),
                helper.make_node(
                    "Add", [bias_name, matmul_output], [output_name],
                    name=stem + "/Add", const_bitdepth=16,
                    const_scale=-1.0, output_bitdepth=8,
                    output_scale=output_scale,
                ),
            ])
            exported_initializers.extend([
                numpy_helper.from_array(head_weight, name=weight_name),
                numpy_helper.from_array(head_bias, name=bias_name),
            ])
            exported_outputs.append(helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [batch, tokens, head_width]
            ))
            records.append({
                "branch": branch,
                "head": head,
                "output": output_name,
                "output_scale": output_scale,
            })

    del model.graph.node[:]
    model.graph.node.extend(exported_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(exported_initializers)
    del model.graph.output[:]
    model.graph.output.extend(exported_outputs)
    name = f"qkv_head_split_l{args.layer:02d}_a8_to_18xa8"
    model.graph.name = name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{name}.onnx"
    onnx.save_model(
        model, str(output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=output.name + ".data",
        size_threshold=1024,
    )
    manifest = {
        "schema_version": 1,
        "policy": "static-head-scales-no-gain",
        "source": str(args.source.resolve()),
        "contract": str(args.contract.resolve()),
        "layer": args.layer,
        "input_scale": input_scale,
        "head_width": head_width,
        "outputs": records,
        "onnx": output.name,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "input_scale": input_scale,
        "model": str(output),
        "outputs": len(records),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
