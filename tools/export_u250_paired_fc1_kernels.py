#!/usr/bin/env python3
"""Export two-output FC1 kernels to reduce U250 physical dispatches.

Each kernel contains two independent 384x256 INT8 MatMul branches sharing one
INT8 token input.  DS NPU has already qualified two-output attention programs;
this probe applies the same bounded fan-out to FC1 without changing arithmetic
or concatenation order.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CHANNELS_PER_BRANCH = 256
BRANCHES_PER_KERNEL = 2
LAYERS = 12


def attribute(node: onnx.NodeProto, name: str) -> object:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    raise KeyError(f"{node.name}: missing attribute {name}")


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in value.split(",") if item)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be a non-empty unique list")
    if any(layer < 0 or layer >= LAYERS for layer in layers):
        raise ValueError("layers must be in 0..11")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default=",".join(str(x) for x in range(LAYERS)))
    args = parser.parse_args()
    try:
        layers = parse_layers(args.layers)
    except ValueError as error:
        parser.error(str(error))

    source = onnx.load(str(args.model), load_external_data=True)
    nodes = {node.name: node for node in source.graph.node}
    initializers = {item.name: item for item in source.graph.initializer}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for layer in layers:
        matmul = nodes[f"/blocks.{layer}/mlp/fc1/MatMul"]
        add = nodes[f"/blocks.{layer}/mlp/fc1/Add"]
        weight = numpy_helper.to_array(initializers[matmul.input[1]]).astype(np.float32)
        bias_name = add.input[0] if add.input[1] == matmul.output[0] else add.input[1]
        bias = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
        if weight.shape != (384, 1536) or bias.shape != (1536,):
            raise ValueError(f"layer {layer}: unsupported FC1 dimensions")
        scales = np.asarray(attribute(matmul, "B_scales"), dtype=np.float32)
        if scales.shape != (1536,):
            raise ValueError(f"layer {layer}: missing per-channel FC1 scales")

        pair_width = CHANNELS_PER_BRANCH * BRANCHES_PER_KERNEL
        for pair, pair_begin in enumerate(range(0, weight.shape[1], pair_width)):
            pair_end = min(pair_begin + pair_width, weight.shape[1])
            if pair_end - pair_begin != pair_width:
                raise ValueError("FC1 output does not divide into exact branch pairs")
            name = f"mlp_fc1_pair_l{layer:02d}_p{pair:02d}"
            body = []
            values = []
            outputs = []
            original_kernels = []
            for branch in range(BRANCHES_PER_KERNEL):
                begin = pair_begin + branch * CHANNELS_PER_BRANCH
                end = begin + CHANNELS_PER_BRANCH
                branch_matmul = copy.deepcopy(matmul)
                branch_add = copy.deepcopy(add)
                weight_name = f"{name}_weight_{branch}"
                branch_bias_name = f"{name}_bias_{branch}"
                matmul_output = f"{name}/matmul_{branch}"
                output = f"{name}/output_{branch}"
                branch_matmul.name = f"{name}/MatMul_{branch}"
                branch_matmul.input[:] = ["input0", weight_name]
                branch_matmul.output[:] = [matmul_output]
                replace_attribute(
                    branch_matmul, "B_scales", scales[begin:end].tolist()
                )
                branch_add.name = f"{name}/Add_{branch}"
                branch_add.input[:] = [branch_bias_name, matmul_output]
                branch_add.output[:] = [output]
                body.extend([branch_matmul, branch_add])
                values.extend([
                    numpy_helper.from_array(
                        np.ascontiguousarray(weight[:, begin:end]), name=weight_name
                    ),
                    numpy_helper.from_array(
                        np.ascontiguousarray(bias[begin:end]), name=branch_bias_name
                    ),
                ])
                outputs.append(helper.make_tensor_value_info(
                    output, TensorProto.FLOAT, [1, 1370, CHANNELS_PER_BRANCH]
                ))
                original_kernels.append(
                    f"mlp_fc1_l{layer:02d}_c{begin // CHANNELS_PER_BRANCH:02d}"
                )
            graph = helper.make_graph(
                body, name,
                [helper.make_tensor_value_info(
                    "input0", TensorProto.FLOAT, [1, 1370, 384]
                )],
                outputs, values,
            )
            model = helper.make_model(
                graph, opset_imports=copy.deepcopy(source.opset_import),
                producer_name=source.producer_name,
                producer_version=source.producer_version,
            )
            model.ir_version = source.ir_version
            path = args.output_dir / f"{name}.onnx"
            onnx.save_model(
                model, str(path), save_as_external_data=True,
                all_tensors_to_one_file=True, location=path.name + ".data",
                size_threshold=1024,
            )
            records.append({
                "name": name,
                "layer": layer,
                "pair": pair,
                "onnx": path.name,
                "input_shape": [1, 1370, 384],
                "output_shapes": [[1, 1370, CHANNELS_PER_BRANCH]] * 2,
                "replaces": original_kernels,
                "physical_dispatch_reduction": 1,
            })

    report = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "layers": list(layers),
        "branches_per_kernel": BRANCHES_PER_KERNEL,
        "original_dispatches": len(records) * BRANCHES_PER_KERNEL,
        "candidate_dispatches": len(records),
        "physical_dispatch_reduction": len(records),
        "kernels": records,
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest),
        "candidate_dispatches": report["candidate_dispatches"],
        "physical_dispatch_reduction": report["physical_dispatch_reduction"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
