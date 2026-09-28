#!/usr/bin/env python3
"""Combine six calibrated attention heads without changing their input ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

try:
    from .fuse_u250_qkv_attention_layer import (
        synchronize_attention_matmul_scales,
    )
except ImportError:
    from fuse_u250_qkv_attention_layer import (
        synchronize_attention_matmul_scales,
    )


def clone_value_info(value: onnx.ValueInfoProto, name: str) -> onnx.ValueInfoProto:
    clone = onnx.ValueInfoProto()
    clone.CopyFrom(value)
    clone.name = name
    return clone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    block = contract["encoder"][args.layer]
    head_scales = {
        int(head["head"]): head["scales_bf16"]
        for head in block["attention"]["heads"]
    }

    inputs = []
    outputs = []
    nodes = []
    initializers = []
    sources = []
    logical_outputs = []
    opsets = None
    ir_version = None
    for head in range(args.heads):
        source = args.attention_dir / (
            f"attention2_l{args.layer:02d}_h{head:02d}.onnx"
        )
        model = onnx.load(str(source), load_external_data=True)
        if opsets is None:
            opsets = list(model.opset_import)
            ir_version = model.ir_version
        if [value.name for value in model.graph.input] != [
                "input0", "input1", "input2", "input3"]:
            raise ValueError(f"{source}: unexpected calibrated attention ABI")
        prefix = f"/attention6_l{args.layer:02d}_h{head:02d}"
        mapping = {}
        for input_index, value in enumerate(model.graph.input):
            # The compiler layout CLI addresses public inputs as input0..N.
            # Keep that canonical ABI even though internal values are prefixed.
            name = f"input{head * 4 + input_index}"
            mapping[value.name] = name
            inputs.append(clone_value_info(value, name))
        for initializer in model.graph.initializer:
            clone = onnx.TensorProto()
            clone.CopyFrom(initializer)
            clone.name = prefix + initializer.name
            initializers.append(clone)
        for node in model.graph.node:
            clone = onnx.NodeProto()
            clone.CopyFrom(node)
            clone.name = prefix + node.name
            scales = head_scales[head]
            synchronize_attention_matmul_scales(
                clone,
                q_scale=float(scales["q"]),
                k_scale=float(scales["k"]),
                v_scale=float(scales["v"]),
            )
            del clone.input[:]
            clone.input.extend(
                mapping.get(name, prefix + name) for name in node.input
            )
            del clone.output[:]
            clone.output.extend(prefix + name for name in node.output)
            nodes.append(clone)
        for chunk, value in enumerate(model.graph.output):
            name = prefix + value.name
            outputs.append(clone_value_info(value, name))
            logical_outputs.append({
                "graph_output": len(outputs) - 1,
                "head": head,
                "chunk": chunk,
                "name": name,
            })
        sources.append(str(source.resolve()))

    name = f"attention6_l{args.layer:02d}_a8_to_12xbf16"
    graph = onnx.helper.make_graph(
        nodes, name, inputs, outputs, initializer=initializers
    )
    model = onnx.helper.make_model(graph, opset_imports=opsets or [])
    if ir_version is not None:
        model.ir_version = ir_version
    # DS-Compiler consumes custom quantization attributes (for example
    # A_bitdepth/A_scales) that the stock ONNX checker intentionally rejects.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{name}.onnx"
    onnx.save_model(
        model, str(output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=output.name + ".data",
        size_threshold=1024,
    )
    manifest = {
        "schema_version": 1,
        "policy": "six-head-attention-fusion-preserve-calibrated-a8-abi-no-gain",
        "layer": args.layer,
        "heads": args.heads,
        "inputs_per_head": 4,
        "outputs_per_head": 2,
        "input_order": "head-major",
        "graph_output_order": "head-major",
        "logical_outputs": logical_outputs,
        "sources": sources,
        "contract": str(args.contract.resolve()),
        "onnx": output.name,
        "amplitude_gain": 1.0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "inputs": len(inputs), "model": str(output), "outputs": len(outputs),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
