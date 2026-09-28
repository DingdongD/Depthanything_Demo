#!/usr/bin/env python3
"""Create direct Q/K/V projection cut points for U250 localization."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


BRANCHES = {
    "q": "/blocks.0/attn/qkv/q/Add_output_0",
    "k": "/blocks.0/attn/qkv/k/Add_output_0",
    "v": "/blocks.0/attn/qkv/v/Add_output_0",
}


def slice_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    producer = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    required_nodes: set[int] = set()
    required_initializers: set[str] = set()
    pending = list(names)
    while pending:
        tensor = pending.pop()
        if tensor in graph_inputs:
            continue
        if tensor in initializers:
            required_initializers.add(tensor)
            continue
        node = producer.get(tensor)
        if node is None:
            raise KeyError(f"no producer for {tensor!r}")
        if id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        pending.extend(input_name for input_name in node.input if input_name)

    nodes = [copy.deepcopy(node) for node in model.graph.node if id(node) in required_nodes]
    input_value = copy.deepcopy(model.graph.input[0])
    input_value.type.tensor_type.elem_type = TensorProto.FLOAT
    del input_value.type.tensor_type.shape.dim[:]
    for dim in (1, 1370, 384):
        input_value.type.tensor_type.shape.dim.add().dim_value = dim
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1370, 384])
        for name in names
    ]
    graph = helper.make_graph(
        nodes,
        "block0_direct_" + "".join(name.rsplit("/", 2)[-2][0] for name in names),
        [input_value],
        outputs,
        [copy.deepcopy(value) for key, value in initializers.items() if key in required_initializers],
    )
    result = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(model.opset_import),
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    result.ir_version = model.ir_version
    return result


def save_external(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(str(path) + ".data")
    for old in (path, data_path):
        if old.exists():
            old.unlink()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.model), load_external_data=True)
    arrays = dict(np.load(args.input, allow_pickle=False))
    input_value = np.asarray(next(iter(arrays.values())), dtype=np.float32)
    if input_value.shape == (1, 1, 1370, 384):
        input_value = input_value[:, 0]
    if input_value.shape != (1, 1370, 384):
        raise ValueError(f"unexpected input shape {input_value.shape}")

    for case, branches in (("q", ["q"]), ("k", ["k"]), ("v", ["v"]), ("qkv", ["q", "k", "v"])):
        path = args.output_dir / f"block0_{case}_from_norm1_a8b8.onnx"
        sliced = slice_outputs(model, [BRANCHES[branch] for branch in branches])
        save_external(sliced, path)
        fp = copy.deepcopy(sliced)
        for node in fp.graph.node:
            del node.attribute[:]
        fp_path = args.output_dir / f"block0_{case}_from_norm1_fp.onnx"
        save_external(fp, fp_path)
        session = ort.InferenceSession(str(fp_path), providers=["CPUExecutionProvider"])
        golden = session.run(None, {session.get_inputs()[0].name: input_value})
        np.savez(args.output_dir / f"block0_{case}_from_norm1_golden.npz", **{
            branch: value for branch, value in zip(branches, golden)
        })
        print(f"{case}: outputs={branches} model={path}")


if __name__ == "__main__":
    main()
