#!/usr/bin/env python3
"""Cut block-0 static INT8 attention at instruction-level boundaries."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


CASES = {
    "qk00": ("/blocks.0/attn/MatMul_output_0", [1, 1, 256, 1370]),
    "sm00": ("/blocks.0/attn/Softmax_output_0", [1, 1, 256, 1370]),
    "av00": ("/blocks.0/attn/MatMul_1_output_0", [1, 1, 256, 64]),
    "head0": ("/blocks.0/attn/Reshape_3_output_0", [1, 1370, 64]),
}


def backward_slice(
    model: onnx.ModelProto, output_name: str, shape: list[int]
) -> onnx.ModelProto:
    producer = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    required: set[int] = set()
    required_initializers: set[str] = set()
    pending = [output_name]
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
        if id(node) in required:
            continue
        required.add(id(node))
        pending.extend(input_name for input_name in node.input if input_name)

    graph = helper.make_graph(
        [copy.deepcopy(node) for node in model.graph.node if id(node) in required],
        "block0_attention_" + output_name.rsplit("/", 1)[-1],
        [copy.deepcopy(model.graph.input[0])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
        [
            copy.deepcopy(value)
            for name, value in initializers.items()
            if name in required_initializers
        ],
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
    parser.add_argument("--compiler-model", type=Path, required=True)
    parser.add_argument("--fp-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compiler_model = onnx.load(str(args.compiler_model), load_external_data=True)
    fp_model = onnx.load(str(args.fp_model), load_external_data=True)
    arrays = dict(np.load(args.input, allow_pickle=False))
    input_value = np.asarray(next(iter(arrays.values())), dtype=np.float32)
    if input_value.shape == (1, 1, 1370, 384):
        input_value = input_value[:, 0]

    for case, (output_name, shape) in CASES.items():
        compiler_path = args.output_dir / f"block0_attention_{case}_a8b8.onnx"
        fp_path = args.output_dir / f"block0_attention_{case}_fp.onnx"
        save_external(backward_slice(compiler_model, output_name, shape), compiler_path)
        save_external(backward_slice(fp_model, output_name, shape), fp_path)
        session = ort.InferenceSession(str(fp_path), providers=["CPUExecutionProvider"])
        golden = session.run(None, {session.get_inputs()[0].name: input_value})[0]
        if case == "sm00":
            golden = np.clip(np.rint(golden / np.float32(0.000244140625)), -128, 127).astype(np.int8)
        np.save(args.output_dir / f"block0_attention_{case}_golden.npy", golden)
        print(
            f"{case}: nodes={len(backward_slice(compiler_model, output_name, shape).graph.node)} "
            f"shape={shape} range=[{float(golden.min())}, {float(golden.max())}]"
        )


if __name__ == "__main__":
    main()
