#!/usr/bin/env python3
"""Cut one attention head into a six-output, one-launch U250 case.

Each query chunk is exposed independently after a rank-4 to rank-3 Reshape.
The Reshape gives the AV MatMul a real downstream consumer before TDDR and the
separate graph outputs let ACPC retire each chunk instead of retaining all six
inside the 4 MiB feature-memory ring for a final Concat.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def attention_av_nodes(model: onnx.ModelProto, block: int) -> list[onnx.NodeProto]:
    producer = {output: node for node in model.graph.node for output in node.output}
    prefix = f"/blocks.{block}/attn/"
    result = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or not node.name.startswith(prefix):
            continue
        lhs = producer.get(node.input[0])
        if lhs is not None and lhs.op_type == "Softmax":
            result.append(node)
    return result


def backward_slice(model: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
    producer = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    required: set[int] = set()
    required_initializers: set[str] = set()
    pending = list(output_names)
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
        pending.extend(name for name in node.input if name)
    graph = helper.make_graph(
        [copy.deepcopy(node) for node in model.graph.node if id(node) in required],
        model.graph.name + "_head_multioutput",
        [copy.deepcopy(model.graph.input[0])],
        [],
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


def append_outputs(
    model: onnx.ModelProto,
    source_names: list[str],
    rows: list[int],
    block: int,
    head: int,
) -> None:
    for chunk, (source, count) in enumerate(zip(source_names, rows)):
        shape_name = f"/DSHead{block}_{head}/chunk{chunk}/shape"
        output_name = f"head{head}_chunk{chunk}"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray([1, count, 64], dtype=np.int64), name=shape_name
            )
        )
        model.graph.node.append(
            helper.make_node(
                "Reshape",
                [source, shape_name],
                [output_name],
                name=f"/DSHead{block}_{head}/chunk{chunk}/Reshape",
            )
        )
        model.graph.output.append(
            helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [1, count, 64]
            )
        )


def save_external(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    compiler_source = onnx.load(str(args.compiler_model), load_external_data=True)
    fp_source = onnx.load(str(args.fp_model), load_external_data=True)
    compiler_av = attention_av_nodes(compiler_source, args.block)
    fp_av = attention_av_nodes(fp_source, args.block)
    chunks_per_head = 6
    begin = args.head * chunks_per_head
    end = begin + chunks_per_head
    if len(compiler_av) < end or len(fp_av) < end:
        raise ValueError("requested block/head does not have six AV nodes")
    compiler_names = [node.output[0] for node in compiler_av[begin:end]]
    fp_names = [node.output[0] for node in fp_av[begin:end]]
    rows = [256, 256, 256, 256, 256, 90]

    compiler = backward_slice(compiler_source, compiler_names)
    fp = backward_slice(fp_source, fp_names)
    append_outputs(compiler, compiler_names, rows, args.block, args.head)
    append_outputs(fp, fp_names, rows, args.block, args.head)
    stem = f"block{args.block}_head{args.head}_multioutput"
    compiler_path = args.output_dir / f"{stem}_a8b8.onnx"
    fp_path = args.output_dir / f"{stem}_fp.onnx"
    save_external(compiler, compiler_path)
    save_external(fp, fp_path)

    values = dict(np.load(args.input, allow_pickle=False))
    input_value = np.asarray(next(iter(values.values())), dtype=np.float32)
    if input_value.shape == (1, 1, 1370, 384):
        input_value = input_value[:, 0]
    session = ort.InferenceSession(str(fp_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: input_value})
    np.savez(
        args.output_dir / f"{stem}_golden.npz",
        **{f"output{index}_bf16": value for index, value in enumerate(outputs)},
    )
    print(f"model={compiler_path}")
    print(f"outputs={[list(value.shape) for value in outputs]}")


if __name__ == "__main__":
    main()
