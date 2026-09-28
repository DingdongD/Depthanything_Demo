#!/usr/bin/env python3
"""Create progressively larger block-0 prefix models and FP32 goldens.

The source model already contains the DS compiler's A8xB8/static-attention
attributes.  Each generated model keeps those attributes and cuts the graph at
one semantically useful boundary.  A matching attribute-free FP model is run
with ONNX Runtime to produce the board-comparison golden tensor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


CASES = {
    "ln_q": ("/blocks.0/attn/qkv/q/Add_output_0", [1, 1370, 384]),
    "ln_qkv": ("/blocks.0/attn/qkv/Add_output_0", [1, 1370, 1152]),
    "attention": ("/blocks.0/attn/Concat_6_output_0", [1, 1370, 384]),
    "attention_proj": ("/blocks.0/attn/proj/Add_output_0", [1, 1370, 384]),
    "attention_residual": ("/blocks.0/Add_output_0", [1, 1370, 384]),
    "norm2_fc1": ("/blocks.0/mlp/fc1/Add_output_0", [1, 1370, 1536]),
    "relu025": ("/blocks.0/mlp/act/Mul_1_output_0", [1, 1370, 1536]),
    "block0": ("/blocks.0/Add_1_output_0", [1, 1370, 384]),
}


def backward_slice(model: onnx.ModelProto, output_name: str, shape: list[int]) -> onnx.ModelProto:
    producer = {out: node for node in model.graph.node for out in node.output}
    initializer = {item.name: item for item in model.graph.initializer}
    graph_inputs = {item.name for item in model.graph.input}
    required_nodes: set[str] = set()
    required_initializers: set[str] = set()
    pending = [output_name]
    seen: set[str] = set()

    while pending:
        value = pending.pop()
        if not value or value in seen:
            continue
        seen.add(value)
        if value in graph_inputs:
            continue
        if value in initializer:
            required_initializers.add(value)
            continue
        node = producer.get(value)
        if node is None:
            raise KeyError(f"no producer for required tensor {value!r}")
        required_nodes.add(node.name)
        pending.extend(node.input)

    nodes = [copy.deepcopy(node) for node in model.graph.node if node.name in required_nodes]
    produced = {out for node in nodes for out in node.output}
    value_info = [
        copy.deepcopy(value)
        for value in model.graph.value_info
        if value.name in produced and value.name != output_name
    ]
    inputs = [copy.deepcopy(value) for value in model.graph.input if value.name in seen]
    if len(inputs) != 1 or inputs[0].name != "input0":
        raise ValueError(f"unexpected sliced graph inputs: {[value.name for value in inputs]}")
    inputs[0].type.tensor_type.elem_type = TensorProto.FLOAT
    del inputs[0].type.tensor_type.shape.dim[:]
    for dim in [1, 1370, 384]:
        inputs[0].type.tensor_type.shape.dim.add().dim_value = dim

    graph = helper.make_graph(
        nodes,
        f"depth_anything_v2_block0_prefix_{output_name.rsplit('/', 1)[-1]}",
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
        [copy.deepcopy(initializer[name]) for name in initializer if name in required_initializers],
        value_info=value_info,
    )
    result = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(model.opset_import),
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
    )
    result.ir_version = model.ir_version
    result.metadata_props.extend(copy.deepcopy(model.metadata_props))
    return result


def save_external(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def run_golden(model_path: Path, input_value: np.ndarray, output_path: Path) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: input_value})[0]
    np.save(output_path, output)
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "min": float(output.min()),
        "max": float(output.max()),
        "mean": float(output.mean()),
        "finite": bool(np.isfinite(output).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-model", type=Path, required=True)
    parser.add_argument("--fp-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    args = parser.parse_args()

    requested = args.case or list(CASES)
    compiler_model = onnx.load(str(args.compiler_model), load_external_data=True)
    fp_model = onnx.load(str(args.fp_model), load_external_data=True)
    input_arrays = dict(np.load(args.input, allow_pickle=False))
    input_value = np.asarray(next(iter(input_arrays.values())), dtype=np.float32)
    if input_value.shape == (1, 1, 1370, 384):
        input_value = input_value[:, 0]
    if input_value.shape != (1, 1370, 384):
        raise ValueError(f"unexpected input shape {input_value.shape}")

    manifest = {}
    for case in requested:
        output_name, shape = CASES[case]
        compiler_path = args.output_dir / f"block0_prefix_{case}_a8b8.onnx"
        fp_path = args.output_dir / f"block0_prefix_{case}_fp.onnx"
        golden_path = args.output_dir / f"block0_prefix_{case}_golden.npy"
        save_external(backward_slice(compiler_model, output_name, shape), compiler_path)
        save_external(backward_slice(fp_model, output_name, shape), fp_path)
        stats = run_golden(fp_path, input_value, golden_path)
        manifest[case] = {
            "boundary": output_name,
            "shape": shape,
            "compiler_model": compiler_path.name,
            "fp_model": fp_path.name,
            "golden": golden_path.name,
            "golden_stats": stats,
        }
        print(f"{case}: {output_name} -> {shape}")

    manifest_path = args.output_dir / "block0_bisect_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
