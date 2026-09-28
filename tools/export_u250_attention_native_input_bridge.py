#!/usr/bin/env python3
"""Adapt one 280-token attention kernel to QKV's native output ABI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def replace_initializer(
    model: onnx.ModelProto, name: str, value: np.ndarray
) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def set_shape(value: onnx.ValueInfoProto, dims: list[int]) -> None:
    del value.type.tensor_type.shape.dim[:]
    for value_dim in dims:
        value.type.tensor_type.shape.dim.add().dim_value = value_dim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.source), load_external_data=True)
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {"input0", "input1", "input2", "input3"}:
        raise ValueError("unexpected attention input names")
    # Q1 carries only the 145 real rows.  K keeps its logical [1,64,401]
    # shape; selecting NCHW at compile time gives QKV's physical K ABI.
    set_shape(inputs["input1"], [1, 145, 64])
    set_shape(outputs["output1"], [1, 145, 64])
    replace_initializer(
        model, "/chunk1/output_shape", np.array([1, 145, 64], np.int64)
    )

    name = f"attention2_native_l{args.layer:02d}_h{args.head:02d}_q1x145_knchw"
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
        "policy": "q1-real-rows-and-qkv-native-k-layout",
        "source": str(args.source.resolve()),
        "layer": args.layer,
        "head": args.head,
        "onnx": output.name,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"model": str(output), "q1_rows": 145}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
