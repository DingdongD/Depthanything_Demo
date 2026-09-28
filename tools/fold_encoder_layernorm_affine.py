#!/usr/bin/env python3
"""Fold ViT block LayerNorm gamma/beta into the following linear layers."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import external_data_helper, numpy_helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input.resolve()), load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)

    replacements: dict[str, onnx.TensorProto] = {}
    records = []
    rng = np.random.default_rng(20260904)
    for norm in model.graph.node:
        if norm.op_type != "LayerNormalization" or "/blocks." not in norm.name:
            continue
        if len(norm.input) < 3:
            raise ValueError(f"{norm.name}: affine LayerNorm inputs are missing")
        gamma_name, beta_name = norm.input[1], norm.input[2]
        gamma = numpy_helper.to_array(initializers[gamma_name]).astype(np.float32)
        beta = numpy_helper.to_array(initializers[beta_name]).astype(np.float32)
        if gamma.ndim != 1 or beta.shape != gamma.shape:
            raise ValueError(f"{norm.name}: unsupported affine shapes")

        linears = consumers[norm.output[0]]
        expected = 3 if "/norm1/" in norm.name else 1
        if len(linears) != expected or any(node.op_type != "MatMul" for node in linears):
            raise ValueError(f"{norm.name}: expected {expected} direct MatMul consumers")
        for matmul in linears:
            weight_name = matmul.input[1]
            weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
            if weight.ndim != 2 or weight.shape[0] != gamma.size:
                raise ValueError(f"{matmul.name}: unsupported weight shape {weight.shape}")
            add_nodes = consumers[matmul.output[0]]
            if len(add_nodes) != 1 or add_nodes[0].op_type != "Add":
                raise ValueError(f"{matmul.name}: expected one bias Add")
            add = add_nodes[0]
            bias_names = [name for name in add.input if name in initializers]
            if len(bias_names) != 1:
                raise ValueError(f"{add.name}: expected one initializer bias")
            bias_name = bias_names[0]
            bias = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
            folded_weight = gamma[:, None] * weight
            folded_bias = bias + beta @ weight

            # Verify the algebra locally before changing the graph.
            sample = rng.normal(size=(7, gamma.size)).astype(np.float32)
            before = (sample * gamma + beta) @ weight + bias
            after = sample @ folded_weight + folded_bias
            max_error = float(np.max(np.abs(before - after)))
            replacements[weight_name] = numpy_helper.from_array(
                folded_weight, name=weight_name
            )
            replacements[bias_name] = numpy_helper.from_array(
                folded_bias, name=bias_name
            )
            records.append({
                "layernorm": norm.name, "matmul": matmul.name,
                "weight": weight_name, "bias": bias_name,
                "shape": list(weight.shape), "algebra_max_abs_error": max_error,
            })

        replacements[gamma_name] = numpy_helper.from_array(
            np.ones_like(gamma), name=gamma_name
        )
        replacements[beta_name] = numpy_helper.from_array(
            np.zeros_like(beta), name=beta_name
        )

    if len(records) != 48:
        raise ValueError(f"expected 48 folded linears, got {len(records)}")
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name in replacements:
            model.graph.initializer[index].CopyFrom(replacements[initializer.name])

    external_data_helper.convert_model_from_external_data(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(args.output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=args.output.name + ".data",
        size_threshold=1024, convert_attribute=False,
    )
    report = {
        "schema_version": 1,
        "source_model": str(args.input.resolve()),
        "output_model": str(args.output.resolve()),
        "layernorms": 24,
        "folded_linears": len(records),
        "maximum_algebra_abs_error": max(item["algebra_max_abs_error"] for item in records),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "layernorms", "folded_linears", "maximum_algebra_abs_error"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
