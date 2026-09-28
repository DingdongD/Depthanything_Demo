#!/usr/bin/env python3
"""Extract the 12 fused A8 Q/K/V projection kernels from the ViT tail."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def attr(node: onnx.NodeProto, name: str):
    for value in node.attribute:
        if value.name == name:
            return helper.get_attribute_value(value)
    raise KeyError(f"{node.name}: missing {name}")


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def save_external(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-profile", type=Path)
    parser.add_argument("--tokens", type=int, default=1370,
                        help="number of ViT tokens including CLS")
    parser.add_argument("--head-dim", type=int, default=64,
                        help="attention head width used by the model's Q scale")
    parser.add_argument(
        "--output-scale-manifest", type=Path,
        help="known-good Q/K/V manifest whose INT8 output scales are restored",
    )
    args = parser.parse_args()
    if args.tokens <= 1:
        raise ValueError("tokens must include CLS and be greater than one")
    if args.head_dim <= 0:
        raise ValueError("head dimension must be positive")
    source = onnx.load(str(args.model), load_external_data=True)
    nodes = {node.name: node for node in source.graph.node}
    initializers = {value.name: value for value in source.graph.initializer}
    profile = (json.loads(args.scale_profile.read_text())
               if args.scale_profile is not None else None)
    input_scales = ({item["node"]: float(item["a8_scale"])
                     for item in profile["operators"]
                     if item["kind"] == "encoder_linear"}
                    if profile is not None else {})
    output_scales = None
    if args.output_scale_manifest is not None:
        reference = json.loads(args.output_scale_manifest.read_text())
        output_scales = {
            int(item["layer"]): {
                branch: float(item["output_scales"][branch])
                for branch in ("q", "k", "v")
            }
            for item in reference["kernels"]
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = []
    for block in range(12):
        selected = []
        required = set()
        initializer_overrides = {}
        outputs = []
        scales = {}
        for branch in ("q", "k", "v"):
            prefix = f"/blocks.{block}/attn/qkv/{branch}"
            mm = copy.deepcopy(nodes[prefix + "/MatMul"])
            if mm.name in input_scales:
                replace_attr(mm, "A_scales", [input_scales[mm.name]])
            add = copy.deepcopy(nodes[prefix + "/Add"])
            if branch == "q":
                # DSAttention4D applies q *= 1/sqrt(head_dim) after the QKV
                # projection.  Extraction cuts before that Mul, so fold this
                # model-defined factor into Q weights and bias exactly.  The
                # calibration profile already observes scaled Q.
                q_factor = np.float32(1.0 / math.sqrt(args.head_dim))
                weight_name = mm.input[1]
                weight = numpy_helper.to_array(initializers[weight_name])
                initializer_overrides[weight_name] = numpy_helper.from_array(
                    np.ascontiguousarray(weight * q_factor), name=weight_name
                )
                replace_attr(mm, "B_scales", [
                    float(value) * float(q_factor)
                    for value in attr(mm, "B_scales")
                ])
                bias_name = next(
                    name for name in add.input if name in initializers
                )
                bias = numpy_helper.to_array(initializers[bias_name])
                initializer_overrides[bias_name] = numpy_helper.from_array(
                    np.ascontiguousarray(bias * q_factor), name=bias_name
                )
            if output_scales is not None:
                scale = output_scales[block][branch]
                if not scale > 0.0:
                    raise ValueError(
                        f"layer {block} {branch}: invalid reference output scale {scale}")
                replace_attr(add, "output_bitdepth", 8)
                replace_attr(add, "output_scale", scale)
            mm.input[0] = "input0"
            selected.extend((mm, add))
            required.update(name for name in (*mm.input, *add.input)
                            if name in initializers)
            outputs.append(add.output[0])
            output_scale = float(attr(add, "output_scale"))
            scales[branch] = output_scale if output_scale > 0.0 else None
        graph = helper.make_graph(
            selected, f"depthanything_block{block}_qkv_a8",
            [helper.make_tensor_value_info("input0", TensorProto.FLOAT,
                                           [1, args.tokens, 384])],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                           [1, args.tokens, 384]) for name in outputs],
            [copy.deepcopy(initializer_overrides.get(name, initializers[name]))
             for name in sorted(required)],
        )
        model = helper.make_model(graph,
                                  opset_imports=copy.deepcopy(source.opset_import),
                                  producer_name=source.producer_name,
                                  producer_version=source.producer_version)
        model.ir_version = source.ir_version
        name = f"qkv_projection_l{block:02d}"
        path = args.output_dir / f"{name}.onnx"
        save_external(model, path)
        kernels.append({"layer": block, "name": name, "onnx": path.name,
                        "input_shape": [1, args.tokens, 384],
                        "input_scale": float(attr(selected[0], "A_scales")[0]),
                        "output_shapes": [[1, args.tokens, 384]] * 3,
                        "output_scales": scales})
    manifest = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "strategy": "host LayerNorm boundary; exact model Q scale; unit AV gain",
        "query_scale": 1.0 / math.sqrt(args.head_dim),
        "output_scale_manifest": (
            str(args.output_scale_manifest.resolve())
            if args.output_scale_manifest is not None else None),
        "kernels": kernels,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
