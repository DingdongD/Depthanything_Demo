#!/usr/bin/env python3
"""Export the small host-side plan/parameters for the hybrid U250 runtime."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def encode_attr(node_index: int, attribute: onnx.AttributeProto,
                params: dict[str, np.ndarray]) -> object:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, onnx.TensorProto):
        key = f"constant_{node_index}_{attribute.name}"
        params[key] = numpy_helper.to_array(value)
        return {"param": key}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--frontend-model", type=Path,
                        help="full RGB-input ONNX providing cls/position tensors")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.model.resolve()), load_external_data=True)
    initializers = {item.name: numpy_helper.to_array(item)
                    for item in model.graph.initializer}
    decoder = json.loads(args.decoder_manifest.read_text())
    conv_by_name = {item["source_node"]: item for item in decoder["kernels"]}
    params: dict[str, np.ndarray] = {}
    frontend = None
    if args.patch_manifest is not None:
        patch = json.loads(args.patch_manifest.read_text())
        frontend_source = onnx.load(
            str((args.frontend_model or args.model).resolve()), load_external_data=True
        )
        frontend_initializers = {
            item.name: numpy_helper.to_array(item)
            for item in frontend_source.graph.initializer
        }
        if {"pretrained.cls_token", "pretrained.pos_embed"} <= frontend_initializers.keys():
            params["pretrained.cls_token"] = frontend_initializers["pretrained.cls_token"]
            params["pretrained.pos_embed"] = frontend_initializers["pretrained.pos_embed"]
        else:
            # The runtime-class-token export folds the learned CLS and the
            # interpolated positional embedding into unnamed initializers and
            # Constant subgraphs.  Capture the two image-independent tensors
            # once instead of relying on exporter-generated initializer names.
            capture = copy.deepcopy(frontend_source)
            capture_names = ["/Add_output_0", "/Cast_output_0"]
            existing = {item.name for item in capture.graph.output}
            for name in capture_names:
                if name not in existing:
                    capture.graph.output.append(
                        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
                    )
            session = ort.InferenceSession(
                capture.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            input_shape = [int(value) for value in session.get_inputs()[0].shape]
            zeros = np.zeros(input_shape, dtype=np.float32)
            cls_token, pos_embed = session.run(capture_names, {
                session.get_inputs()[0].name: zeros
            })
            params["pretrained.cls_token"] = np.ascontiguousarray(cls_token)
            params["pretrained.pos_embed"] = np.ascontiguousarray(pos_embed)
        frontend = {
            "input_shape": list(frontend_source.graph.input[0].type.tensor_type.shape.dim[index].dim_value
                                for index in range(4)),
            "patch_size": 14,
            "patch_grid": [int(patch["patch_grid"]), int(patch["patch_grid"])],
            "patch_channels": 588,
            "projection_kernels": [item["name"] for item in patch["kernels"]],
            "cls_token": "pretrained.cls_token",
            "pos_embed": "pretrained.pos_embed",
        }

    encoder = []
    for layer in range(12):
        norms = []
        for suffix in ("norm1", "norm2"):
            name = f"/blocks.{layer}/{suffix}/LayerNormalization"
            node = next(item for item in model.graph.node if item.name == name)
            for tensor_name in node.input[1:]:
                params[tensor_name] = initializers[tensor_name]
            norms.append({
                "name": node.name, "input": node.input[0], "output": node.output[0],
                "scale": node.input[1], "bias": node.input[2],
                "axis": int(helper.get_attribute_value(next(
                    value for value in node.attribute if value.name == "axis"))),
                "epsilon": float(helper.get_attribute_value(next(
                    value for value in node.attribute if value.name == "epsilon"))),
            })
        encoder.append({"layer": layer, "norm1": norms[0], "norm2": norms[1]})

    start = next(index for index, node in enumerate(model.graph.node)
                 if node.name == "/norm/LayerNormalization")
    steps = []
    for node_index, node in enumerate(model.graph.node[start:], start=start):
        conv = conv_by_name.get(node.name)
        if conv is not None:
            if "safe_channel_slices" in conv:
                kernels = conv["safe_channel_slices"]
                row_tiles = 1
                tile_output_rows = None
                channel_sliced = True
            elif "safe_tile_variants" not in conv:
                kernels = [{"name": conv["name"], "position": "full"}]
                row_tiles = 1
                tile_output_rows = None
                channel_sliced = False
            else:
                kernels = conv["safe_tile_variants"]
                row_tiles = conv["row_tiles"]
                tile_output_rows = conv["tile_output_rows"]
                channel_sliced = False
            steps.append({
                "backend": "npu", "name": node.name,
                "inputs": list(node.input), "outputs": list(node.output),
                "input_scale": conv["input_scale"], "index": conv["index"],
                "row_tiles": row_tiles,
                "tile_output_rows": tile_output_rows,
                "channel_sliced": channel_sliced,
                "kernels": kernels,
            })
            continue
        for tensor_name in node.input:
            if tensor_name in initializers:
                params[tensor_name] = initializers[tensor_name]
        steps.append({
            "backend": "host", "name": node.name, "op_type": node.op_type,
            "inputs": list(node.input), "outputs": list(node.output),
            "attrs": {value.name: encode_attr(node_index, value, params)
                      for value in node.attribute},
        })
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.params.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1, "model_input": model.graph.input[0].name,
        "model_outputs": [item.name for item in model.graph.output],
        "encoder": encoder, "decoder_steps": steps,
        "capture_layers": [2, 5, 8, 11],
        "capture_tensor_names": [f"/blocks.{x}/Add_1_output_0" for x in (2, 5, 8, 11)],
    }
    if frontend is not None:
        plan["frontend"] = frontend
    args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    np.savez(args.params, **params)
    print(json.dumps({"plan": str(args.plan), "steps": len(steps),
                      "params": len(params), "param_bytes": sum(x.nbytes for x in params.values())},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
