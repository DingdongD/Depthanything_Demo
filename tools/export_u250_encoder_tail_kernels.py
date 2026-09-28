#!/usr/bin/env python3
"""Export post-attention and MLP kernels around host LayerNorm boundaries.

The U250 bitstream cannot execute the encoder LayerNormalization reliably.  A
block is therefore split into QKV, attention, post-attention, host norm2, and
MLP stages.  This exporter creates the latter two compiler-ready stages while
preserving the calibrated A8xB8 attributes from the full model.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper


WIDTH = 384


def attr(node: onnx.NodeProto, name: str) -> float:
    for value in node.attribute:
        if value.name == name:
            data = helper.get_attribute_value(value)
            if isinstance(data, (list, tuple)):
                return float(data[0])
            return float(data)
    raise KeyError(f"{node.name}: missing {name}")


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


def make_submodel(
    source: onnx.ModelProto,
    selected: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    output_name: str,
    graph_name: str,
    initializer_overrides: dict[str, onnx.TensorProto] | None = None,
    output_shape: list[int] | None = None,
    tokens: int = 1370,
) -> onnx.ModelProto:
    initializers = {value.name: value for value in source.graph.initializer}
    required = {
        name
        for node in selected
        for name in node.input
        if name in initializers
    }
    overrides = initializer_overrides or {}
    graph = helper.make_graph(
        selected,
        graph_name,
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT,
                                       output_shape or [1, tokens, WIDTH])],
        [copy.deepcopy(overrides.get(name, initializers[name]))
         for name in sorted(required)],
    )
    model = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(source.opset_import),
        producer_name=source.producer_name,
        producer_version=source.producer_version,
    )
    model.ir_version = source.ir_version
    return model


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-profile", type=Path,
                        help="override encoder MatMul A_scales by node name")
    parser.add_argument("--layers", type=int, nargs="*", default=list(range(12)))
    parser.add_argument("--mlp-chunks", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=1370,
                        help="number of ViT tokens including CLS")
    args = parser.parse_args()
    if args.tokens <= 1:
        raise ValueError("tokens must include CLS and be greater than one")

    source = onnx.load(str(args.model), load_external_data=True)
    by_name = {node.name: node for node in source.graph.node}
    positions = {node.name: i for i, node in enumerate(source.graph.node)}
    initializer_arrays = {
        value.name: numpy_helper.to_array(value) for value in source.graph.initializer
    }
    scale_by_node = {}
    if args.scale_profile is not None:
        profile = json.loads(args.scale_profile.read_text())
        scale_by_node = {
            item["node"]: float(item["a8_scale"])
            for item in profile["operators"]
            if item.get("kind") == "encoder_linear"
        }

    def override_a_scale(node: onnx.NodeProto) -> None:
        if node.name in scale_by_node:
            replace_attr(node, "A_scales", [scale_by_node[node.name]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def vi(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT,
                                             [1, args.tokens, WIDTH])

    for layer in args.layers:
        prefix = f"/blocks.{layer}"

        # attn output projection -> layer scale -> residual add
        post_names = [
            prefix + "/attn/proj/MatMul",
            prefix + "/attn/proj/Add",
            prefix + "/ls1/Mul",
            prefix + "/Add",
        ]
        post = [copy.deepcopy(by_name[name]) for name in post_names]
        override_a_scale(post[0])
        post[0].input[0] = "attention_input"
        post[-1].input[0] = "residual_input"
        # Extracted Eltwise nodes lose the quantizer metadata inferred from
        # their original surrounding graph.  Bind both the BF16 tensor and
        # learned LayerScale constant explicitly for ACPC.
        replace_attr(post[2], "input_bitdepth", 16)
        replace_attr(post[2], "input_scale", -1.0)
        replace_attr(post[2], "weight_bitdepth", 16)
        replace_attr(post[2], "weight_scale", -1.0)
        replace_attr(post[2], "output_bitdepth", 16)
        replace_attr(post[2], "output_scale", -1.0)
        # The original block0 residual comes directly from graph input0 and
        # was annotated by calibration.  Later blocks lose that annotation
        # when extracted, otherwise ACPC sees bitDepth=0 at the residual Add.
        replace_attr(post[-1], "left_bitdepth", 16)
        replace_attr(post[-1], "left_scale", -1.0)
        replace_attr(post[-1], "right_bitdepth", 16)
        replace_attr(post[-1], "right_scale", -1.0)
        replace_attr(post[-1], "output_bitdepth", 16)
        replace_attr(post[-1], "output_scale", -1.0)
        post_output = post[-1].output[0]
        post_model = make_submodel(
            source, post, [vi("attention_input"), vi("residual_input")],
            post_output, f"depthanything_post_attention_l{layer:02d}",
            tokens=args.tokens,
        )
        post_name = f"post_attention_l{layer:02d}"
        save_external(post_model, args.output_dir / (post_name + ".onnx"))

        # host norm2 output -> fc1 -> ReLUApprox -> fc2 -> layer scale.  Keep
        # the final residual add on the host: when an otherwise-unused graph
        # input first appears at the terminal Add, this compiler assigns it
        # bitDepth=0 and aborts in setEltOpIO.
        first_name = prefix + "/mlp/fc1/MatMul"
        last_name = prefix + "/ls2/Mul"
        first = positions[first_name]
        last = positions[last_name]
        mlp = [copy.deepcopy(node) for node in source.graph.node[first:last + 1]]
        for node in mlp:
            override_a_scale(node)
        mlp[0].input[0] = "norm2_input"
        mlp_output = mlp[-1].output[0]
        mlp_model = make_submodel(
            source, mlp, [vi("norm2_input")],
            mlp_output, f"depthanything_mlp_l{layer:02d}",
            tokens=args.tokens,
        )
        mlp_name = f"mlp_l{layer:02d}"
        save_external(mlp_model, args.output_dir / (mlp_name + ".onnx"))

        # Hardware-safe MLP decomposition.  BF16 EPU activation is not valid
        # on the deployed bitstream, so GELU is evaluated by the host between
        # these two A8xB8 projections.
        fc1_nodes = [
            copy.deepcopy(by_name[first_name]),
            copy.deepcopy(by_name[prefix + "/mlp/fc1/Add"]),
        ]
        override_a_scale(fc1_nodes[0])
        fc1_nodes[0].input[0] = "norm2_input"
        fc1_kernel_name = f"mlp_fc1_l{layer:02d}"
        fc1_model = make_submodel(
            source,
            fc1_nodes,
            [vi("norm2_input")],
            fc1_nodes[-1].output[0],
            f"depthanything_{fc1_kernel_name}",
            output_shape=[1, args.tokens, 1536], tokens=args.tokens,
        )
        save_external(fc1_model, args.output_dir / (fc1_kernel_name + ".onnx"))

        fc2_name = prefix + "/mlp/fc2/MatMul"
        fc2_start = positions[fc2_name]
        fc2_last = positions[prefix + "/ls2/Mul"]
        fc2_nodes = [
            copy.deepcopy(node)
            for node in source.graph.node[fc2_start:fc2_last + 1]
        ]
        override_a_scale(fc2_nodes[0])
        fc2_nodes[0].input[0] = "gelu_input"
        layer_scale = fc2_nodes[-1]
        replace_attr(layer_scale, "input_bitdepth", 16)
        replace_attr(layer_scale, "input_scale", -1.0)
        replace_attr(layer_scale, "weight_bitdepth", 16)
        replace_attr(layer_scale, "weight_scale", -1.0)
        replace_attr(layer_scale, "output_bitdepth", 16)
        replace_attr(layer_scale, "output_scale", -1.0)
        fc2_kernel_name = f"mlp_fc2_l{layer:02d}"
        fc2_model = make_submodel(
            source,
            fc2_nodes,
            [helper.make_tensor_value_info("gelu_input", TensorProto.FLOAT,
                                           [1, args.tokens, 1536])],
            fc2_nodes[-1].output[0],
            f"depthanything_{fc2_kernel_name}",
            tokens=args.tokens,
        )
        save_external(fc2_model, args.output_dir / (fc2_kernel_name + ".onnx"))

        if 1536 % args.mlp_chunks:
            raise ValueError("1536 must be divisible by --mlp-chunks")
        chunk_width = 1536 // args.mlp_chunks
        chunk_records = []
        fc2_pos = positions[fc2_name]
        fc1_weight_name = by_name[first_name].input[1]
        fc1_bias_name = by_name[prefix + "/mlp/fc1/Add"].input[0]
        fc2_weight_name = by_name[fc2_name].input[1]
        for chunk in range(args.mlp_chunks):
            start = chunk * chunk_width
            end = start + chunk_width
            chunk_nodes = [
                copy.deepcopy(node)
                for node in source.graph.node[first:fc2_pos + 1]
            ]
            for node in chunk_nodes:
                override_a_scale(node)
            chunk_nodes[0].input[0] = "norm2_input"
            fc1_mm = chunk_nodes[0]
            replace_attr(
                fc1_mm,
                "B_scales",
                list(helper.get_attribute_value(next(
                    item for item in fc1_mm.attribute if item.name == "B_scales"
                ))[start:end]),
            )
            overrides = {
                fc1_weight_name: numpy_helper.from_array(
                    np.ascontiguousarray(initializer_arrays[fc1_weight_name][:, start:end]),
                    fc1_weight_name,
                ),
                fc1_bias_name: numpy_helper.from_array(
                    np.ascontiguousarray(initializer_arrays[fc1_bias_name][start:end]),
                    fc1_bias_name,
                ),
                fc2_weight_name: numpy_helper.from_array(
                    np.ascontiguousarray(initializer_arrays[fc2_weight_name][start:end, :]),
                    fc2_weight_name,
                ),
            }
            fc1_slice_name = f"mlp_fc1_l{layer:02d}_c{chunk:02d}"
            fc1_slice_model = make_submodel(
                source,
                chunk_nodes[:2],
                [vi("norm2_input")],
                chunk_nodes[1].output[0],
                f"depthanything_{fc1_slice_name}",
                initializer_overrides=overrides,
                output_shape=[1, args.tokens, chunk_width], tokens=args.tokens,
            )
            save_external(
                fc1_slice_model,
                args.output_dir / (fc1_slice_name + ".onnx"),
            )
            chunk_name = f"mlp_partial_l{layer:02d}_c{chunk:02d}"
            chunk_model = make_submodel(
                source,
                chunk_nodes,
                [vi("norm2_input")],
                chunk_nodes[-1].output[0],
                f"depthanything_{chunk_name}",
                initializer_overrides=overrides,
                tokens=args.tokens,
            )
            save_external(chunk_model, args.output_dir / (chunk_name + ".onnx"))
            chunk_records.append({
                "name": chunk_name,
                "onnx": chunk_name + ".onnx",
                "fc1_slice_name": fc1_slice_name,
                "fc1_slice_onnx": fc1_slice_name + ".onnx",
                "hidden_start": start,
                "hidden_end": end,
                "nodes": len(chunk_nodes),
            })

        records.append({
            "layer": layer,
            "post_attention": {
                "name": post_name,
                "onnx": post_name + ".onnx",
                "nodes": len(post),
                "attention_input_scale": attr(post[0], "A_scales"),
            },
            "mlp": {
                "name": mlp_name,
                "onnx": mlp_name + ".onnx",
                "nodes": len(mlp),
                "host_residual_add": True,
                "fc1_input_scale": attr(mlp[0], "A_scales"),
                "fc2_input_scale": attr(fc2_nodes[0], "A_scales"),
                "safe_partial_chunks": chunk_records,
                "host_finalize": "sum partials, add fc2 bias, multiply ls2 gamma, add residual",
                "safe_host_gelu": {
                    "fc1_slices": [
                        {
                            "name": item["fc1_slice_name"],
                            "onnx": item["fc1_slice_onnx"],
                            "hidden_start": item["hidden_start"],
                            "hidden_end": item["hidden_end"],
                        }
                        for item in chunk_records
                    ],
                    "gelu": "host exact GELU then quantize with fc2_input_scale",
                    "fc2_name": fc2_kernel_name,
                    "fc2_onnx": fc2_kernel_name + ".onnx",
                    "host_residual_add": True,
                },
            },
        })

    manifest = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "strategy": "host norm1/norm2; calibrated A8xB8 linear kernels",
        "shape": [1, args.tokens, WIDTH],
        "layers": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
