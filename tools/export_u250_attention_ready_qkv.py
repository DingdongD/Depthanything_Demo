#!/usr/bin/env python3
"""Export QKV with per-head INT8 outputs matching attention input ABIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export_u250_head_split_qkv import initializer_map, source_nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--materialize-bf16-qkv", action="store_true",
        help=(
            "preserve the qualified BF16 Add boundary before converting each "
            "head to the calibrated attention INT8 scale"
        ),
    )
    parser.add_argument(
        "--preserve-full-width-projection", action="store_true",
        help=(
            "retain the qualified 384-channel Q/K/V MatMul+Add operations and "
            "apply per-head quantization only after a BF16 channel slice"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer <= 11:
        parser.error("--layer must be in 0..11")
    if args.preserve_full_width_projection and not args.materialize_bf16_qkv:
        parser.error(
            "--preserve-full-width-projection requires --materialize-bf16-qkv"
        )

    contract = json.loads(args.contract.read_text())
    block = contract["encoder"][args.layer]
    heads = block["attention"]["heads"]
    model = onnx.load(str(args.source), load_external_data=True)
    values = initializer_map(model)
    nodes = source_nodes(model, args.layer)
    input_dims = [int(item.dim_value)
                  for item in model.graph.input[0].type.tensor_type.shape.dim]
    if len(input_dims) != 3 or any(value <= 0 for value in input_dims):
        raise ValueError(f"expected static BWC input, got {input_dims}")
    batch, tokens, hidden = input_dims
    if tokens != 401 or hidden % len(heads):
        raise ValueError("attention-ready ABI currently requires 401 tokens")
    head_width = hidden // len(heads)
    input_scale = float(block["qkv"]["input_quantization"]["scale"])
    prefix = f"/blocks.{args.layer}/attn/qkv_attention_ready"
    graph_nodes = []
    initializers = []
    base_outputs = {}
    scales = {}

    if args.preserve_full_width_projection:
        # Keep the exact projection graph that produced the qualified r168
        # BF16 Q/K/V tensors. Splitting output channels into 18 MatMuls changes
        # the accelerator accumulation/rounding path despite being algebraically
        # equivalent.
        used_initializers = set()
        for branch in ("q", "k", "v"):
            source_matmul, source_add = nodes[branch]
            for source_node in (source_matmul, source_add):
                clone = onnx.NodeProto()
                clone.CopyFrom(source_node)
                graph_nodes.append(clone)
                used_initializers.update(
                    name for name in source_node.input if name in values
                )
        for name in sorted(used_initializers):
            initializers.append(numpy_helper.from_array(
                np.ascontiguousarray(values[name]), name=name,
            ))

        for branch in ("q", "k", "v"):
            source_output = nodes[branch][1].output[0]
            for head, head_contract in enumerate(heads):
                begin, end = head * head_width, (head + 1) * head_width
                scale = float(head_contract["scales_bf16"][branch])
                if branch == "v" and (
                        scale != float(head_contract["scales_bf16"]["av_v"])
                        or float(head_contract["scales_bf16"].get(
                            "av_output_gain", 1.0)) != 1.0):
                    raise ValueError(
                        f"head {head}: V requires a unit-gain AV scale"
                    )
                stem = f"{prefix}/{branch}_h{head:02d}"
                slice_inputs = {}
                for suffix, value in {
                    "starts": np.array([begin], np.int64),
                    "ends": np.array([end], np.int64),
                    "axes": np.array([2], np.int64),
                    "steps": np.array([1], np.int64),
                }.items():
                    name = f"{stem}/{suffix}"
                    slice_inputs[suffix] = name
                    initializers.append(numpy_helper.from_array(value, name=name))
                output = stem + "/Slice_head_bf16_output_0"
                graph_nodes.append(helper.make_node(
                    "Slice",
                    [source_output, slice_inputs["starts"], slice_inputs["ends"],
                     slice_inputs["axes"], slice_inputs["steps"]],
                    [output], name=stem + "/Slice_head",
                    output_bitdepth=8, output_scale=-1.0,
                    output_scales=[scale], output_quant_dim=[],
                ))
                base_outputs[(branch, head)] = output
                scales[(branch, head)] = scale
    else:
        for branch in ("q", "k", "v"):
            source_matmul, source_add = nodes[branch]
            weight = values[source_matmul.input[1]]
            bias = values[source_add.input[0]]
            for head, head_contract in enumerate(heads):
                begin, end = head * head_width, (head + 1) * head_width
                head_weight = np.ascontiguousarray(weight[:, begin:end])
                head_bias = np.ascontiguousarray(bias[begin:end])
                scale = float(head_contract["scales_bf16"][branch])
                if branch == "v" and (
                        scale != float(head_contract["scales_bf16"]["av_v"])
                        or float(head_contract["scales_bf16"].get(
                            "av_output_gain", 1.0)) != 1.0):
                    raise ValueError(
                        f"head {head}: V requires a unit-gain AV scale"
                    )
                stem = f"{prefix}/{branch}_h{head:02d}"
                weight_name, bias_name = stem + "/weight", stem + "/bias"
                matmul_output = stem + "/MatMul_output_0"
                add_output = stem + "/Add_output_0"
                projected_output = (
                    stem + "/Add_bf16_output_0"
                    if args.materialize_bf16_qkv else add_output
                )
                weight_scales = np.maximum(
                    np.max(np.abs(head_weight), axis=0) / 127.0,
                    np.finfo(np.float32).tiny,
                )
                projection_add = helper.make_node(
                    "Add", [bias_name, matmul_output], [projected_output],
                    name=stem + "/Add", const_bitdepth=16, const_scale=-1.0,
                    output_bitdepth=8,
                    output_scale=(
                        -1.0 if args.materialize_bf16_qkv else scale
                    ),
                )
                if args.materialize_bf16_qkv:
                    projection_add.attribute.extend([
                        helper.make_attribute("output_scales", [scale]),
                        helper.make_attribute("output_quant_dim", []),
                    ])
                graph_nodes.extend([
                    helper.make_node(
                        "MatMul", ["input0", weight_name], [matmul_output],
                        name=stem + "/MatMul", A_bitdepth=8, B_bitdepth=8,
                        A_scales=[input_scale],
                        B_scales=weight_scales.astype(np.float32).tolist(),
                        B_quant_dim=[1], output_bitdepth=16,
                        output_scale=-1.0,
                    ),
                    projection_add,
                ])
                initializers.extend([
                    numpy_helper.from_array(head_weight, name=weight_name),
                    numpy_helper.from_array(head_bias, name=bias_name),
                ])
                base_outputs[(branch, head)] = projected_output
                scales[(branch, head)] = scale

    outputs = []
    records = []
    for head in range(len(heads)):
        scale_q = scales[("q", head)]
        q = base_outputs[("q", head)]
        q0 = f"{prefix}/h{head:02d}/q0"
        q1_valid = f"{prefix}/h{head:02d}/q1_valid"
        q1 = q1_valid
        constants = {
            "q0_starts": np.array([0], np.int64),
            "q0_ends": np.array([256], np.int64),
            "q1_starts": np.array([256], np.int64),
            "q1_ends": np.array([401], np.int64),
            "axes": np.array([1], np.int64),
            "steps": np.array([1], np.int64),
        }
        names = {}
        for suffix, value in constants.items():
            name = f"{prefix}/h{head:02d}/{suffix}"
            names[suffix] = name
            initializers.append(numpy_helper.from_array(value, name=name))
        q0_slice = helper.make_node(
            "Slice", [q, names["q0_starts"], names["q0_ends"],
                      names["axes"], names["steps"]], [q0],
            name=f"{prefix}/h{head:02d}/Slice_q0",
        )
        q1_slice = helper.make_node(
            "Slice", [q, names["q1_starts"], names["q1_ends"],
                      names["axes"], names["steps"]], [q1_valid],
            name=f"{prefix}/h{head:02d}/Slice_q1",
        )
        graph_nodes.extend([
            q0_slice,
            # Keep q1 in the same BWC layout class as q0.  The compiler ROI
            # allocator now preserves the non-zero byte offset, so this view
            # is both address-correct and directly consumable by a fused
            # attention MatMul without a host materialization.
            q1_slice,
        ])
        # Materialize K^T in the producer.  Re-labelling the token-major
        # [B,T,C] buffer as an attention [B,C,T] input is not a
        # transpose: it changes only metadata and therefore feeds incorrect
        # values to QK.  Unsqueeze preserves the source NDWC storage so DS
        # Transform receives its supported input layout and emits the compact
        # NCHW [B,C,1,T] attention RHS in one operation.
        k = base_outputs[("k", head)]
        k_4d = f"{prefix}/h{head:02d}/k_4d"
        k_transposed = f"{prefix}/h{head:02d}/k_transposed"
        k_axes_name = f"{prefix}/h{head:02d}/k_axes"
        initializers.append(numpy_helper.from_array(
            np.array([3], np.int64), name=k_axes_name,
        ))
        k_transform = helper.make_node(
            "Transpose", [k_4d], [k_transposed],
            name=f"{prefix}/h{head:02d}/Transpose_k",
            perm=[0, 2, 3, 1],
            force_physical_transpose=1,
        )
        graph_nodes.extend([
            helper.make_node(
                "Unsqueeze", [k, k_axes_name], [k_4d],
                name=f"{prefix}/h{head:02d}/Unsqueeze_k_4d",
            ),
            k_transform,
        ])
        v_output = base_outputs[("v", head)]
        ordered = (
            ("q0", q0, [batch, 256, head_width], scale_q),
            ("k", k_transposed, [batch, head_width, 1, tokens],
             scales[("k", head)]),
            ("v", v_output,
             [batch, tokens, head_width], scales[("v", head)]),
            ("q1", q1, [batch, tokens - 256, head_width], scale_q),
        )
        for role, name, shape, scale in ordered:
            outputs.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, shape
            ))
            records.append({
                "head": head, "role": role, "output": name,
                "output_scale": scale,
            })

    del model.graph.node[:]
    model.graph.node.extend(graph_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    name = f"qkv_attention_ready_l{args.layer:02d}_a8_to_24xa8"
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
        "policy": "attention-input-abi-static-head-scales-no-gain",
        "source": str(args.source.resolve()),
        "contract": str(args.contract.resolve()),
        "layer": args.layer,
        "input_scale": input_scale,
        "outputs": records,
        "materialize_bf16_qkv": args.materialize_bf16_qkv,
        "preserve_full_width_projection": args.preserve_full_width_projection,
        "onnx": output.name,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"model": str(output), "outputs": len(outputs)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
