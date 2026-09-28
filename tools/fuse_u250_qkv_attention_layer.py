#!/usr/bin/env python3
"""Fuse one attention-ready QKV producer with all encoder attention heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def set_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(value, name=name))
            return
    raise KeyError(name)


def renamed(name: str, prefix: str, inputs: dict[str, str]) -> str:
    if not name:
        return name
    if name in inputs:
        return inputs[name]
    return prefix + name


def set_node_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    """Replace one ONNX node attribute while preserving all other metadata."""
    retained = [attribute for attribute in node.attribute
                if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


def synchronize_attention_matmul_scales(
    node: onnx.NodeProto, *, q_scale: float, k_scale: float, v_scale: float
) -> None:
    """Bind cloned attention MatMuls to their physical QKV producer ABI."""
    if node.op_type == "MatMul" and node.name.endswith("/QK"):
        set_node_attribute(node, "A_scales", [q_scale])
        set_node_attribute(node, "B_scales", [k_scale])
    elif node.op_type == "MatMul" and (
        node.name.endswith("/DualRangeFineAV")
        or node.name.endswith("/DualRangeResidualAV")
    ):
        set_node_attribute(node, "B_scales", [v_scale])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--qkv-manifest", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument(
        "--emit-qkv", action="store_true",
        help="also expose the requested Q/K/V/Q1 tensors for ABI diagnosis",
    )
    parser.add_argument(
        "--emit-k-matrix", action="store_true",
        help="also expose each K tensor after the fused attention ABI bridge",
    )
    parser.add_argument(
        "--k-bridge",
        choices=("transb", "carrier-direct", "direct-transpose",
                 "carrier-squeeze"),
        default="transb",
        help="physical ABI used to feed K into the fused QK MatMul",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qkv = onnx.load(str(args.qkv), load_external_data=True)
    qkv_manifest = json.loads(args.qkv_manifest.read_text())
    records = qkv_manifest["outputs"]
    by_head_role = {
        (int(record["head"]), record["role"]): record["output"]
        for record in records
    }
    output_scales = {
        (int(record["head"]), record["role"]): float(record["output_scale"])
        for record in records
    }
    qkv_producer = {
        output: node for node in qkv.graph.node for output in node.output
    }
    expected = {(head, role) for head in range(args.heads)
                for role in ("q0", "k", "v", "q1")}
    if not expected.issubset(by_head_role):
        raise ValueError("QKV manifest does not describe the requested attention ABI")

    nodes = list(qkv.graph.node)
    initializers = list(qkv.graph.initializer)
    outputs = []
    k_matrix_outputs = []
    attention_sources = []
    for head in range(args.heads):
        source = args.attention_dir / (
            f"attention2_l{args.layer:02d}_h{head:02d}.onnx"
        )
        attention = onnx.load(str(source), load_external_data=True)
        if {value.name for value in attention.graph.input} != {
                "input0", "input1", "input2", "input3"}:
            raise ValueError(f"{source}: unexpected attention input ABI")
        k_input = next(value for value in attention.graph.input
                       if value.name == "input2")
        k_dims = [int(dim.dim_value)
                  for dim in k_input.type.tensor_type.shape.dim]
        if len(k_dims) != 3 or any(dim <= 0 for dim in k_dims):
            raise ValueError(f"{source}: expected static rank-3 K input")
        set_initializer(
            attention, "/chunk0/output_shape",
            np.array([1, 256, 64], np.int64),
        )
        set_initializer(
            attention, "/chunk1/output_shape",
            np.array([1, 145, 64], np.int64),
        )

        # Preserve the rank-3 ABI used by the independently qualified
        # attention kernels.  K is emitted by QKV as physical NCHW
        # [B,C,1,T]; the marked Squeeze asks the compiler to materialize its
        # logical [B,C,T] matrix as the BWC/NDWC right-hand-side ABI expected
        # by QK.  Q, V and both context outputs remain ordinary BWC tensors.
        prefix = f"/fused_l{args.layer:02d}_h{head:02d}"
        k_matrix = prefix + "/k_matrix"
        q0_input = by_head_role[(head, "q0")]
        q1_input = by_head_role[(head, "q1")]
        v_input = by_head_role[(head, "v")]
        if args.k_bridge in ("carrier-direct", "carrier-squeeze"):
            matrix_axes = prefix + "/k_matrix_axes"
            initializers.append(numpy_helper.from_array(
                np.array([2], np.int64), name=matrix_axes,
            ))
            nodes.append(helper.make_node(
                "Squeeze", [by_head_role[(head, "k")], matrix_axes],
                [k_matrix], name=prefix + (
                    "/Squeeze_k_direct" if args.k_bridge == "carrier-direct"
                    else "/Squeeze_k"
                ),
            ))
        elif args.k_bridge == "direct-transpose":
            # The standalone ABI materializes K as NCHW [B,C,1,T].  Inside a
            # fused graph that creates a needless NCHW round trip.  Recover
            # the native token-major K projection before Unsqueeze/Transpose
            # and perform the one physical transpose QK actually requires.
            carrier = qkv_producer[by_head_role[(head, "k")]]
            if carrier.op_type != "Transpose":
                raise ValueError(f"head {head}: K carrier is not Transpose")
            unsqueeze = qkv_producer[carrier.input[0]]
            if unsqueeze.op_type != "Unsqueeze":
                raise ValueError(f"head {head}: K carrier input is not Unsqueeze")
            k_rhs_4d = prefix + "/k_rhs_4d"
            k_rhs_transposed = prefix + "/k_rhs_transposed"
            k_rhs_unsqueeze_axes = prefix + "/k_rhs_unsqueeze_axes"
            k_rhs_squeeze_axes = prefix + "/k_rhs_squeeze_axes"
            initializers.extend([
                numpy_helper.from_array(
                    np.array([1], np.int64), name=k_rhs_unsqueeze_axes),
                numpy_helper.from_array(
                    np.array([1], np.int64), name=k_rhs_squeeze_axes),
            ])
            nodes.extend([
                helper.make_node(
                    "Unsqueeze", [unsqueeze.input[0], k_rhs_unsqueeze_axes],
                    [k_rhs_4d], name=prefix + "/Unsqueeze_k_rhs",
                ),
                helper.make_node(
                    "Transpose", [k_rhs_4d], [k_rhs_transposed],
                    name=prefix + "/Transpose_k_rhs", perm=[0, 1, 3, 2],
                    force_physical_transpose=1, output_bitdepth=8,
                    output_scale=output_scales[(head, "k")],
                ),
                helper.make_node(
                    "Squeeze", [k_rhs_transposed, k_rhs_squeeze_axes],
                    [k_matrix], name=prefix + "/Squeeze_k_rhs",
                ),
            ])
        else:
            # Keep Q/K/V rank-4 and present QK as MatMul(A, Transpose(K)).
            # ACMOSACombineExt can then lower this exact producer/consumer
            # shape to MatMulwithTransB, avoiding a standalone Transform.
            carrier = qkv_producer[by_head_role[(head, "k")]]
            if carrier.op_type != "Transpose":
                raise ValueError(f"head {head}: K carrier is not Transpose")
            native_unsqueeze = qkv_producer[carrier.input[0]]
            if native_unsqueeze.op_type != "Unsqueeze":
                raise ValueError(f"head {head}: K carrier input is not Unsqueeze")
            axes_name = prefix + "/matrix_axes"
            initializers.append(numpy_helper.from_array(
                np.array([1], np.int64), name=axes_name,
            ))
            q0_input = prefix + "/q0_4d"
            q1_input = prefix + "/q1_4d"
            k_native_4d = prefix + "/k_native_4d"
            nodes.extend([
                helper.make_node(
                    "Unsqueeze", [by_head_role[(head, "q0")], axes_name],
                    [q0_input], name=prefix + "/Unsqueeze_q0",
                ),
                helper.make_node(
                    "Unsqueeze", [by_head_role[(head, "q1")], axes_name],
                    [q1_input], name=prefix + "/Unsqueeze_q1",
                ),
                helper.make_node(
                    "Unsqueeze", [native_unsqueeze.input[0], axes_name],
                    [k_native_4d], name=prefix + "/Unsqueeze_k_native",
                ),
                helper.make_node(
                    "Transpose", [k_native_4d], [k_matrix],
                    name=prefix + "/Transpose_k_transb",
                    perm=[0, 1, 3, 2], force_physical_transpose=1,
                    output_bitdepth=8,
                    output_scale=output_scales[(head, "k")],
                ),
            ])
        input_map = {
            "input0": q0_input,
            "input1": q1_input,
            "input2": k_matrix,
            "input3": v_input,
        }
        if args.emit_k_matrix:
            k_matrix_outputs.append(helper.make_tensor_value_info(
                k_matrix, TensorProto.FLOAT, k_dims,
            ))
        for initializer in attention.graph.initializer:
            clone = onnx.TensorProto()
            clone.CopyFrom(initializer)
            clone.name = prefix + initializer.name
            initializers.append(clone)
        for node in attention.graph.node:
            clone = onnx.NodeProto()
            clone.CopyFrom(node)
            clone.name = prefix + node.name
            # QKV output scales are the physical producer ABI.  Attention
            # source models can predate a later per-head calibration (Block6
            # Head2/4/5 is one real example), so copying their MatMul input
            # scales verbatim silently pairs A8 bytes with the wrong scale.
            # Make the fused consumer contract follow the producer manifest.
            synchronize_attention_matmul_scales(
                clone,
                q_scale=output_scales[(head, "q0")],
                k_scale=output_scales[(head, "k")],
                v_scale=output_scales[(head, "v")],
            )
            del clone.input[:]
            clone.input.extend(renamed(name, prefix, input_map)
                               for name in node.input)
            del clone.output[:]
            mapped_outputs = [prefix + name for name in node.output]
            if args.k_bridge == "transb" and node.name.endswith("/QK"):
                if len(mapped_outputs) != 1:
                    raise ValueError(f"{source}: QK must have one output")
                rank4_output = mapped_outputs[0] + "/rank4"
                clone.output.extend([rank4_output])
                nodes.extend([
                    clone,
                    helper.make_node(
                        "Squeeze", [rank4_output, axes_name],
                        mapped_outputs,
                        name=prefix + node.name + "/Squeeze_logits",
                    ),
                ])
            else:
                clone.output.extend(mapped_outputs)
                nodes.append(clone)
        for index, source_output in enumerate(attention.graph.output):
            rows = 256 if index == 0 else 145
            # Match the tensor name produced by the cloned source node.  The
            # source graph calls these bare `output0`/`output1`, while all
            # internal tensors begin with '/'.
            name = prefix + source_output.name
            outputs.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [1, rows, 64],
            ))
        attention_sources.append(str(source.resolve()))

    diagnostic_outputs = []
    if args.emit_qkv:
        qkv_value_info = {value.name: value for value in qkv.graph.output}
        for head in range(args.heads):
            for role in ("q0", "k", "v", "q1"):
                name = by_head_role[(head, role)]
                diagnostic_outputs.append(qkv_value_info[name])
    outputs = diagnostic_outputs + k_matrix_outputs + outputs

    del qkv.graph.node[:]
    qkv.graph.node.extend(nodes)
    del qkv.graph.initializer[:]
    qkv.graph.initializer.extend(initializers)
    del qkv.graph.output[:]
    qkv.graph.output.extend(outputs)
    name = f"qkv_attention_fused_l{args.layer:02d}_a8_to_12xbf16"
    qkv.graph.name = name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{name}.onnx"
    onnx.save_model(
        qkv, str(output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=output.name + ".data",
        size_threshold=1024,
    )
    manifest = {
        "schema_version": 1,
        "policy": "fused-qkv-six-head-dual-attention-no-gain",
        "layer": args.layer,
        "heads": args.heads,
        "k_bridge": args.k_bridge,
        "qkv": str(args.qkv.resolve()),
        "qkv_manifest": str(args.qkv_manifest.resolve()),
        "attention_sources": attention_sources,
        "onnx": output.name,
        "outputs": [value.name for value in outputs],
        "diagnostic_qkv_outputs": [value.name for value in diagnostic_outputs],
        "diagnostic_k_matrix_outputs": [value.name for value in k_matrix_outputs],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"model": str(output), "outputs": len(outputs)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
