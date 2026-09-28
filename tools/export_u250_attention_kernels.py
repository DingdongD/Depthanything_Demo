#!/usr/bin/env python3
"""Emit calibrated 256x1370 attention kernels for DS-Compiler.

One kernel is emitted per transformer layer/head because the current compiler
bakes all four static scales into instructions.  The same kernel is invoked six
times by the host runtime with different query slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ds_models.static_int8_attention import StaticAttentionProfile, plan_query_chunks  # noqa: E402


def make_kernel(path: Path, q_scale: float, k_scale: float,
                p_scale: float, v_accumulator_scale: float) -> None:
    inputs = [
        helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 256, 64]),
        helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64, 1370]),
        helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 1370, 64]),
    ]
    output = helper.make_tensor_value_info("net_output0", TensorProto.FLOAT, [1, 256, 64])
    nodes = [
        helper.make_node(
            "MatMul", ["input0", "input1"], ["/MatMul_output_0"], name="/MatMul",
            A_bitdepth=8, A_scales=[q_scale], B_bitdepth=8, B_scales=[k_scale],
            output_bitdepth=16, output_scale=-1.0,
        ),
        helper.make_node(
            "Softmax", ["/MatMul_output_0"], ["/Softmax_output_0"],
            name="/Softmax", axis=-1,
            input_bitdepth=16, input_scale=-1.0,
            output_bitdepth=8, output_scales=[p_scale],
        ),
        helper.make_node(
            "MatMul", ["/Softmax_output_0", "input2"], ["net_output0"],
            name="/MatMul_1",
            A_bitdepth=8, A_scales=[p_scale], B_bitdepth=8,
            B_scales=[v_accumulator_scale],
            output_bitdepth=16, output_scale=-1.0,
        ),
    ]
    graph = helper.make_graph(nodes, path.stem, inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    # DS consumes quantization attributes directly on standard ONNX nodes.
    # Upstream ONNX checker intentionally rejects such vendor attributes.
    onnx.save(model, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = StaticAttentionProfile.load(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = []
    for layer in range(len(profile.layers)):
        for head in range(profile.heads):
            scale = profile.scale(layer, head)
            name = f"attention_l{layer:02d}_h{head:02d}"
            path = args.output_dir / f"{name}.onnx"
            make_kernel(
                path, scale.q, scale.k, scale.probability,
                scale.effective_av_value_scale,
            )
            kernels.append({
                "layer": layer,
                "head": head,
                "name": name,
                "onnx": path.name,
                "compiled_prefix": name,
                "scales_bf16": scale.__dict__,
                "av_value_accumulator_scale": scale.effective_av_value_scale,
            })
    chunks = plan_query_chunks(profile.tokens)
    manifest = {
        "schema_version": 1,
        "profile": str(args.profile.resolve()),
        "physical_inputs": {
            "input0": {"shape": [1, 256, 64], "dtype": "int8", "meaning": "Q slice"},
            "input1": {"shape": [1, 64, 1370], "dtype": "int8", "meaning": "full K^T"},
            "input2": {"shape": [1, 1370, 64], "dtype": "int8", "meaning": "full V"},
        },
        "physical_output": {"shape": [1, 256, 64], "dtype": "bf16"},
        "chunks": [c.__dict__ | {"valid_rows": c.valid_rows} for c in chunks],
        "kernels": kernels,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"kernels={len(kernels)}")
    print(f"manifest={(args.output_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
