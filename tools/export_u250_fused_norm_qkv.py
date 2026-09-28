#!/usr/bin/env python3
"""Export a pure LayerNorm core fused with affine-folded Q/K/V projections."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: numpy_helper.to_array(item).astype(np.float32, copy=False)
        for item in model.graph.initializer
    }


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for item in model.graph.initializer:
        if item.name == name:
            item.CopyFrom(numpy_helper.from_array(
                np.ascontiguousarray(value, dtype=np.float32), name=name
            ))
            return
    raise KeyError(name)


def calibrated_core_values(
    traces: list[Path], layer: int, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    values = []
    key = f"encoder_l{layer:02d}_norm1"
    for path in traces:
        with np.load(path, allow_pickle=False) as archive:
            affine = np.asarray(archive[key], dtype=np.float32)
        values.append(np.abs((affine - beta) / gamma).reshape(-1))
    if not values:
        raise ValueError("at least one calibration trace is required")
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("calibrated LayerNorm core contains non-finite values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--percentiles", default="99.9,99.99,100")
    parser.add_argument(
        "--projection-layout", choices=("branched", "combined"),
        default="branched",
        help="Use one 384->1152 MatMul to avoid a shared-LayerNorm fanout",
    )
    parser.add_argument(
        "--layernorm-output-int8", action="store_true",
        help="Quantize in the SPU LayerNorm output instead of a later Stick",
    )
    parser.add_argument(
        "--omit-layernorm", action="store_true",
        help="Export only affine-folded QKV consuming a pre-normalized core",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer <= 11:
        parser.error("--layer must be in 0..11")
    try:
        percentiles = tuple(float(item) for item in args.percentiles.split(","))
    except ValueError as error:
        parser.error("--percentiles must be comma-separated numbers")
    if not percentiles or any(not 0.0 < item <= 100.0 for item in percentiles):
        parser.error("percentiles must be in (0, 100]")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    prefix = f"pretrained.blocks.{args.layer}.norm1"
    gamma = state[f"{prefix}.weight"].cpu().numpy().astype(np.float32)
    beta = state[f"{prefix}.bias"].cpu().numpy().astype(np.float32)
    if np.any(gamma == 0.0):
        raise ValueError("LayerNorm gamma contains zero and cannot be inverted")
    core_values = calibrated_core_values(args.trace, args.layer, gamma, beta)

    source = onnx.load(str(args.source), load_external_data=True)
    source_values = initializer_map(source)
    qkv_prefix = f"/blocks.{args.layer}/attn/qkv"
    fold_records = []
    algebra_max = 0.0
    rng = np.random.default_rng(20260913 + args.layer)
    sample = rng.normal(size=(17, gamma.size)).astype(np.float32)
    core = rng.normal(size=(17, gamma.size)).astype(np.float32)
    for branch in ("q", "k", "v"):
        weight_name = f"{qkv_prefix}/{branch}/weight"
        bias_name = f"{qkv_prefix}/{branch}/bias"
        weight = source_values[weight_name]
        bias = source_values[bias_name]
        if weight.shape != (gamma.size, gamma.size) or bias.shape != gamma.shape:
            raise ValueError(f"{branch}: unexpected projection shape")
        folded_weight = gamma[:, None] * weight
        folded_bias = beta @ weight + bias
        expected = (sample * gamma + beta) @ weight + bias
        actual = sample @ folded_weight + folded_bias
        algebra_max = max(algebra_max, float(np.max(np.abs(expected - actual))))
        fold_records.append((weight_name, bias_name, folded_weight, folded_bias))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for percentile in percentiles:
        scale = float(np.percentile(core_values, percentile) / 127.0)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid scale for percentile {percentile}")
        model = copy.deepcopy(source)
        for weight_name, bias_name, weight, bias in fold_records:
            replace_initializer(model, weight_name, weight)
            replace_initializer(model, bias_name, bias)
        norm_output = f"/blocks.{args.layer}/norm1/core_output"
        norm_scale = f"/blocks.{args.layer}/norm1/core_scale"
        norm_bias = f"/blocks.{args.layer}/norm1/core_bias"
        norm = helper.make_node(
            "LayerNormalization", ["input0", norm_scale, norm_bias],
            [norm_output], name=f"/blocks.{args.layer}/norm1/core",
            axis=-1, epsilon=1.0e-6, input_bitdepth=16, input_scale=-1.0,
            output_bitdepth=(8 if args.layernorm_output_int8 else 16),
            output_scale=(scale if args.layernorm_output_int8 else -1.0),
        )
        if args.projection_layout == "branched":
            for node in model.graph.node:
                if node.op_type == "MatMul":
                    if node.input[0] != "input0":
                        raise ValueError(f"unexpected MatMul input in {node.name}")
                    if not args.omit_layernorm:
                        node.input[0] = norm_output
                    replace_attribute(node, "A_scales", [scale])
                    weight = next(item[2] for item in fold_records
                                  if item[0] == node.input[1])
                    channel_scales = np.maximum(
                        np.max(np.abs(weight), axis=0) / 127.0,
                        np.finfo(np.float32).tiny,
                    )
                    replace_attribute(node, "B_scales", channel_scales.tolist())
            if not args.omit_layernorm:
                model.graph.node.insert(0, norm)
        else:
            # A single projection consumes the LayerNorm result exactly once.
            # This avoids the compiler's three-consumer intermediate-buffer
            # hazard.  Runtime slicing of the last dimension is a view.
            weights = np.concatenate([item[2] for item in fold_records], axis=1)
            biases = np.concatenate([item[3] for item in fold_records], axis=0)
            weight_name = f"{qkv_prefix}/combined/weight"
            bias_name = f"{qkv_prefix}/combined/bias"
            matmul_output = f"{qkv_prefix}/combined/MatMul_output_0"
            output_name = f"{qkv_prefix}/combined/Add_output_0"
            template_matmul = next(
                copy.deepcopy(node) for node in model.graph.node
                if node.op_type == "MatMul"
            )
            template_add = next(
                copy.deepcopy(node) for node in model.graph.node
                if node.op_type == "Add"
            )
            template_matmul.name = f"{qkv_prefix}/combined/MatMul"
            template_matmul.input[:] = [
                "input0" if args.omit_layernorm else norm_output, weight_name
            ]
            template_matmul.output[:] = [matmul_output]
            replace_attribute(template_matmul, "A_scales", [scale])
            replace_attribute(
                template_matmul, "B_scales",
                np.maximum(
                    np.max(np.abs(weights), axis=0) / 127.0,
                    np.finfo(np.float32).tiny,
                ).tolist(),
            )
            template_add.name = f"{qkv_prefix}/combined/Add"
            template_add.input[:] = [matmul_output, bias_name]
            template_add.output[:] = [output_name]
            del model.graph.node[:]
            model.graph.node.extend(
                ([norm] if not args.omit_layernorm else [])
                + [template_matmul, template_add]
            )
            model.graph.initializer.extend([
                numpy_helper.from_array(
                    np.ascontiguousarray(weights), name=weight_name
                ),
                numpy_helper.from_array(
                    np.ascontiguousarray(biases), name=bias_name
                ),
            ])
            del model.graph.output[:]
            model.graph.output.extend([
                helper.make_tensor_value_info(
                    output_name, TensorProto.FLOAT,
                    [1, 401, gamma.size * 3],
                )
            ])
        if not args.omit_layernorm:
            model.graph.initializer.extend([
                numpy_helper.from_array(np.ones_like(gamma), name=norm_scale),
                numpy_helper.from_array(np.zeros_like(beta), name=norm_bias),
            ])
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
        tag = str(percentile).replace(".", "p")
        suffix = "_combined" if args.projection_layout == "combined" else ""
        if args.layernorm_output_int8:
            suffix += "_ln8"
        prefix_name = "folded_qkv" if args.omit_layernorm else "norm1_qkv"
        name = f"{prefix_name}_l{args.layer:02d}_p{tag}{suffix}"
        model.graph.name = name
        output = args.output_dir / f"{name}.onnx"
        onnx.save_model(
            model, str(output), save_as_external_data=True,
            all_tensors_to_one_file=True, location=output.name + ".data",
            size_threshold=1024,
        )
        records.append({
            "name": name, "onnx": output.name, "layer": args.layer,
            "percentile": percentile, "input_scale": scale,
        })

    manifest = {
        "schema_version": 1,
        "strategy": "SPU LayerNorm core + affine-folded A8xB8 QKV",
        "projection_layout": args.projection_layout,
        "layernorm_output_int8": args.layernorm_output_int8,
        "omit_layernorm": args.omit_layernorm,
        "source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "calibration_traces": [str(path.resolve()) for path in args.trace],
        "layer": args.layer,
        "algebra_max_abs_error": algebra_max,
        "core_abs_percentiles": {
            str(value): float(np.percentile(core_values, value))
            for value in percentiles
        },
        "kernels": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
