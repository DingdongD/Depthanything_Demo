#!/usr/bin/env python3
"""Export fused decoder LayerNorm/layout/project-Conv kernels for U250.

The U250 SPU LayerNorm instruction emits the normalization core.  Decoder
LayerNorm affine parameters are therefore folded exactly into the following
1x1 project convolution.  Slice/Transpose/Reshape remain inside the compiled
graph so a resident BF16 token tensor never has to be materialized by the host.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CAPTURE_LAYERS = (2, 5, 8, 11)
OUTPUT_CHANNEL_LIMIT = 64


def attribute(node: onnx.NodeProto, name: str, default: object = None) -> object:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return default


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def layernorm_core_scale(paths: list[Path], layer: int, percentile: float) -> float:
    maxima = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            value = np.asarray(archive[f"capture_l{layer:02d}"], dtype=np.float32)
        mean = np.mean(value, axis=-1, keepdims=True, dtype=np.float32)
        variance = np.mean(np.square(value - mean), axis=-1,
                           keepdims=True, dtype=np.float32)
        core = (value - mean) / np.sqrt(variance + np.float32(1.0e-6))
        maxima.append(np.abs(core).reshape(-1))
    if not maxima:
        raise ValueError("at least one --capture-npz is required")
    threshold = float(np.percentile(np.concatenate(maxima), percentile))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"layer {layer}: invalid calibration threshold {threshold}")
    return threshold / 127.0


def save_model(model: onnx.ModelProto, path: Path) -> None:
    onnx.save_model(
        model, str(path), save_as_external_data=True,
        all_tensors_to_one_file=True, location=path.name + ".data",
        size_threshold=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture-npz", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument(
        "--affine-mode", choices=("epu", "fold"), default="epu",
        help="keep gamma/beta as BF16 EPU Mul/Add, or fold them into Conv",
    )
    parser.add_argument(
        "--stem-input", choices=("capture", "normalized", "patches"),
        default="capture",
        help="start from the encoder capture or an already normalized token tensor",
    )
    args = parser.parse_args()
    if not 0.0 < args.percentile <= 100.0:
        parser.error("--percentile must be in (0, 100]")

    source = onnx.load(str(args.model), load_external_data=True)
    nodes = {node.name: node for node in source.graph.node}
    initializers = {item.name: item for item in source.graph.initializer}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    maximum_algebra_error = 0.0
    rng = np.random.default_rng(20260908)

    for index, layer in enumerate(CAPTURE_LAYERS):
        norm = copy.deepcopy(nodes["/norm" + ("" if index == 0 else f"_{index}")
                                   + "/LayerNormalization"])
        slice_node = copy.deepcopy(nodes[f"/Slice_{index + 1}"])
        transpose = copy.deepcopy(nodes["/depth_head/Transpose"
                                        + ("" if index == 0 else f"_{index}")])
        reshape = copy.deepcopy(nodes["/depth_head/Reshape"
                                      + ("" if index == 0 else f"_{index}")])
        conv = copy.deepcopy(nodes[f"/depth_head/projects.{index}/Conv"])

        gamma = numpy_helper.to_array(initializers[norm.input[1]]).astype(np.float32)
        beta = numpy_helper.to_array(initializers[norm.input[2]]).astype(np.float32)
        weight = numpy_helper.to_array(initializers[conv.input[1]]).astype(np.float32)
        bias = numpy_helper.to_array(initializers[conv.input[2]]).astype(np.float32)
        if weight.shape[1:] != (gamma.size, 1, 1) or beta.shape != gamma.shape:
            raise ValueError(f"project {index}: unsupported affine/Conv shapes")
        folded_weight = weight * gamma[None, :, None, None]
        folded_bias = bias + weight[:, :, 0, 0] @ beta

        sample = rng.normal(size=(7, gamma.size)).astype(np.float32)
        before = (sample * gamma + beta) @ weight[:, :, 0, 0].T + bias
        after = sample @ folded_weight[:, :, 0, 0].T + folded_bias
        algebra_error = float(np.max(np.abs(before - after)))
        maximum_algebra_error = max(maximum_algebra_error, algebra_error)

        norm.input[0] = "input0"
        norm_scale_name = f"decoder_stem_{index}_norm_scale"
        norm_bias_name = f"decoder_stem_{index}_norm_bias"
        norm.input[1:] = [norm_scale_name, norm_bias_name]
        replace_attribute(norm, "input_bitdepth", 16)
        replace_attribute(norm, "input_scale", -1.0)
        replace_attribute(norm, "output_bitdepth", 16)
        replace_attribute(norm, "output_scale", -1.0)
        common_initializers = {
            norm_scale_name: numpy_helper.from_array(
                np.ones_like(gamma), name=norm_scale_name),
            norm_bias_name: numpy_helper.from_array(
                np.zeros_like(beta), name=norm_bias_name),
        }
        stem_nodes = [norm] if args.stem_input == "capture" else []
        if args.stem_input in ("normalized", "patches"):
            common_initializers = {}
            if args.stem_input == "normalized":
                slice_node.input[0] = "input0"
            else:
                transpose.input[0] = "input0"
        elif args.affine_mode == "epu":
            core_name = norm.output[0] + "/core"
            affine_mul_name = norm.output[0] + "/affine_mul"
            affine_output_name = norm.output[0]
            norm.output[0] = core_name
            gamma_name = f"decoder_stem_{index}_gamma"
            beta_name = f"decoder_stem_{index}_beta"
            common_initializers[gamma_name] = numpy_helper.from_array(
                gamma, name=gamma_name)
            common_initializers[beta_name] = numpy_helper.from_array(
                beta, name=beta_name)
            stem_nodes.extend([
                helper.make_node(
                    "Mul", [core_name, gamma_name], [affine_mul_name],
                    name=f"decoder_stem_{index}/affine_mul",
                    weight_bitdepth=16, weight_scale=-1.0,
                    output_bitdepth=16, output_scale=-1.0,
                ),
                helper.make_node(
                    "Add", [affine_mul_name, beta_name], [affine_output_name],
                    name=f"decoder_stem_{index}/affine_add",
                    const_bitdepth=16, const_scale=-1.0,
                    output_bitdepth=16, output_scale=-1.0,
                ),
            ])
        for name in slice_node.input[1:]:
            producer = next((item for item in source.graph.node
                             if name in item.output), None)
            if producer is None or producer.op_type != "Constant":
                raise ValueError(f"{slice_node.name}: unresolved Slice input {name}")
            tensor = attribute(producer, "value")
            common_initializers[name] = copy.deepcopy(tensor)
            common_initializers[name].name = name
        shape_name = reshape.input[1]
        shape_producer = next((item for item in source.graph.node
                               if shape_name in item.output), None)
        if shape_producer is None or shape_producer.op_type != "Constant":
            raise ValueError(f"{reshape.name}: unresolved shape input")
        common_initializers[shape_name] = copy.deepcopy(
            attribute(shape_producer, "value"))
        common_initializers[shape_name].name = shape_name

        if args.stem_input in ("normalized", "patches"):
            kernel_weight = weight
            kernel_bias = bias
            input_scale = float(attribute(conv, "input_scales")[0])
        elif args.affine_mode == "epu":
            kernel_weight = weight
            kernel_bias = bias
            input_scale = float(attribute(conv, "input_scales")[0])
        else:
            kernel_weight = folded_weight
            kernel_bias = folded_bias
            input_scale = layernorm_core_scale(
                args.capture_npz, layer, args.percentile)
        output_channels = int(weight.shape[0])
        for start in range(0, output_channels, OUTPUT_CHANNEL_LIMIT):
            end = min(start + OUTPUT_CHANNEL_LIMIT, output_channels)
            sliced_conv = copy.deepcopy(conv)
            sliced_weight = np.ascontiguousarray(kernel_weight[start:end])
            sliced_bias = np.ascontiguousarray(kernel_bias[start:end])
            weight_name = f"decoder_stem_{index}_weight_{start}_{end}"
            bias_name = f"decoder_stem_{index}_bias_{start}_{end}"
            sliced_conv.input[1:] = [weight_name, bias_name]
            replace_attribute(sliced_conv, "input_scales", [input_scale])
            channel_scales = np.max(np.abs(sliced_weight), axis=(1, 2, 3)) / 127.0
            channel_scales = np.maximum(channel_scales, np.finfo(np.float32).tiny)
            replace_attribute(sliced_conv, "weight_ch_scales",
                              channel_scales.astype(np.float32).tolist())
            prefix = ("decoder_stem" if args.stem_input == "capture" else
                      "decoder_layout_project" if args.stem_input == "normalized"
                      else "decoder_patch_project")
            name = f"{prefix}_{index}_co{start:03d}_{end:03d}"
            sliced_conv.name = name + "/project"
            sliced_conv.output[0] = name + "/output"
            graph_initializers = list(copy.deepcopy(common_initializers))
            values = [copy.deepcopy(common_initializers[key])
                      for key in graph_initializers]
            values.extend([
                numpy_helper.from_array(sliced_weight, name=weight_name),
                numpy_helper.from_array(sliced_bias, name=bias_name),
            ])
            body_nodes = ([copy.deepcopy(slice_node), copy.deepcopy(transpose),
                           copy.deepcopy(reshape)] if args.stem_input != "patches"
                          else [copy.deepcopy(transpose), copy.deepcopy(reshape)])
            input_rows = 1369 if args.stem_input == "patches" else 1370
            graph = helper.make_graph(
                [*copy.deepcopy(stem_nodes), *body_nodes, sliced_conv],
                name,
                [helper.make_tensor_value_info(
                    "input0", TensorProto.FLOAT, [1, input_rows, 384])],
                [helper.make_tensor_value_info(
                    sliced_conv.output[0], TensorProto.FLOAT,
                    [1, end - start, 37, 37])],
                values,
            )
            model = helper.make_model(
                graph, opset_imports=copy.deepcopy(source.opset_import),
                producer_name=source.producer_name,
                producer_version=source.producer_version,
            )
            model.ir_version = source.ir_version
            path = args.output_dir / f"{name}.onnx"
            save_model(model, path)
            records.append({
                "name": name, "onnx": path.name, "capture_layer": layer,
                "project_index": index, "channel_start": start,
                "channel_end": end, "input_shape": [1, input_rows, 384],
                "input_tensor": (slice_node.output[0]
                                 if args.stem_input == "patches" else
                                 f"/blocks.{layer}/Add_1_output_0"
                                 if args.stem_input == "capture" else
                                 nodes["/norm" + ("" if index == 0 else f"_{index}")
                                       + "/LayerNormalization"].output[0]),
                "output_shape": [1, end - start, 37, 37],
                "input_precision": "BF16", "weight_precision": "INT8",
                "layernorm_core_precision": (
                    "BF16" if args.stem_input == "capture" else None
                ),
                "project_input_scale": input_scale,
                "layernorm_affine": (
                    "preapplied_on_host" if args.stem_input != "capture" else
                    "epu_bf16_mul_add" if args.affine_mode == "epu" else
                    "folded_into_project_conv"
                ),
                "layout_ops": (["SliceCLS", "Transpose", "Reshape"]
                               if args.stem_input != "patches" else
                               ["Transpose", "Reshape"]),
            })

    report = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "capture_npz": [str(path.resolve()) for path in args.capture_npz],
        "calibration_percentile": args.percentile,
        "strategy": ("host LayerNorm/Slice + device layout + A8xB8 project Conv"
                     if args.stem_input == "patches" else
                     "host LayerNorm + device layout + A8xB8 project Conv"
                     if args.stem_input == "normalized" else
                     "SPU LayerNorm core + EPU BF16 affine + device layout + "
                     "A8xB8 project Conv" if args.affine_mode == "epu" else
                     "SPU LayerNorm core + folded affine + device layout + "
                     "A8xB8 project Conv"),
        "affine_mode": args.affine_mode,
        "stem_input": args.stem_input,
        "maximum_affine_fold_algebra_abs_error": maximum_algebra_error,
        "logical_stems": 4,
        "physical_kernels": len(records),
        "kernels": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(path), "physical_kernels": len(records),
        "maximum_affine_fold_algebra_abs_error": maximum_algebra_error,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
