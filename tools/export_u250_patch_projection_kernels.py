#!/usr/bin/env python3
"""Export safe <=64-channel U250 kernels for the ViT /14 patch projection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def attr(node: onnx.NodeProto, name: str):
    for value in node.attribute:
        if value.name == name:
            return helper.get_attribute_value(value)
    raise KeyError(f"{node.name}: {name}")


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def remove_attr(node: onnx.NodeProto, name: str) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True,
                        help="calibrated patchify + 1x1 Conv ONNX")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels-per-kernel", type=int, default=64)
    parser.add_argument("--patch-grid", type=int, default=37,
                        help="square ViT patch grid (image_size / 14)")
    parser.add_argument("--input-scale", type=float,
                        help="positive A8 input quantization step; omit for BF16 input")
    args = parser.parse_args()
    if args.channels_per_kernel <= 0 or args.channels_per_kernel > 64:
        raise ValueError("channels per kernel must be in [1, 64]")
    if args.input_scale is not None and args.input_scale <= 0:
        raise ValueError("input scale must be positive")
    if args.patch_grid <= 0:
        raise ValueError("patch grid must be positive")

    source = onnx.load(str(args.model.resolve()), load_external_data=True)
    initializers = {value.name: value for value in source.graph.initializer}
    convs = [
        node for node in source.graph.node
        if node.op_type == "Conv" and node.input[1] in initializers
        and numpy_helper.to_array(initializers[node.input[1]]).shape
        in ((384, 588, 1, 1), (384, 3, 14, 14))
    ]
    if len(convs) != 1:
        raise ValueError(f"expected one ViT patch Conv, got {len(convs)}")
    conv = convs[0]
    weight = numpy_helper.to_array(initializers[conv.input[1]])
    bias = numpy_helper.to_array(initializers[conv.input[2]])
    if weight.shape == (384, 3, 14, 14):
        # Match ONNX SpaceToDepth channel ordering (dy, dx, input channel).
        weight = weight.transpose(0, 2, 3, 1).reshape(384, 588, 1, 1)
    if weight.shape != (384, 588, 1, 1) or bias.shape != (384,):
        raise ValueError(f"unexpected patch parameters: {weight.shape}, {bias.shape}")
    try:
        scales = list(attr(conv, "weight_ch_scales"))
    except KeyError:
        scales = (np.max(np.abs(weight).reshape(384, -1), axis=1) / 127.0)
        scales = np.where(scales > 0.0, scales, 1.0 / 127.0).tolist()
    if len(scales) != 384:
        raise ValueError(f"expected 384 weight scales, got {len(scales)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, 384, args.channels_per_kernel):
        end = min(start + args.channels_per_kernel, 384)
        name = f"patch_projection_co{start:03d}_{end:03d}"
        node = copy.deepcopy(conv)
        node.name = name
        node.input[0] = "input0"
        node.output[0] = "output0"
        for attr_name, attr_value in (
            ("kernel_shape", [1, 1]), ("strides", [1, 1]),
            ("dilations", [1, 1]), ("pads", [0, 0, 0, 0]),
        ):
            replace_attr(node, attr_name, attr_value)
        weight_name = name + "_weight"
        bias_name = name + "_bias"
        node.input[1] = weight_name
        node.input[2] = bias_name
        replace_attr(node, "weight_ch_scales", scales[start:end])
        replace_attr(node, "weight_bitdepth", 8)
        replace_attr(node, "output_bitdepth", 16)
        replace_attr(node, "output_scale", -1.0)
        if args.input_scale is not None:
            replace_attr(node, "input_bitdepth", 8)
            remove_attr(node, "input_scale")
            replace_attr(node, "input_scales", [args.input_scale])
        graph = helper.make_graph(
            [node], name,
            [helper.make_tensor_value_info(
                "input0", TensorProto.FLOAT,
                [1, 588, args.patch_grid, args.patch_grid])],
            [helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT,
                [1, end - start, args.patch_grid, args.patch_grid])],
            [numpy_helper.from_array(np.ascontiguousarray(weight[start:end]), weight_name),
             numpy_helper.from_array(np.ascontiguousarray(bias[start:end]), bias_name)],
        )
        model = helper.make_model(
            graph, opset_imports=copy.deepcopy(source.opset_import),
            producer_name="depthanything-u250-patch-projection",
        )
        model.ir_version = source.ir_version
        path = args.output_dir / f"{name}.onnx"
        onnx.save_model(
            model, str(path), save_as_external_data=True,
            all_tensors_to_one_file=True, location=path.name + ".data",
            size_threshold=1024, convert_attribute=False,
        )
        records.append({"name": name, "channel_start": start,
                        "channel_end": end,
                        "input_shape": [1, 588, args.patch_grid, args.patch_grid],
                        "output_shape": [1, end - start, args.patch_grid,
                                         args.patch_grid]})
    input_precision = "INT8" if args.input_scale is not None else "BF16"
    manifest = {
        "schema_version": 1,
        "implementation": f"host exact /14 patchify + {input_precision}xINT8 NPU 1x1 Conv",
        "input_precision": input_precision,
        "input_scale": args.input_scale,
        "patch_grid": args.patch_grid,
        "kernels": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "kernels": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
