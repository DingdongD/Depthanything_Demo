#!/usr/bin/env python3
"""Build width-padded variants of a channel-sliced decoder Conv kernel.

The U250 NORMAL16 path only returns the first complete 16-column output tile
reliably for decoder resize3 (37 -> 19).  Padding the logical input width to a
value whose convolution output is a multiple of 16 keeps the wanted 19
columns out of the physical tail tile.  The runtime crops the extra outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def attribute(node: onnx.NodeProto, name: str, default: object) -> object:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return default


def set_width(value: onnx.ValueInfoProto, width: int) -> None:
    dims = value.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError(f"expected rank-4 tensor, got {len(dims)}")
    dims[3].dim_value = width


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=int, default=6)
    parser.add_argument("--input-width", type=int, default=64)
    parser.add_argument("--crop-width", type=int, default=19)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    entry = next(item for item in manifest["kernels"]
                 if int(item["index"]) == args.index)
    variants = entry.get("safe_channel_slices")
    if not variants:
        raise ValueError(f"decoder kernel {args.index} is not channel sliced")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in variants:
        source_path = args.model_dir / item["onnx"]
        model = onnx.load(str(source_path), load_external_data=True)
        if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Conv":
            raise ValueError(f"{source_path}: expected one Conv node")
        node = model.graph.node[0]
        weight = next(value for value in model.graph.initializer
                      if value.name == node.input[1])
        kernel_width = int(numpy_helper.to_array(weight).shape[3])
        pads = list(attribute(node, "pads", [0, 0, 0, 0]))
        stride = list(attribute(node, "strides", [1, 1]))
        dilation = list(attribute(node, "dilations", [1, 1]))
        effective = dilation[1] * (kernel_width - 1) + 1
        output_width = ((args.input_width + pads[1] + pads[3] - effective)
                        // stride[1] + 1)
        if output_width < args.crop_width or output_width % 16:
            raise ValueError(
                f"padded output width {output_width} must cover crop "
                f"{args.crop_width} and be divisible by 16"
            )
        set_width(model.graph.input[0], args.input_width)
        set_width(model.graph.output[0], output_width)
        old_name = item["name"]
        new_name = old_name + f"_w{args.input_width}"
        model.graph.name = "depthanything_" + new_name
        path = args.output_dir / (new_name + ".onnx")
        onnx.save_model(
            copy.deepcopy(model), str(path), save_as_external_data=True,
            all_tensors_to_one_file=True, location=path.name + ".data",
            size_threshold=1024,
        )
        records.append({
            **item,
            "name": new_name,
            "onnx": path.name,
            "input_shape": [*item["input_shape"][:3], args.input_width],
            "output_shape": [*item["output_shape"][:3], output_width],
        })

    report = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_kernel_index": args.index,
        "input_width": args.input_width,
        "output_width": records[0]["output_shape"][3],
        "crop_width": args.crop_width,
        "kernels": records,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
