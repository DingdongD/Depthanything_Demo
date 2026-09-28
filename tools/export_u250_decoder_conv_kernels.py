#!/usr/bin/env python3
"""Export the 32 calibrated decoder convolutions as resident A8xB8 kernels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


LEGACY_INPUT_SHAPES = [
    [1, 384, 37, 37], [1, 48, 37, 37],
    [1, 384, 37, 37], [1, 96, 37, 37],
    [1, 384, 37, 37], [1, 384, 37, 37], [1, 384, 37, 37],
    [1, 48, 148, 148], [1, 96, 74, 74], [1, 192, 37, 37],
    [1, 384, 19, 19],
    [1, 64, 19, 19], [1, 64, 19, 19], [1, 64, 37, 37],
    [1, 64, 37, 37], [1, 64, 37, 37], [1, 64, 37, 37],
    [1, 64, 37, 37], [1, 64, 74, 74],
    [1, 64, 74, 74], [1, 64, 74, 74], [1, 64, 74, 74],
    [1, 64, 74, 74], [1, 64, 148, 148],
    [1, 64, 148, 148], [1, 64, 148, 148], [1, 64, 148, 148],
    [1, 64, 148, 148], [1, 64, 296, 296], [1, 64, 296, 296],
    [1, 32, 518, 518], [1, 32, 518, 518],
]


def get_attr(node: onnx.NodeProto, name: str, default: object = None) -> object:
    for value in node.attribute:
        if value.name == name:
            return helper.get_attribute_value(value)
    return default


def output_shape(node: onnx.NodeProto, weight: onnx.TensorProto,
                 input_shape: list[int]) -> list[int]:
    w = numpy_helper.to_array(weight)
    pads = list(get_attr(node, "pads", [0, 0, 0, 0]))
    strides = list(get_attr(node, "strides", [1, 1]))
    dilations = list(get_attr(node, "dilations", [1, 1]))
    spatial = []
    for axis in range(2):
        effective = dilations[axis] * (w.shape[axis + 2] - 1) + 1
        spatial.append((input_shape[axis + 2] + pads[axis] + pads[axis + 2]
                        - effective) // strides[axis] + 1)
    return [input_shape[0], int(w.shape[0]), *spatial]


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def save_kernel(source: onnx.ModelProto, node: onnx.NodeProto,
                initializers: dict[str, onnx.TensorProto], name: str,
                input_shape: list[int], path: Path) -> list[int]:
    required = [initializers[value] for value in node.input[1:] if value]
    out_shape = output_shape(node, initializers[node.input[1]], input_shape)
    graph = helper.make_graph(
        [node], "depthanything_" + name,
        [helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, out_shape)],
        [copy.deepcopy(value) for value in required],
    )
    model = helper.make_model(
        graph,
        opset_imports=copy.deepcopy(source.opset_import),
        producer_name=source.producer_name,
        producer_version=source.producer_version,
    )
    model.ir_version = source.ir_version
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)
    return out_shape


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scale-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = onnx.load(str(args.model), load_external_data=True)
    nodes = {node.name: node for node in source.graph.node}
    initializers = {value.name: value for value in source.graph.initializer}
    profile = json.loads(args.scale_profile.read_text())
    records = [item for item in profile["operators"]
               if item["kind"] == "decoder_conv"]
    if len(records) != len(LEGACY_INPUT_SHAPES):
        raise ValueError(f"expected 32 decoder convolutions, got {len(records)}")
    tensors = profile["tensors"]
    input_shapes = []
    for record in records:
        shape = tensors[record["activation"]]["shape"]
        if len(shape) != 4 or any(int(value) <= 0 for value in shape):
            raise ValueError(f"{record['node']}: invalid captured input shape {shape}")
        input_shapes.append([int(value) for value in shape])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (record, shape) in enumerate(zip(records, input_shapes)):
        node = copy.deepcopy(nodes[record["node"]])
        node.input[0] = "input0"
        replace_attr(node, "input_scales", [float(record["a8_scale"])])
        name = f"decoder_conv_{index:02d}"
        path = args.output_dir / (name + ".onnx")
        out_shape = save_kernel(source, node, initializers, name, shape, path)
        entry = {
            "index": index,
            "name": name,
            "onnx": path.name,
            "source_node": record["node"],
            "input_scale": float(record["a8_scale"]),
            "input_shape": shape,
            "output_shape": out_shape,
        }
        if out_shape[1] > 64:
            weight_name = node.input[1]
            weight_array = numpy_helper.to_array(initializers[weight_name])
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
            bias_array = (numpy_helper.to_array(initializers[bias_name])
                          if bias_name is not None else None)
            channel_scales = list(get_attr(node, "weight_ch_scales"))
            channel_slices = []
            # The current U250 Conv datapath corrupts lanes 64+ when a kernel
            # exposes more than 64 output channels.  This limit is specific
            # to Conv; the MatMul path safely uses 256-channel slices.
            for start in range(0, out_shape[1], 64):
                end = min(start + 64, out_shape[1])
                slice_node = copy.deepcopy(node)
                replace_attr(slice_node, "weight_ch_scales", channel_scales[start:end])
                sliced_initializers = dict(initializers)
                sliced_initializers[weight_name] = numpy_helper.from_array(
                    weight_array[start:end], name=weight_name
                )
                if bias_name is not None:
                    sliced_initializers[bias_name] = numpy_helper.from_array(
                        bias_array[start:end], name=bias_name
                    )
                slice_name = f"{name}_co{start:03d}_{end:03d}"
                slice_path = args.output_dir / (slice_name + ".onnx")
                slice_output = save_kernel(
                    source, slice_node, sliced_initializers, slice_name, shape, slice_path
                )
                channel_slices.append({
                    "name": slice_name, "onnx": slice_path.name,
                    "channel_start": start, "channel_end": end,
                    "input_shape": shape, "output_shape": slice_output,
                })
            entry["safe_channel_slices"] = channel_slices
        if shape[2] >= 148:
            weight = numpy_helper.to_array(initializers[node.input[1]])
            kernel_h = int(weight.shape[2])
            # A 64x74x296 BF16 output plus layout buffers still exceeds one
            # 4 MiB FM bank.  That sole 1x1 case uses 37-row tiles; all other
            # high-resolution convolutions fit safely at 74 output rows.
            if shape == LEGACY_INPUT_SHAPES[index]:
                tile_rows = 37 if index == 28 else 74
            else:
                # Select the largest exact divisor that keeps a tile within
                # the board-qualified <=74-row envelope.  This supports
                # 20/40/80/160/280 spatial pyramids without padding seams.
                tile_rows = max(
                    value for value in range(1, min(74, shape[2]) + 1)
                    if shape[2] % value == 0
                )
            if shape[2] % tile_rows:
                raise ValueError(
                    f"{name}: height is not divisible by {tile_rows}"
                )
            variants = []
            if kernel_h == 1:
                specs = [(f"tile{tile_rows}", tile_rows, None)]
            elif kernel_h == 3 and list(get_attr(node, "strides", [1, 1])) == [1, 1]:
                # Horizontal padding is unchanged.  Vertical halo rows are
                # supplied by the runtime, so only the image boundary tiles
                # retain one-sided vertical padding.
                specs = [
                    ("tile_first", tile_rows + 1, [1, 1, 0, 1]),
                    ("tile_middle", tile_rows + 2, [0, 1, 0, 1]),
                    ("tile_last", tile_rows + 1, [0, 1, 1, 1]),
                ]
            else:
                raise ValueError(f"{name}: unsupported tiled kernel {weight.shape}")
            for suffix, height, pads in specs:
                tile_node = copy.deepcopy(node)
                if pads is not None:
                    replace_attr(tile_node, "pads", pads)
                tile_name = name + "_" + suffix
                tile_shape = [shape[0], shape[1], height, shape[3]]
                tile_path = args.output_dir / (tile_name + ".onnx")
                tile_output = save_kernel(
                    source, tile_node, initializers, tile_name, tile_shape, tile_path
                )
                if tile_output[2] != tile_rows:
                    raise ValueError(
                        f"{tile_name}: expected {tile_rows} output rows, got {tile_output}"
                    )
                variants.append({
                    "name": tile_name,
                    "onnx": tile_path.name,
                    "position": suffix.removeprefix("tile_"),
                    "input_shape": tile_shape,
                    "output_shape": tile_output,
                })
            entry["tile_output_rows"] = tile_rows
            entry["row_tiles"] = shape[2] // tile_rows
            entry["safe_tile_variants"] = variants
        manifest.append(entry)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "strategy": "host resize/relu/add; resident A8xB8 convolution kernels",
        "kernels": manifest,
        "kernels_total": len(manifest),
    }, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
