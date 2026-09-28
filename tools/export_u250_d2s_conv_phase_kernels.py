#!/usr/bin/env python3
"""Lower decoder DepthToSpace->Conv pairs to low-resolution phase Conv kernels.

The U250 bitstream does not reliably execute the standalone DepthToSpace path.
For each output pixel phase this exporter rewrites the following 3x3 Conv as a
sparse 3x3 Conv over the tensor before DepthToSpace.  Interleaving the r**2
low-resolution outputs is algebraically identical to DepthToSpace followed by
the original Conv, including zero-padding at image boundaries.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DEFAULT_PAIRS = {7: 4, 8: 2}


def get_attr(node: onnx.NodeProto, name: str, default: object = None) -> object:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def input_channel(c: int, phase_y: int, phase_x: int, factor: int) -> int:
    if factor == 2:
        return c * 4 + phase_y * 2 + phase_x
    if factor == 4:
        # The model spells x4 as two consecutive CRD DepthToSpace(x2) nodes.
        # The outer phase is consumed by the second node, while the inner
        # phase determines the high-resolution coordinate's upper bit.
        inner_y, outer_y = divmod(phase_y, 2)
        inner_x, outer_x = divmod(phase_x, 2)
        return (c * 16 + (outer_y * 2 + outer_x) * 4
                + inner_y * 2 + inner_x)
    raise ValueError(f"unsupported factor {factor}; expected 2 or 4")


def phase_weight(weight: np.ndarray, factor: int,
                 output_phase_y: int, output_phase_x: int) -> np.ndarray:
    if weight.ndim != 4 or weight.shape[2:] != (3, 3):
        raise ValueError(f"expected OIHW 3x3 weight, got {weight.shape}")
    output_channels, input_channels = weight.shape[:2]
    fused = np.zeros(
        (output_channels, input_channels * factor * factor, 3, 3),
        dtype=weight.dtype,
    )
    for kernel_y in range(3):
        for kernel_x in range(3):
            source_y = output_phase_y + kernel_y - 1
            source_x = output_phase_x + kernel_x - 1
            low_y, phase_y = divmod(source_y, factor)
            low_x, phase_x = divmod(source_x, factor)
            for channel in range(input_channels):
                fused[
                    :, input_channel(channel, phase_y, phase_x, factor),
                    low_y + 1, low_x + 1,
                ] = weight[:, channel, kernel_y, kernel_x]
    return fused


def validate_pair(weight: np.ndarray, factor: int, seed: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(
        (1, weight.shape[1] * factor * factor, 7, 6), generator=generator
    )
    shuffled = source
    if factor == 4:
        shuffled = functional.pixel_shuffle(shuffled, 2)
    shuffled = functional.pixel_shuffle(shuffled, 2)
    reference = functional.conv2d(
        shuffled, torch.from_numpy(weight.copy()), padding=1
    )
    lowered = torch.empty_like(reference)
    for phase_y in range(factor):
        for phase_x in range(factor):
            fused = torch.from_numpy(
                phase_weight(weight, factor, phase_y, phase_x)
            )
            lowered[:, :, phase_y::factor, phase_x::factor] = functional.conv2d(
                source, fused, padding=1
            )
    error = lowered - reference
    return {
        "max_abs": float(error.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(reference).clamp_min(1e-30)
        ),
    }


def save_model(source: onnx.ModelProto, node: onnx.NodeProto,
               weight: np.ndarray, weight_name: str, name: str,
               input_shape: list[int], output_shape: list[int], path: Path) -> None:
    phase_node = copy.deepcopy(node)
    phase_node.name = name
    del phase_node.input[:]
    phase_node.input.extend(["input0", weight_name])
    del phase_node.output[:]
    phase_node.output.extend(["output0"])
    replace_attr(phase_node, "pads", [1, 1, 1, 1])
    replace_attr(phase_node, "strides", [1, 1])
    replace_attr(phase_node, "dilations", [1, 1])
    replace_attr(phase_node, "kernel_shape", [3, 3])
    graph = helper.make_graph(
        [phase_node], "depthanything_" + name,
        [helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(weight, name=weight_name)],
    )
    model = helper.make_model(
        graph, opset_imports=copy.deepcopy(source.opset_import),
        producer_name=source.producer_name, producer_version=source.producer_version,
    )
    model.ir_version = source.ir_version
    onnx.save_model(
        model, str(path), save_as_external_data=True,
        all_tensors_to_one_file=True, location=path.name + ".data",
        size_threshold=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=[7, 8])
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    source_manifest = json.loads(args.manifest.read_text())
    records_by_index = {
        int(record["index"]): record for record in source_manifest["kernels"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    validations = {}
    for index in args.indices:
        factor = DEFAULT_PAIRS[index]
        record = records_by_index[index]
        model_path = args.model_dir / record["onnx"]
        source = onnx.load(str(model_path), load_external_data=True)
        if len(source.graph.node) != 1:
            raise ValueError(f"{model_path}: expected exactly one Conv")
        node = source.graph.node[0]
        if node.op_type != "Conv" or list(get_attr(node, "pads")) != [1, 1, 1, 1]:
            raise ValueError(f"{model_path}: expected same-padded Conv")
        initializers = {item.name: item for item in source.graph.initializer}
        weight_name = node.input[1]
        weight = numpy_helper.to_array(initializers[weight_name])
        if len(node.input) > 2 and node.input[2]:
            raise ValueError(f"{model_path}: biased Conv is not supported yet")
        if not args.skip_validation:
            validations[str(index)] = validate_pair(weight, factor, 1000 + index)

        high_shape = list(record["input_shape"])
        if high_shape[2] % factor or high_shape[3] % factor:
            raise ValueError(f"{model_path}: spatial shape is not divisible by {factor}")
        low_shape = [
            high_shape[0], high_shape[1] * factor * factor,
            high_shape[2] // factor, high_shape[3] // factor,
        ]
        phase_shape = [high_shape[0], int(weight.shape[0]), *low_shape[2:]]
        phases = []
        for phase_y in range(factor):
            for phase_x in range(factor):
                name = f"decoder_d2sconv_{index:02d}_py{phase_y}_px{phase_x}"
                path = args.output_dir / f"{name}.onnx"
                save_model(
                    source, node,
                    phase_weight(weight, factor, phase_y, phase_x),
                    weight_name, name, low_shape, phase_shape, path,
                )
                phases.append({
                    "name": name, "onnx": path.name,
                    "phase_y": phase_y, "phase_x": phase_x,
                    "input_shape": low_shape, "output_shape": phase_shape,
                })
        records.append({
            "index": index, "source_node": record["source_node"],
            "factor": factor, "input_scale": record["input_scale"],
            "input_shape": low_shape, "phase_output_shape": phase_shape,
            "interleaved_output_shape": record["output_shape"], "phases": phases,
        })

    manifest = {
        "schema_version": 1,
        "strategy": "exact low-resolution polyphase lowering of DepthToSpace->Conv",
        "source_manifest": str(args.manifest.resolve()),
        "kernels_total": sum(len(record["phases"]) for record in records),
        "validations": validations,
        "kernels": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "kernels": manifest["kernels_total"], "validations": validations,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
