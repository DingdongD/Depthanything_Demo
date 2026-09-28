#!/usr/bin/env python3
"""Lower CRD DepthToSpace to low-resolution one-hot 1x1 Conv gathers.

Each kernel selects one spatial phase's channels.  The host only interleaves
the phase tensors; all channel selection and INT8->BF16 conversion runs on NPU.
This is much cheaper than folding the following 3x3 Conv into dense sparse
phase kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export_u250_d2s_conv_phase_kernels import input_channel


DEFAULT_PAIRS = {7: 4, 8: 2}


def bf16_scalar(value: float) -> float:
    array = np.asarray([value], np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return float((rounded & np.uint32(0xFFFF0000)).view(np.float32)[0])


def save_kernel(name: str, input_shape: list[int], output_shape: list[int],
                weight: np.ndarray, input_scale: float, path: Path) -> None:
    weight_name = name + ".weight"
    node = helper.make_node(
        "Conv", ["input0", weight_name], ["output0"], name=name,
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0],
        strides=[1, 1], input_bitdepth=8, input_scales=[input_scale],
        weight_bitdepth=8, weight_ch_scales=[1.0] * output_shape[1],
        output_bitdepth=16, output_scale=-1.0,
    )
    graph = helper.make_graph(
        [node], "depthanything_" + name,
        [helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)],
        [numpy_helper.from_array(weight, name=weight_name)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save_model(
        model, str(path), save_as_external_data=True,
        all_tensors_to_one_file=True, location=path.name + ".data",
        size_threshold=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=[7, 8])
    args = parser.parse_args()

    source = json.loads(args.phase_manifest.read_text())
    source_records = {int(record["index"]): record for record in source["kernels"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in args.indices:
        factor = DEFAULT_PAIRS[index]
        record = source_records[index]
        input_shape = list(record["input_shape"])
        output_channels = int(input_shape[1] // (factor * factor))
        interleaved_output_shape = [
            input_shape[0], output_channels,
            input_shape[2] * factor, input_shape[3] * factor,
        ]
        scale = bf16_scalar(float(record["input_scale"]))
        phases = []
        for phase_y in range(factor):
            for phase_x in range(factor):
                slices = []
                for start in range(0, output_channels, 64):
                    end = min(start + 64, output_channels)
                    weight = np.zeros(
                        (end - start, input_shape[1], 1, 1), dtype=np.float32
                    )
                    for output_channel in range(start, end):
                        weight[
                            output_channel - start,
                            input_channel(output_channel, phase_y, phase_x, factor),
                            0, 0,
                        ] = 1.0
                    name = (f"decoder_d2sgather_{index:02d}_py{phase_y}_px{phase_x}"
                            f"_co{start:03d}_{end:03d}")
                    output_shape = [input_shape[0], end - start, *input_shape[2:]]
                    save_kernel(
                        name, input_shape, output_shape, weight, scale,
                        args.output_dir / (name + ".onnx"),
                    )
                    slices.append({
                        "name": name, "onnx": name + ".onnx",
                        "channel_start": start, "channel_end": end,
                        "output_shape": output_shape,
                    })
                phases.append({
                    "phase_y": phase_y, "phase_x": phase_x, "slices": slices,
                })
        records.append({
            "index": index, "factor": factor, "input_scale": scale,
            "input_shape": input_shape,
            "interleaved_output_shape": interleaved_output_shape,
            "phases": phases,
        })
    manifest = {
        "schema_version": 1,
        "strategy": "NPU one-hot 1x1 channel gather plus host phase interleave",
        "kernels_total": sum(len(p["slices"]) for r in records for p in r["phases"]),
        "kernels": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path),
                      "kernels": manifest["kernels_total"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
