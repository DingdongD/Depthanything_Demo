#!/usr/bin/env python3
"""Bind board-observed LayerNorm-core scales to folded QKV/FC1 linears."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def attribute(node: onnx.NodeProto, name: str):
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    raise KeyError(f"{node.name}: missing {name}")


def set_attributes(node: onnx.NodeProto, **values: object) -> None:
    names = set(values)
    kept = [item for item in node.attribute if item.name not in names]
    del node.attribute[:]
    node.attribute.extend(kept)
    for name, value in values.items():
        node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--board-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--statistic", default="abs_p9999")
    args = parser.parse_args()

    model = onnx.load(str(args.input.resolve()), load_external_data=True)
    summary = json.loads(args.board_summary.read_text())
    calibration = summary["hybrid_calibration"]
    initializers = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    nodes = {node.name: node for node in model.graph.node}
    records = []
    for layer in range(12):
        targets = [
            (f"/blocks.{layer}/attn/qkv/{branch}/MatMul", "norm1")
            for branch in ("q", "k", "v")
        ] + [(f"/blocks.{layer}/mlp/fc1/MatMul", "norm2")]
        for name, norm_name in targets:
            node = nodes[name]
            weight = np.asarray(initializers[node.input[1]], dtype=np.float32)
            weight_scales = np.max(np.abs(weight), axis=0) / 127.0
            weight_scales = np.where(
                weight_scales > 0.0, weight_scales, 1.0 / 127.0
            ).astype(np.float32)
            calibration_name = (
                f"/blocks.{layer}/{norm_name}/LayerNormalization_output_0"
            )
            stats = calibration[calibration_name]
            input_scale = float(stats[args.statistic]) / 127.0
            old_scale = float(attribute(node, "A_scales")[0])
            set_attributes(
                node,
                A_bitdepth=8,
                A_scales=[input_scale],
                B_bitdepth=8,
                B_scales=weight_scales.tolist(),
                B_quant_dim=[1],
            )
            records.append({
                "layer": layer, "norm": norm_name, "node": name,
                "old_input_scale": old_scale, "new_input_scale": input_scale,
                "observed_max_abs": float(stats["max_abs"]),
                "observed_statistic": float(stats[args.statistic]),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(args.output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=args.output.name + ".data",
        size_threshold=1024, convert_attribute=False,
    )
    report = {
        "schema_version": 1,
        "source_model": str(args.input.resolve()),
        "board_summary": str(args.board_summary.resolve()),
        "statistic": args.statistic,
        "updated_linears": len(records),
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "updated_linears": len(records),
        "minimum_scale": min(item["new_input_scale"] for item in records),
        "maximum_scale": max(item["new_input_scale"] for item in records),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
