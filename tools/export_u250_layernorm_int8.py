#!/usr/bin/env python3
"""Change a compiler LayerNormalization boundary to calibrated INT8 output."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scale <= 0.0:
        parser.error("--scale must be positive")
    model = onnx.load(str(args.source), load_external_data=True)
    nodes = [node for node in model.graph.node if node.op_type == "LayerNormalization"]
    if len(nodes) != 1:
        raise ValueError("expected exactly one LayerNormalization node")
    replace_attribute(nodes[0], "output_bitdepth", 8)
    replace_attribute(nodes[0], "output_scale", args.scale)
    model.graph.name = args.output.stem
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(args.output), save_as_external_data=True,
        all_tensors_to_one_file=True, location=args.output.name + ".data",
        size_threshold=1024,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
