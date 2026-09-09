#!/usr/bin/env python3
"""Clone one calibrated Conv kernel ONNX at several input scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def scale_tag(scale: float) -> str:
    return f"s{scale:.9f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.model.read_bytes()
    source = onnx.load_from_string(source_bytes)
    convs = [node for node in source.graph.node if node.op_type == "Conv"]
    if len(convs) != 1:
        raise ValueError(f"expected one Conv node, got {len(convs)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for scale in args.scales:
        model = onnx.load_from_string(source_bytes)
        replace_attr(model.graph.node[0], "input_scales", [float(scale)])
        tag = scale_tag(scale)
        name = f"{args.model.stem}_{tag}"
        path = args.output_dir / f"{name}.onnx"
        onnx.save(model, path)
        variants.append({
            "name": name,
            "onnx": path.name,
            "input_scale": float(scale),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    manifest = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "attribute": "input_scales",
        "variants": variants,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
