#!/usr/bin/env python3
"""Export one-input-scale variants of a single-MatMul U250 kernel model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import external_data_helper, helper


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
    source = onnx.load(str(args.model), load_external_data=True)
    external_data_helper.convert_model_from_external_data(source)
    matmuls = [node for node in source.graph.node if node.op_type == "MatMul"]
    if len(matmuls) != 1:
        raise ValueError(f"expected one MatMul node, got {len(matmuls)}")

    variants = []
    for scale in args.scales:
        if not scale > 0.0:
            raise ValueError(f"invalid scale: {scale}")
        model = copy.deepcopy(source)
        node = next(node for node in model.graph.node if node.op_type == "MatMul")
        replace_attr(node, "A_scales", [float(scale)])
        tag = scale_tag(scale)
        variant_dir = args.output_dir / tag
        variant_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the kernel name so the standard compiler layout router can
        # recognize post_attention_lXX.
        path = variant_dir / args.model.name
        onnx.save(model, path)
        variants.append({
            "name": args.model.stem,
            "tag": tag,
            "onnx": str(path.relative_to(args.output_dir)),
            "input_scale": float(scale),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "attribute": "A_scales",
        "variants": variants,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
