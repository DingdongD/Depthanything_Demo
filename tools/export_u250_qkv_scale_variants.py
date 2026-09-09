#!/usr/bin/env python3
"""Export single-layer QKV projection variants with alternate A8 scales."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import helper


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def tag(scale: float) -> str:
    return f"{scale:.9f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    args = parser.parse_args()

    source = onnx.load(str(args.model), load_external_data=True)
    matmuls = [node for node in source.graph.node if node.op_type == "MatMul"]
    if len(matmuls) != 3:
        raise ValueError(f"expected Q/K/V MatMul nodes, got {len(matmuls)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for scale in args.scales:
        if not scale > 0.0:
            raise ValueError(f"invalid scale: {scale}")
        model = copy.deepcopy(source)
        changed = 0
        for node in model.graph.node:
            if node.op_type == "MatMul":
                replace_attr(node, "A_scales", [scale])
                changed += 1
        name = f"qkv_projection_l03_s{tag(scale)}"
        path = args.output_dir / f"{name}.onnx"
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".data",
            size_threshold=1024,
        )
        variants.append({"name": name, "onnx": path.name, "scale": scale})
    manifest = {
        "schema_version": 1,
        "source_model": str(args.model.resolve()),
        "attribute": "A_scales",
        "matmul_count": len(matmuls),
        "variants": variants,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
