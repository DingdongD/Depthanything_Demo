#!/usr/bin/env python3
"""Retune one U250 attention layer's per-head K scales from DA-2K stats."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


def bf16_scalar(value: float) -> float:
    data = np.asarray([value], dtype=np.float32)
    bits = data.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return float((rounded & np.uint32(0xFFFF0000)).view(np.float32)[0])


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scale-analysis", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    analysis = json.loads(args.scale_analysis.read_text())
    scales = {}
    for record in analysis["priorities"]:
        if (record.get("stage") == "attention_k"
                and int(record.get("layer", -1)) == args.layer):
            scales[int(record["head"])] = bf16_scalar(float(record["optimal_scale"]))
    if set(scales) != set(range(6)):
        raise ValueError(f"expected six K scales, got heads {sorted(scales)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for head in range(6):
        name = f"attention2_l{args.layer:02d}_h{head:02d}"
        source = args.source_dir / f"{name}.onnx"
        model = onnx.load(str(source))
        changed = 0
        for node in model.graph.node:
            if node.op_type == "MatMul" and node.name.endswith("/QK"):
                replace_attr(node, "B_scales", [scales[head]])
                changed += 1
        if changed != 2:
            raise ValueError(f"{name}: expected two QK nodes, got {changed}")
        path = args.output_dir / source.name
        onnx.save(model, str(path))
        old = contract["encoder"][args.layer]["attention"]["heads"][head][
            "scales_bf16"]["k"]
        contract["encoder"][args.layer]["attention"]["heads"][head][
            "scales_bf16"]["k"] = scales[head]
        records.append({
            "head": head,
            "layer": args.layer,
            "name": name,
            "onnx": path.name,
            "old_k_scale": old,
            "k_scale_bf16": scales[head],
        })
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "strategy": "DA-2K histogram-MSE per-head K scale, single-layer gate",
        "scale_analysis": str(args.scale_analysis.resolve()),
        "kernels": records,
    }, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "depthanything_u250_runtime_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"layer": args.layer, "scales": scales}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
