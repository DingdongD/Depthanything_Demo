#!/usr/bin/env python3
"""Materialize board-aware attention scales in ONNX kernels and a contract."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import helper


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-model-dir", type=Path, required=True)
    parser.add_argument("--output-contract", type=Path, required=True)
    parser.add_argument(
        "--validation-gate", action="store_true",
        help="keep the old head scales when held-out relative L2 does not improve",
    )
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    manifest = json.loads(args.base_manifest.read_text())
    contract = json.loads(args.contract.read_text())
    manifest_records = {
        (int(item["layer"]), int(item["head"])): item
        for item in manifest["kernels"]
    }
    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    output_records = []

    for layer_text, layer_record in calibration["layers"].items():
        layer = int(layer_text)
        for head_record in layer_record["heads"]:
            head = int(head_record["head"])
            selected = head_record["selected_scales_bf16"]
            accepted = (not args.validation_gate or
                        head_record["selected_validation"]["relative_l2"]
                        < head_record["old_validation"]["relative_l2"])
            scales = selected if accepted else head_record["old_scales_bf16"]
            q, k, v = (float(scales[name]) for name in ("q", "k", "v"))
            probability = float(scales["probability"])
            gain = float(scales.get("av_output_gain", 1.0))
            av_v = float(scales.get("av_v", v * gain))
            source_record = manifest_records[(layer, head)]
            name = source_record["name"]
            model = onnx.load(str(args.base_model_dir / source_record["onnx"]))
            for node in model.graph.node:
                if node.op_type == "MatMul" and node.name.endswith("/QK"):
                    replace_attribute(node, "A_scales", [q])
                    replace_attribute(node, "B_scales", [k])
                elif node.op_type == "Softmax":
                    replace_attribute(node, "output_scales", [probability])
                elif node.op_type == "MatMul" and node.name.endswith("/AV"):
                    replace_attribute(node, "A_scales", [probability])
                    replace_attribute(node, "B_scales", [av_v])
            path = args.output_model_dir / source_record["onnx"]
            onnx.save(model, path)
            output_record = copy.deepcopy(source_record)
            output_record["scales_bf16"] = {
                "q": q, "k": k, "v": v, "probability": probability,
                "av_output_gain": gain, "av_v": av_v,
            }
            output_record["validation_gate_accepted"] = accepted
            output_records.append(output_record)
            contract["encoder"][layer]["attention"]["heads"][head][
                "scales_bf16"
            ] = copy.deepcopy(output_record["scales_bf16"])

    output_manifest = {
        "schema_version": 1,
        "strategy": "board-state FP32-target Softmax/AV compensation",
        "source_calibration": str(args.calibration.resolve()),
        "kernels": output_records,
        "kernels_total": len(output_records),
    }
    (args.output_model_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    args.output_contract.parent.mkdir(parents=True, exist_ok=True)
    args.output_contract.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "accepted": sum(bool(item["validation_gate_accepted"])
                        for item in output_records),
        "kernels": len(output_records),
        "manifest": str(args.output_model_dir / "manifest.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
