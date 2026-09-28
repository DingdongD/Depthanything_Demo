#!/usr/bin/env python3
"""Export one calibrated ONNX variant for every selected decoder kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import external_data_helper, helper


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    scales = {
        int(layer["index"]): float(layer["selected"]["scale"])
        for layer in calibration["layers"]
    }
    plan = json.loads(args.host_plan.read_text())
    steps = {
        int(step["index"]): step for step in plan["decoder_steps"]
        if step["backend"] == "npu"
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for index in sorted(scales):
        scale = scales[index]
        for kernel in steps[index]["kernels"]:
            name = kernel["name"]
            source = args.source_dir / f"{name}.onnx"
            if not source.is_file():
                raise FileNotFoundError(source)
            model = onnx.load(str(source), load_external_data=True)
            external_data_helper.convert_model_from_external_data(model)
            convs = [node for node in model.graph.node if node.op_type == "Conv"]
            if len(convs) != 1:
                raise ValueError(f"{source}: expected one Conv, found {len(convs)}")
            replace_attribute(convs[0], "input_scales", [scale])
            output = args.output_dir / f"{name}.onnx"
            onnx.save(model, output)
            variants.append({
                "decoder_index": index, "kernel": name,
                "input_scale": scale, "source": str(source.resolve()),
                "source_sha256": sha256(source), "onnx": output.name,
                "onnx_sha256": sha256(output),
            })
    manifest = {
        "schema": "depthanything-u250-mixed-decoder-variants-v1",
        "calibration": str(args.calibration.resolve()),
        "variants": variants,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "variants": len(variants)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
