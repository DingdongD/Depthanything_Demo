#!/usr/bin/env python3
"""Append CRD DepthToSpace to the decoder's board-safe Conv channel slices."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def set_attributes(node: onnx.NodeProto, **values: object) -> None:
    names = set(values)
    kept = [attribute for attribute in node.attribute if attribute.name not in names]
    del node.attribute[:]
    node.attribute.extend(kept)
    for name, value in values.items():
        node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="*", default=[1, 3])
    parser.add_argument(
        "--output-scale", action="append", default=[], metavar="INDEX=SCALE",
        help="use an A8 Conv/D2S boundary at this dequantization step",
    )
    parser.add_argument(
        "--final-bf16", action="store_true",
        help="keep D2S inputs A8 but emit the last stage as BF16",
    )
    args = parser.parse_args()

    output_scales = {}
    for specification in args.output_scale:
        index_text, scale_text = specification.split("=", 1)
        output_scales[int(index_text)] = float(scale_text)

    manifest = json.loads(args.manifest.read_text())
    selected = {1: 2, 3: 1}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for record in manifest["kernels"]:
        index = int(record["index"])
        if index not in args.indices:
            continue
        if index not in selected or "safe_channel_slices" not in record:
            raise ValueError(f"decoder Conv {index} has no DepthToSpace slice policy")
        stages = selected[index]
        for item in record["safe_channel_slices"]:
            source_path = args.model_dir / item["onnx"]
            model = onnx.load(str(source_path), load_external_data=True)
            if len(model.graph.output) != 1:
                raise ValueError(f"{source_path}: expected one output")
            current = model.graph.output[0].name
            output_scale = output_scales.get(index)
            if output_scale is not None:
                set_attributes(
                    model.graph.node[0], output_bitdepth=8,
                    output_scales=[output_scale],
                )
                # A vector output scale is the authoritative A8 contract.
                kept = [attribute for attribute in model.graph.node[0].attribute
                        if attribute.name != "output_scale"]
                del model.graph.node[0].attribute[:]
                model.graph.node[0].attribute.extend(kept)
            channels = int(item["output_shape"][1])
            height = int(item["output_shape"][2])
            width = int(item["output_shape"][3])
            for stage in range(stages):
                output = "output0" if stage == stages - 1 else f"d2s_{stage}"
                final_bf16 = args.final_bf16 and stage == stages - 1
                quantization = (
                    {"input_bitdepth": 8, "input_scales": [output_scale],
                     **({"output_bitdepth": 16, "output_scale": -1.0}
                        if final_bf16 else
                        {"output_bitdepth": 8, "output_scales": [output_scale]})}
                    if output_scale is not None else
                    {"input_bitdepth": 16, "input_scale": -1.0,
                     "output_bitdepth": 16, "output_scale": -1.0}
                )
                model.graph.node.append(helper.make_node(
                    "DepthToSpace", [current], [output],
                    name=f"/DepthToSpace_{stage}", blocksize=2, mode="CRD",
                    **quantization,
                ))
                current = output
                channels //= 4
                height *= 2
                width *= 2
            del model.graph.output[:]
            model.graph.output.append(helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, channels, height, width]
            ))
            name = f"{item['name']}_d2s{stages}"
            model.graph.name = name
            path = args.output_dir / f"{name}.onnx"
            onnx.save_model(
                model, str(path), save_as_external_data=True,
                all_tensors_to_one_file=True, location=path.name + ".data",
                size_threshold=1024, convert_attribute=False,
            )
            records.append({
                "index": index, "name": name, "source_kernel": item["name"],
                "channel_start": item["channel_start"],
                "channel_end": item["channel_end"], "stages": stages,
                "input_shape": item["input_shape"],
                "output_shape": [1, channels, height, width],
                "input_scale": record["input_scale"],
                "output_scale": output_scale,
            })
    output = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "mode": "CRD",
        "strategy": "channel-aligned Conv followed by experimental DepthToSpace",
        "board_validated": False,
        "kernels": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "kernels": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
