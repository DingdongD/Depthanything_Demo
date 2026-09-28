#!/usr/bin/env python3
"""Retune exported decoder Conv input scales from a board hybrid trace."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

import onnx
from onnx import helper


def replace_attr(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--board-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentile-key", default="abs_p9999")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    summary = json.loads(args.board_summary.read_text())
    stats = summary["decoder_calibration"]
    by_index = {int(item["index"]): item for item in manifest["kernels"]}
    scales: dict[int, float] = {}
    report = []
    for index, item in by_index.items():
        node = item["source_node"]
        if node not in stats:
            raise KeyError(f"missing board calibration for {node}")
        scale = float(stats[node][args.percentile_key]) / 127.0
        if not scale > 0.0:
            raise ValueError(f"invalid scale for {node}: {scale}")
        scales[index] = scale
        report.append({
            "index": index,
            "source_node": node,
            "old_scale": float(item["input_scale"]),
            "new_scale": scale,
            "ratio": scale / float(item["input_scale"]),
        })
        item["input_scale"] = scale

    model_count = 0
    for source_path in sorted(args.model_dir.glob("decoder_conv_*.onnx")):
        match = re.match(r"decoder_conv_(\d\d)(?:_|$)", source_path.stem)
        if match is None:
            raise ValueError(f"unrecognized decoder kernel name: {source_path.name}")
        index = int(match.group(1))
        model = onnx.load(str(source_path), load_external_data=False)
        convs = [node for node in model.graph.node if node.op_type == "Conv"]
        if len(convs) != 1:
            raise ValueError(f"{source_path}: expected one Conv, got {len(convs)}")
        replace_attr(convs[0], "input_scales", [scales[index]])
        output_path = args.output_dir / source_path.name
        onnx.save_model(model, str(output_path))
        data_path = Path(str(source_path) + ".data")
        if data_path.is_file():
            shutil.copy2(data_path, Path(str(output_path) + ".data"))
        model_count += 1

    manifest["source_manifest"] = str(args.manifest.resolve())
    manifest["board_summary"] = str(args.board_summary.resolve())
    manifest["scale_policy"] = f"{args.percentile_key}/127"
    manifest["models_retuned"] = model_count
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "retune_report.json").write_text(
        json.dumps({"operators": report}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"models": model_count, "operators": len(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
