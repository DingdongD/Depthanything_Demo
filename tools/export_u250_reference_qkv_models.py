#!/usr/bin/env python3
"""Restore every encoder QKV projection to checkpoint amplitude."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper
import numpy as np
import torch

from export_u250_block_alignment_models import (
    folded_qkv,
    head_gains,
    initializer,
    relative_l2,
    replace_initializer,
)


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0-11")
    args = parser.parse_args()

    if "-" in args.layers:
        first, last = (int(item) for item in args.layers.split("-", 1))
        layers = list(range(first, last + 1))
    else:
        layers = [int(item) for item in args.layers.split(",")]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = json.loads(args.contract.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for layer in layers:
        specification = contract["encoder"][layer]
        heads = len(specification["attention"]["heads"])
        expected = folded_qkv(state, layer, heads)
        source = args.source_dir / f"qkv_projection_l{layer:02d}.onnx"
        model = onnx.load(str(source), load_external_data=True)
        prefix = f"/blocks.{layer}/attn/qkv"
        for name in ("q", "k"):
            weight, bias = expected[name]
            weight_error = relative_l2(
                initializer(model, f"{prefix}/{name}/weight"), weight
            )
            bias_error = relative_l2(
                initializer(model, f"{prefix}/{name}/bias"), bias
            )
            if weight_error > 1e-6 or bias_error > 1e-6:
                raise ValueError(
                    f"layer {layer} {name}: source is not checkpoint-aligned: "
                    f"weight={weight_error}, bias={bias_error}"
                )
        v_weight, v_bias = expected["v"]
        old_v_weight = initializer(model, f"{prefix}/v/weight")
        old_v_bias = initializer(model, f"{prefix}/v/bias")
        old_gains = head_gains(old_v_weight, v_weight, heads)
        before_weight = relative_l2(old_v_weight, v_weight)
        before_bias = relative_l2(old_v_bias, v_bias)
        replace_initializer(model, f"{prefix}/v/weight", v_weight)
        replace_initializer(model, f"{prefix}/v/bias", v_bias)

        input_scale = float(specification["qkv"]["input_quantization"]["scale"])
        changed = 0
        v_scale_changed = 0
        output_changed = 0
        for node in model.graph.node:
            if node.op_type == "MatMul":
                replace_attribute(node, "A_scales", [input_scale])
                changed += 1
                if node.name.endswith("/v/MatMul"):
                    # Per-output-channel symmetric B8 quantization.  Keeping
                    # scales from the amplified weights would preserve the
                    # old integer codes and silently reintroduce their gain.
                    weight_scales = np.max(np.abs(v_weight), axis=0) / 127.0
                    replace_attribute(
                        node, "B_scales", weight_scales.astype(np.float32).tolist()
                    )
                    v_scale_changed += 1
            elif node.op_type == "Add":
                # The resident attention bridge consumes decoded BF16 Q/K/V.
                replace_attribute(node, "output_bitdepth", 16)
                replace_attribute(node, "output_scale", -1.0)
                output_changed += 1
        if changed != 3:
            raise ValueError(f"layer {layer}: expected three QKV MatMuls, got {changed}")
        if v_scale_changed != 1 or output_changed != 3:
            raise ValueError(
                f"layer {layer}: expected one V scale and three BF16 outputs, "
                f"got {v_scale_changed} and {output_changed}"
            )
        output = args.output_dir / source.name
        onnx.save(model, output)
        records.append({
            "layer": layer,
            "name": source.stem,
            "onnx": output.name,
            "input_scale": input_scale,
            "v_weight_relative_l2_before": before_weight,
            "v_bias_relative_l2_before": before_bias,
            "v_weight_head_gains_before": old_gains,
            "v_weight_relative_l2_after": relative_l2(
                initializer(model, f"{prefix}/v/weight"), v_weight
            ),
            "v_bias_relative_l2_after": relative_l2(
                initializer(model, f"{prefix}/v/bias"), v_bias
            ),
        })

    manifest = {
        "schema_version": 1,
        "policy": "checkpoint-exact-qkv-with-no-v-amplitude-compensation",
        "source_dir": str(args.source_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "kernels": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
