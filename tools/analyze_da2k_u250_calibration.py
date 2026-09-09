#!/usr/bin/env python3
"""Compare a U250 runtime contract with DA-2K activation histograms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def decoder_module(source_node: str) -> str:
    path = source_node.removeprefix("/depth_head/").removesuffix("/Conv")
    path = path.replace("/", ".")
    if path.startswith("resize_layers.") and path.endswith(".conv"):
        path = path[:-5]
    if path.startswith("projects.") or path.startswith("resize_layers."):
        return "depth_head." + path
    if path.startswith("output_conv2.output_conv2."):
        path = path.replace("output_conv2.output_conv2.", "output_conv2.", 1)
    return "depth_head.scratch." + path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--histograms", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.statistics.read_bytes())
    stats = report["activations"]
    histograms = np.load(args.histograms, allow_pickle=False)
    contract = json.loads(args.contract.read_bytes())
    bins = int(report["histogram_contract"]["bins"])
    low = float(report["histogram_contract"]["minimum"])
    high = float(report["histogram_contract"]["maximum"])
    centers = np.exp2(low + (np.arange(bins) + 0.5) * (high - low) / bins)

    def evaluate(key: str, current: float, stage: str, **identity) -> dict:
        item = stats[key]
        counts = histograms[item["histogram_key"]].astype(np.float64)
        total = float(item["count"])

        def clipping(scale: float) -> float:
            return float(counts[centers > scale * 127.0].sum() / total)

        def mse(scale: float) -> float:
            quantized = np.minimum(127.0, np.rint(centers / scale)) * scale
            return float(np.dot(counts, (centers - quantized) ** 2) / total)

        cumulative = np.cumsum(counts)
        nonzero = counts.sum()
        candidates = []
        for probability in np.linspace(0.99, 0.999999, 96):
            rank = max(0.0, probability * total - float(item["zero_count"]))
            index = min(int(np.searchsorted(cumulative, rank, side="left")), bins - 1)
            candidates.append(float(centers[index] / 127.0))
        candidates.extend((float(item["abs_max"]) / 127.0, current))
        candidates = sorted({value for value in candidates if value > 0})
        optimal = min(candidates, key=mse)
        current_mse = mse(current)
        optimal_mse = mse(optimal)
        return {
            "stage": stage, "activation": key, **identity,
            "current_scale": current, "optimal_scale": optimal,
            "scale_ratio": optimal / current,
            "current_clipping_fraction": clipping(current),
            "optimal_clipping_fraction": clipping(optimal),
            "current_histogram_mse": current_mse,
            "optimal_histogram_mse": optimal_mse,
            "mse_improvement_ratio": (
                current_mse / optimal_mse if optimal_mse > 0 else math.inf
            ),
            "p999_scale": item["symmetric_int8_scales"]["p999"],
            "p9999_scale": item["symmetric_int8_scales"]["p9999"],
        }

    entries = []
    for layer, block in enumerate(contract["encoder"]):
        mappings = (
            ("encoder_qkv_input", f"pretrained.blocks.{layer}.attn.qkv.input",
             block["qkv"]["input_quantization"]["scale"]),
            ("encoder_post_attention_input", f"pretrained.blocks.{layer}.attn.proj.input",
             block["post_attention"]["input_quantization"]["scale"]),
            ("encoder_fc1_input", f"pretrained.blocks.{layer}.mlp.fc1.input",
             block["mlp"]["fc1_input_quantization"]["scale"]),
            ("encoder_fc2_input", f"pretrained.blocks.{layer}.mlp.fc2.input",
             block["mlp"]["fc2_input_quantization"]["scale"]),
        )
        for stage, key, scale in mappings:
            entries.append(evaluate(key, float(scale), stage, layer=layer))
        for head in block["attention"]["heads"]:
            for kind in ("q", "k", "v"):
                key = f"encoder.block{layer:02d}.{kind}.head{head['head']}"
                entries.append(evaluate(
                    key, float(head["scales_bf16"][kind]),
                    "attention_" + kind, layer=layer, head=int(head["head"]),
                ))

    for index, step in enumerate(contract["decoder"]):
        if step.get("backend") != "npu" or "input_quantization" not in step:
            continue
        module = decoder_module(step["source_node"])
        key = module + ".input"
        if key not in stats:
            raise KeyError(f"decoder source {step['source_node']} mapped to missing {key}")
        entries.append(evaluate(
            key, float(step["input_quantization"]["scale"]), "decoder_conv_input",
            decoder_index=index, source_node=step["source_node"], module=module,
        ))

    priorities = sorted(
        entries,
        key=lambda item: (item["mse_improvement_ratio"],
                          item["current_clipping_fraction"]),
        reverse=True,
    )
    stages = {}
    for stage in sorted({item["stage"] for item in entries}):
        selected = [item for item in entries if item["stage"] == stage]
        stages[stage] = {
            "count": len(selected),
            "mean_current_clipping_fraction": float(np.mean([
                item["current_clipping_fraction"] for item in selected
            ])),
            "median_scale_ratio": float(np.median([
                item["scale_ratio"] for item in selected
            ])),
            "max_mse_improvement_ratio": max(
                item["mse_improvement_ratio"] for item in selected
            ),
        }
    output = {
        "schema": "depthanything-da2k-u250-scale-analysis-v1",
        "calibration_samples": report["samples"],
        "method": "2048-bin log-absolute histogram symmetric-INT8 MSE search",
        "stage_summary": stages,
        "priorities": priorities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"entries": len(entries), "stages": stages,
                      "top_priorities": priorities[:12]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
