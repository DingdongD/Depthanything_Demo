#!/usr/bin/env python3
"""Export all layer/head A8 attention kernels using the safe two-chunk form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

from export_u250_attention_head_multioutput import make_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--attention-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    projections = json.loads(args.projection_report.read_text())["blocks"]
    attention = json.loads(args.attention_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = []
    for layer_record in attention["layers"]:
        layer = int(layer_record["layer"])
        p = projections[str(layer)]
        # The source manifest stores six query chunks for every head.  Their
        # scales are identical within a head, so select the first chunk only.
        triplets = layer_record["triplets"]
        for head in range(6):
            record = triplets[head * 6]
            v_input_scale = float(p["v_projection_scale"])
            # Older manifests contain an accumulator-scale field produced by
            # the original floating-model calibration, but the r43 U250 path
            # intentionally ignored it.  Treat the field as active only when
            # the manifest explicitly opts into compensation with a gain.
            compensation_enabled = "av_output_gain" in record
            av_output_gain = float(record.get("av_output_gain", 1.0))
            av_v_scale = (float(record.get(
                "av_value_accumulator_scale", v_input_scale * av_output_gain))
                if compensation_enabled else v_input_scale)
            name = f"attention2_l{layer:02d}_h{head:02d}"
            path = args.output_dir / f"{name}.onnx"
            onnx.save(make_model(
                float(p["q_projection_scale"]),
                float(p["k_projection_scale"]),
                float(record["probability_scale"]),
                v_input_scale,
                2,
                av_v_scale=av_v_scale,
            ), path)
            kernels.append({
                "layer": layer, "head": head, "name": name,
                "onnx": path.name, "calls_per_head": 3,
                "query_groups": [[0, 1], [2, 3], [4, 5]],
                "last_query_valid_rows": 90,
                "scales_bf16": {
                    "q": p["q_projection_scale"],
                    "k": p["k_projection_scale"],
                    "probability": record["probability_scale"],
                    "v": v_input_scale,
                    "av_v": av_v_scale,
                    "av_output_gain": av_output_gain,
                },
            })
    manifest = {
        "schema_version": 1,
        "strategy": "two attention chains per launch; one resident kernel invoked 3x/head",
        "kernels": kernels,
        "kernels_total": len(kernels),
        "calls_per_head": 3,
        "calls_per_model": len(kernels) * 3,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
