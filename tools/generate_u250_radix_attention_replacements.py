#!/usr/bin/env python3
"""Generate capture-and-replace traces for an additive radix-A8 attention.

This is a board downstream-impact experiment, not a claim that the current
bitstream can produce the residual digits.  Q/K/V are taken from an actual U250
trace, V remains unchanged, and every later route encodes only the exact
non-negative residual left by the preceding route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_u250_attention_operator_error import bf16, paired_paths, quantize, softmax
from evaluate_u250_multi_a8_probability import dynamic_row_max_codes, radix_codes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--representation", choices=("radix", "dynamic-row-max"),
                        default="radix")
    parser.add_argument("--routes", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    layer_record = contract["encoder"][args.layer]
    heads = layer_record["attention"]["heads"]
    host_heads = set(layer_record.get("host_attention_heads", []))
    pairs = paired_paths(args.trace_dir, args.reference_dir)
    outputs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for trace_path, _ in pairs:
        with np.load(trace_path) as trace:
            q_all = trace[f"q_l{args.layer:02d}"][0, 0]
            k_all = trace[f"k_l{args.layer:02d}"][0, 0]
            v_all = trace[f"v_l{args.layer:02d}"][0, 0]
            attention = np.ascontiguousarray(
                trace[f"attention_l{args.layer:02d}"], dtype=np.float32
            ).copy()
        for head, specification in enumerate(heads):
            if head in host_heads:
                continue
            scales = specification["scales_bf16"]
            if float(scales.get("av_output_gain", 1.0)) != 1.0:
                raise ValueError(f"layer {args.layer} head {head}: forbidden AV gain")
            if float(scales.get("av_v", scales["v"])) != float(scales["v"]):
                raise ValueError(f"layer {args.layer} head {head}: AV scale differs from V")
            begin, end = head * 64, (head + 1) * 64
            qi = quantize(q_all[:, begin:end], float(scales["q"])).astype(np.int32)
            ki = quantize(k_all[:, begin:end], float(scales["k"])).astype(np.int32)
            vi = quantize(v_all[:, begin:end], float(scales["v"])).astype(np.int32)
            logits = bf16(
                (qi @ ki.T).astype(np.float32)
                * np.float32(float(scales["q"]) * float(scales["k"]))
            )
            probability = bf16(softmax(logits))
            if args.representation == "dynamic-row-max":
                code, row_scale = dynamic_row_max_codes(probability)
                attention[0, 0, :, begin:end] = bf16(
                    (code @ vi).astype(np.float32)
                    * row_scale
                    * np.float32(float(scales["v"]))
                )
            else:
                route_codes, route_scales = radix_codes(probability, args.routes)
                partials = [
                    bf16(
                        (code @ vi).astype(np.float32)
                        * np.float32(scale * float(scales["v"]))
                    )
                    for code, scale in zip(route_codes, route_scales)
                ]
                attention[0, 0, :, begin:end] = bf16(sum(partials))
        relative = trace_path.relative_to(args.trace_dir)
        output = args.output_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            **{f"encoder_l{args.layer:02d}_attention": attention},
        )
        outputs.append({
            "trace": str(relative),
            "replacement": str(output.relative_to(args.output_dir)),
            "sha256": sha256(output),
        })
        print(output)

    manifest = {
        "schema_version": 1,
        "layer": args.layer,
        "routes": (args.routes if args.representation == "radix" else 1),
        "representation": (
            "base-127 exact non-negative residual digits"
            if args.representation == "radix"
            else "SPU-compatible BF16 per-row max/127 nearest dynamic A8"
        ),
        "av_gain_policy": "unit-required",
        "av_v_policy": "must-equal-v",
        "host_heads_preserved_from_board_trace": sorted(host_heads),
        "hardware_status": (
            "capture-and-replace downstream-impact experiment; current bitstream "
            "cannot form residual digits"
            if args.representation == "radix"
            else "SPU dynamic A8 instruction path already qualified on U250"
        ),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
