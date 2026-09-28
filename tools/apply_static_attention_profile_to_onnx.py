#!/usr/bin/env python3
"""Attach calibrated static-INT8 attributes to every attention triplet.

The DS quantizer preserves the 12-layer, 6-head, 6-query-chunk topology, but
defaults the activation inputs of MatMul and Softmax to BF16.  This tool binds
the measured per-layer/per-head scales to the QK MatMul, SPU Softmax and AV
MatMul nodes so the compiler emits the U250 INT8 path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import onnx
from onnx import helper


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ds_models.static_int8_attention import StaticAttentionProfile, plan_query_chunks  # noqa: E402


ATTENTION_NAME = re.compile(r"^/blocks\.(\d+)/attn/")
QUANTIZATION_ATTRIBUTES = {
    "A_bitdepth",
    "A_scale",
    "A_scales",
    "B_bitdepth",
    "B_scale",
    "B_scales",
    "input_bitdepth",
    "input_scale",
    "input_scales",
    "output_bitdepth",
    "output_scale",
    "output_scales",
}


def replace_attributes(node: onnx.NodeProto, **values: object) -> None:
    """Replace DS quantization attributes without disturbing standard attrs."""

    retained = [a for a in node.attribute if a.name not in QUANTIZATION_ATTRIBUTES]
    del node.attribute[:]
    node.attribute.extend(retained)
    for name, value in values.items():
        node.attribute.append(helper.make_attribute(name, value))


def bind_profile(model: onnx.ModelProto, profile: StaticAttentionProfile) -> dict:
    producer = {}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for output in node.output:
            if output:
                if output in producer:
                    raise ValueError(f"multiple producers for tensor {output!r}")
                producer[output] = node
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    by_layer: dict[int, list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]]] = defaultdict(list)
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        match = ATTENTION_NAME.match(softmax.name)
        if not match:
            continue
        layer = int(match.group(1))
        if len(softmax.input) != 1 or softmax.input[0] not in producer:
            raise ValueError(f"Softmax {softmax.name!r} has no unique producer")
        qk = producer[softmax.input[0]]
        av_candidates = [n for n in consumers.get(softmax.output[0], []) if n.op_type == "MatMul"]
        if qk.op_type != "MatMul" or len(av_candidates) != 1:
            raise ValueError(
                f"expected MatMul -> {softmax.name} -> MatMul, got "
                f"{qk.op_type} and {len(av_candidates)} downstream MatMul nodes"
            )
        by_layer[layer].append((qk, softmax, av_candidates[0]))

    expected_layers = len(profile.layers)
    expected_chunks = len(plan_query_chunks(profile.tokens))
    expected_per_layer = profile.heads * expected_chunks
    if set(by_layer) != set(range(expected_layers)):
        raise ValueError(
            f"attention layers mismatch: found {sorted(by_layer)}, "
            f"expected 0..{expected_layers - 1}"
        )

    manifest_layers = []
    seen_nodes: set[str] = set()
    for layer in range(expected_layers):
        triplets = by_layer[layer]
        if len(triplets) != expected_per_layer:
            raise ValueError(
                f"layer {layer} has {len(triplets)} attention triplets, "
                f"expected {expected_per_layer}"
            )
        records = []
        for ordinal, (qk, softmax, av) in enumerate(triplets):
            head, chunk = divmod(ordinal, expected_chunks)
            scale = profile.scale(layer, head)
            for node in (qk, softmax, av):
                if node.name in seen_nodes:
                    raise ValueError(f"attention node reused by multiple triplets: {node.name}")
                seen_nodes.add(node.name)
            replace_attributes(
                qk,
                A_bitdepth=8,
                A_scales=[scale.q],
                B_bitdepth=8,
                B_scales=[scale.k],
                output_bitdepth=16,
                output_scale=-1.0,
            )
            replace_attributes(
                softmax,
                input_bitdepth=16,
                input_scale=-1.0,
                output_bitdepth=8,
                output_scales=[scale.probability],
            )
            replace_attributes(
                av,
                A_bitdepth=8,
                A_scales=[scale.probability],
                B_bitdepth=8,
                B_scales=[scale.effective_av_value_scale],
                output_bitdepth=16,
                output_scale=-1.0,
            )
            records.append({
                "head": head,
                "chunk": chunk,
                "qk": qk.name,
                "softmax": softmax.name,
                "av": av.name,
                "q_scale": scale.q,
                "k_scale": scale.k,
                "probability_scale": scale.probability,
                "av_value_accumulator_scale": scale.effective_av_value_scale,
            })
        manifest_layers.append({"layer": layer, "triplets": records})

    total = sum(len(value) for value in by_layer.values())
    expected_total = expected_layers * expected_per_layer
    if total != expected_total:
        raise ValueError(f"found {total} triplets, expected {expected_total}")
    return {
        "schema_version": 1,
        "layers": manifest_layers,
        "attention_layers": expected_layers,
        "heads_per_layer": profile.heads,
        "query_chunks_per_head": expected_chunks,
        "triplets_per_layer": expected_per_layer,
        "triplets_total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ONNX 1.17 only derives the external-data base directory for ``str``
    # paths; passing pathlib.Path makes it incorrectly use the process cwd.
    model = onnx.load(str(input_path), load_external_data=True)
    profile = StaticAttentionProfile.load(args.profile)
    manifest = bind_profile(model, profile)
    manifest.update({
        "input": str(input_path),
        "output": str(output_path),
        "profile": str(args.profile.resolve()),
    })

    external_name = output_path.with_suffix(".data").name
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=1024,
    )
    manifest_path = (args.manifest or output_path.with_suffix(".manifest.json")).resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"triplets={manifest['triplets_total']}")
    print(f"model={output_path}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
