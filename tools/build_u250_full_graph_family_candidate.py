#!/usr/bin/env python3
"""Create an executable family ablation from one joint calibration proposal."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re


FAMILY_KERNELS = {
    "attention": {"attention6"},
    "encoder_activation": {"qkv", "post_attention", "mlp_fc1", "mlp_fc2"},
    "decoder": {"decoder"},
}


def parse_indices(value: str | None, upper: int) -> set[int] | None:
    if value is None:
        return None
    result = set()
    for part in value.split(","):
        fields = part.split("-", 1)
        begin, end = (int(fields[0]), int(fields[-1]))
        if begin > end:
            raise ValueError(f"descending index range: {part}")
        result.update(range(begin, end + 1))
    if not result or min(result) < 0 or max(result) >= upper:
        raise ValueError(f"indices must be within [0, {upper - 1}]")
    return result


def encoder_kernel_layer(name: str) -> int:
    match = re.search(r"_l(\d{2})(?:_|$)", name)
    if match is None:
        raise ValueError(f"cannot resolve encoder layer from kernel {name}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--proposed-contract", type=Path, required=True)
    parser.add_argument("--base-host-plan", type=Path, required=True)
    parser.add_argument("--proposed-host-plan", type=Path, required=True)
    parser.add_argument("--compiled-manifest", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="+", choices=sorted(FAMILY_KERNELS), required=True
    )
    parser.add_argument("--attention-layers",
                        help="optional comma/range subset, for example 1-3,8")
    parser.add_argument("--encoder-layers",
                        help="optional comma/range subset for encoder A8")
    parser.add_argument("--decoder-indices",
                        help="optional comma/range subset for decoder A8")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = set(args.families)
    attention_layers = parse_indices(args.attention_layers, 12)
    encoder_layers = parse_indices(args.encoder_layers, 12)
    decoder_indices = parse_indices(args.decoder_indices, 32)
    if attention_layers is not None and "attention" not in selected:
        parser.error("--attention-layers requires the attention family")
    if encoder_layers is not None and "encoder_activation" not in selected:
        parser.error("--encoder-layers requires the encoder_activation family")
    if decoder_indices is not None and "decoder" not in selected:
        parser.error("--decoder-indices requires the decoder family")
    base = json.loads(args.base_contract.read_text())
    proposed = json.loads(args.proposed_contract.read_text())
    plan = json.loads(args.base_host_plan.read_text())
    proposed_plan = json.loads(args.proposed_host_plan.read_text())
    contract = copy.deepcopy(base)

    for layer, (target, source) in enumerate(zip(
        contract["encoder"], proposed["encoder"]
    )):
        if ("attention" in selected
                and (attention_layers is None or layer in attention_layers)):
            target["attention"] = copy.deepcopy(source["attention"])
        if ("encoder_activation" in selected
                and (encoder_layers is None or layer in encoder_layers)):
            target["qkv"]["input_quantization"] = copy.deepcopy(
                source["qkv"]["input_quantization"]
            )
            target["post_attention"]["input_quantization"] = copy.deepcopy(
                source["post_attention"]["input_quantization"]
            )
            for name in (
                "fc1_input_quantization", "fc2_input_quantization"
            ):
                target["mlp"][name] = copy.deepcopy(source["mlp"][name])

    if "decoder" in selected:
        source_steps = {item["name"]: item
                        for item in proposed_plan["decoder_steps"]}
        source_contract = {item.get("source_node"): item
                           for item in proposed["decoder"]
                           if item.get("backend") == "npu"}
        target_contract = {item.get("source_node"): item
                           for item in contract["decoder"]
                           if item.get("backend") == "npu"}
        for step in plan["decoder_steps"]:
            if step.get("backend") != "npu":
                continue
            if (decoder_indices is not None
                    and int(step["index"]) not in decoder_indices):
                continue
            source_step = source_steps[step["name"]]
            step["input_scale"] = float(source_step["input_scale"])
            target_contract[step["name"]]["input_quantization"] = copy.deepcopy(
                source_contract[step["name"]]["input_quantization"]
            )

    calibration = contract.setdefault("calibration", {})
    calibration.update(copy.deepcopy(proposed.get("calibration", {})))
    calibration["deployment_status"] = "family_ablation_requires_gate"
    selection = {
        "families": sorted(selected),
        "attention_layers": sorted(attention_layers) if attention_layers is not None else None,
        "encoder_layers": sorted(encoder_layers) if encoder_layers is not None else None,
        "decoder_indices": sorted(decoder_indices) if decoder_indices is not None else None,
    }
    calibration["active_families"] = selection
    compiled = json.loads(args.compiled_manifest.read_text())
    kernel_families = set().union(*(FAMILY_KERNELS[name] for name in selected))

    def selected_kernel(item: dict) -> bool:
        if item["family"] not in kernel_families:
            return False
        if item["family"] == "attention6":
            return (attention_layers is None
                    or encoder_kernel_layer(item["kernel"]) in attention_layers)
        if item["family"] in FAMILY_KERNELS["encoder_activation"]:
            return (encoder_layers is None
                    or encoder_kernel_layer(item["kernel"]) in encoder_layers)
        if item["family"] == "decoder":
            return (decoder_indices is None
                    or int(item["decoder_index"]) in decoder_indices)
        raise AssertionError(item["family"])

    kernels = [item for item in compiled["kernels"] if selected_kernel(item)]
    compiled["kernels"] = kernels
    compiled["kernel_count"] = len(kernels)
    compiled["family_ablation"] = selection

    contract_path = args.output_dir / "proposed_runtime_contract.json"
    plan_path = args.output_dir / "proposed_host_plan.json"
    manifest_path = args.output_dir / "compiled_manifest.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selection": selection, "kernels": len(kernels),
        "contract": str(contract_path.resolve()),
        "host_plan": str(plan_path.resolve()),
        "compiled_manifest": str(manifest_path.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
