#!/usr/bin/env python3
"""Prove the FM address plan for DepthAnything encoder tensor residency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ADDRESS_UNIT_BYTES_PER_BANK = 128
COMBINED_BYTES_PER_ADDRESS_UNIT = 2 * ADDRESS_UNIT_BYTES_PER_BANK
FM_ALIGNMENT_UNITS = 4096 // ADDRESS_UNIT_BYTES_PER_BANK


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def storage_abi(tensor: dict) -> tuple:
    """Return fields that determine valid-lane storage, independent of direction."""
    return (
        str(tensor["layout"]),
        tuple(int(value) for value in tensor["dims"]),
        int(tensor["bitdepth"]),
        int(tensor["size_per_bank"]),
    )


def tensor_units(tensor: dict) -> int:
    combined = int(tensor["size_per_bank"])
    if combined % COMBINED_BYTES_PER_ADDRESS_UNIT:
        raise ValueError("tensor physical extent is not a whole two-bank address unit")
    return combined // COMBINED_BYTES_PER_ADDRESS_UNIT


def tensor_interval(tensor: dict, base_offset_units: int = 0) -> tuple[int, int]:
    begin = base_offset_units + int(tensor["address"])
    return begin, begin + tensor_units(tensor)


def record_end(record: dict, base_offset_units: int = 0) -> int:
    return max(
        tensor_interval(tensor, base_offset_units)[1]
        for tensor in record["inputs"] + record["outputs"]
    )


def _same_abi(left: dict, right: dict) -> bool:
    return storage_abi(left) == storage_abi(right)


def _interval(name: str, begin: int, end: int) -> dict:
    return {
        "name": name,
        "begin_units": begin,
        "end_units": end,
        "bytes_per_bank": (end - begin) * ADDRESS_UNIT_BYTES_PER_BANK,
        "combined_bytes": (end - begin) * COMBINED_BYTES_PER_ADDRESS_UNIT,
    }


def analyze(manifest: dict, contract: dict) -> dict:
    records = {record["name"]: record for record in manifest["cases"]}
    blocks = contract["encoder"]
    if not blocks:
        raise ValueError("runtime contract has no encoder blocks")
    if int(manifest["shared_fm_workspace_bytes"]) % 2:
        raise ValueError("shared FM workspace must split evenly across two banks")
    workspace_units = (
        int(manifest["shared_fm_workspace_bytes"])
        // 2 // ADDRESS_UNIT_BYTES_PER_BANK
    )

    layer_checks = []
    low_end = 0
    for block in blocks:
        layer = int(block["layer"])
        norm_name = block["host_norm1"].get("npu_core")
        norm2_name = block["host_norm2"].get("npu_core")
        if not norm_name or norm_name != norm2_name:
            raise ValueError(f"layer {layer}: norm1/norm2 do not share one NPU core ABI")
        norm = records[norm_name]
        qkv = records[block["qkv"]["kernel"]]
        post = records[block["post_attention"]["kernel"]]
        attention = [records[head["kernel"]] for head in block["attention"]["heads"]]
        fc1 = [records[name] for name in block["mlp"]["fc1_kernels"]]
        fc2 = records[block["mlp"]["fc2_kernel"]]
        low_end = max(low_end, record_end(qkv), *(record_end(item) for item in attention))
        layer_checks.append({
            "layer": layer,
            "residual_x_norm1_to_post_storage_abi": _same_abi(
                norm["inputs"][0], post["inputs"][1]
            ),
            "post_to_norm2_storage_abi": _same_abi(
                post["outputs"][0], norm["inputs"][0]
            ),
            "qkv_low_end_units": record_end(qkv),
            "attention_low_end_units": max(record_end(item) for item in attention),
            "fc1_kernel_count": len(fc1),
            "fc2_output_abi_matches_next_norm_input": _same_abi(
                fc2["outputs"][0], norm["inputs"][0]
            ),
        })
    if not all(
        item["residual_x_norm1_to_post_storage_abi"]
        and item["post_to_norm2_storage_abi"]
        for item in layer_checks
    ):
        raise ValueError("encoder layers do not share the required resident storage ABI")

    first = blocks[0]
    norm = records[first["host_norm1"]["npu_core"]]
    post = records[first["post_attention"]["kernel"]]
    x_units = tensor_units(norm["inputs"][0])
    x_begin = _align(low_end, FM_ALIGNMENT_UNITS)
    x_end = x_begin + x_units
    post_base = x_begin - int(post["inputs"][1]["address"])
    post_begin, post_end = tensor_interval(post["outputs"][0], post_base)
    norm2_base = post_begin - int(norm["inputs"][0]["address"])
    norm2_begin, norm2_end = tensor_interval(norm["outputs"][0], norm2_base)
    post_input_begin, post_input_end = tensor_interval(post["inputs"][0], post_base)

    if (post_input_end != x_begin or post_begin != x_end
            or norm2_begin != post_end):
        raise ValueError("candidate tensors do not form the expected packed FM chain")
    if norm2_end > workspace_units:
        raise ValueError("candidate resident chain exceeds shared FM workspace")

    combined_tensor_bytes = int(norm["inputs"][0]["size_per_bank"])
    block_count = len(blocks)
    saved_h2c = 2 * combined_tensor_bytes * block_count
    saved_pack_logical = (
        2 * int(blocks[0]["input_shape"][1])
        * int(blocks[0]["input_shape"][2]) * 4 * block_count
    )
    boundary_matrix = [
        {
            "boundary": "residual_x(norm1 input)->post residual input",
            "storage_abi": "identical",
            "decision": "retain",
            "reason": "same BF16 NDWC valid-lane layout; intervening QKV/attention stay below x",
        },
        {
            "boundary": "norm1->QKV",
            "storage_abi": "incompatible",
            "decision": "materialize",
            "reason": "BF16 output requires host fixed-scale INT8 quantization",
        },
        {
            "boundary": "QKV->attention",
            "storage_abi": "incompatible",
            "decision": "materialize",
            "reason": "head slicing, K transpose, and query slicing change shape/order",
        },
        {
            "boundary": "attention->post",
            "storage_abi": "incompatible",
            "decision": "materialize",
            "reason": "six-head assembly and BF16-to-fixed-scale-INT8 conversion remain on host",
        },
        {
            "boundary": "post->norm2",
            "storage_abi": "identical",
            "decision": "retain",
            "reason": "same BF16 NDWC valid-lane layout; base relocation makes addresses identical",
        },
        {
            "boundary": "norm2->FC1",
            "storage_abi": "incompatible",
            "decision": "materialize",
            "reason": "BF16 output requires host fixed-scale INT8 quantization",
        },
        {
            "boundary": "FC1->FC2",
            "storage_abi": "incompatible",
            "decision": "materialize",
            "reason": "FP32 GELU, six-slice concatenate, and INT8 quantization remain on host",
        },
        {
            "boundary": "FC2->next norm1",
            "storage_abi": "identical but semantically incomplete",
            "decision": "materialize",
            "reason": "host residual add with retained post is required before next LayerNorm",
        },
    ]
    return {
        "schema_version": 1,
        "bank_sha256": manifest.get("bank_sha256"),
        "encoder_blocks": block_count,
        "workspace": {
            "combined_bytes": int(manifest["shared_fm_workspace_bytes"]),
            "bytes_per_bank": int(manifest["shared_fm_workspace_bytes"]) // 2,
            "units_per_bank": workspace_units,
            "used_units_per_bank": norm2_end,
            "headroom_units_per_bank": workspace_units - norm2_end,
        },
        "address_plan": {
            "unit_bytes_per_bank": ADDRESS_UNIT_BYTES_PER_BANK,
            "intervals": [
                _interval("default_qkv_attention_scratch", 0, low_end),
                _interval("retained_residual_x", x_begin, x_end),
                _interval("resident_post_output", post_begin, post_end),
                _interval("resident_norm2_output", norm2_begin, norm2_end),
            ],
            "kernel_base_offsets_units": {
                "norm1": x_begin,
                "qkv_attention": 0,
                "post_attention": post_base,
                "norm2": norm2_base,
            },
            "post_input0_temporal_reuse": _interval(
                "post_attention_int8_input", post_input_begin, post_input_end
            ),
        },
        "boundary_matrix": boundary_matrix,
        "layer_checks": layer_checks,
        "projected_steady_frame_savings": {
            "h2c_bytes": saved_h2c,
            "native_pack_calls": 2 * block_count,
            "native_pack_logical_bytes": saved_pack_logical,
            "native_pack_physical_bytes": saved_h2c,
            "c2h_bytes": 0,
            "python_submission_groups": block_count,
            "npu_dispatches": 0,
        },
        "padding_policy": (
            "Compatibility covers valid tensor lanes. Padding bytes may retain prior FM "
            "contents and must never be hashed or consumed as logical elements."
        ),
        "qualified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.manifest.read_text()),
        json.loads(args.contract.read_text()),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
