from __future__ import annotations

import pytest

from tools.run_u250_depthanything_hybrid import (
    decoder_capture_offset_plan,
    encoder_resident_offset_plan,
)


def tensor(address: int, size: int) -> dict:
    return {"address": address, "size_per_bank": size}


def records_and_contract():
    records = {
        "norm": {
            "name": "norm", "inputs": [tensor(0, 1_056_768)],
            "outputs": [tensor(4_128, 1_056_768)],
        },
        "post": {
            "name": "post",
            "inputs": [tensor(0, 528_384), tensor(2_064, 1_056_768)],
            "outputs": [tensor(6_192, 1_056_768)],
        },
        "qkv": {
            "name": "qkv", "inputs": [tensor(0, 2_113_536)], "outputs": [],
        },
        "attention": {
            "name": "attention", "inputs": [tensor(0, 2_113_536)], "outputs": [],
        },
    }
    blocks = []
    for layer in range(12):
        blocks.append({
            "layer": layer, "host_norm1": {"npu_core": "norm"},
            "qkv": {"kernel": "qkv"},
            "attention": {"heads": [{"kernel": "attention"}]},
            "post_attention": {"kernel": "post"},
            "capture_for_decoder": layer in (2, 5, 8, 11),
        })
    return records, {"encoder": blocks}


def test_capture_plan_fits_four_persistent_inputs_and_decoder_norm_output():
    records, contract = records_and_contract()
    workspace = 65_536 * 128
    offsets = decoder_capture_offset_plan(records, contract, workspace)
    assert offsets == {2: 20_640, 5: 33_024, 8: 45_408, 11: 53_664}

    for source_layer in (2, 5, 8):
        block = contract["encoder"][source_layer + 1]
        plan = encoder_resident_offset_plan(
            records, block, workspace, offsets[source_layer]
        )
        assert plan["x_begin"] == offsets[source_layer]
        assert plan["x_end"] == offsets[source_layer] + 4_128
        assert plan["scratch_end"] <= 65_536


def test_capture_plan_fails_closed_when_workspace_cannot_hold_last_norm():
    records, contract = records_and_contract()
    with pytest.raises(RuntimeError, match="decoder"):
        decoder_capture_offset_plan(records, contract, 61_919 * 128)


def test_capture_plan_reserves_all_fused_fc1_workspace_slots():
    records, contract = records_and_contract()
    records.update({
        f"fc1_{index}": {
            "name": f"fc1_{index}",
            "inputs": [tensor(0, 262_144)],
            "outputs": [tensor(1_024, 262_144)],
        }
        for index in range(6)
    })
    records["fc2"] = {
        "name": "fc2", "inputs": [tensor(0, 262_144)],
        "outputs": [tensor(1_024, 262_144)],
    }
    for block in contract["encoder"]:
        block["mlp"] = {
            "fc1_kernels": [f"fc1_{index}" for index in range(6)],
            "fc2_kernel": "fc2",
        }

    workspace = 131_072 * 128
    offsets = decoder_capture_offset_plan(records, contract, workspace)
    # One FC1 record spans 2,048 address units; six simultaneous slots occupy
    # [0, 12,288), which must remain disjoint from every retained capture.
    assert offsets[2] == 12_288 + 3 * 4_128
    assert offsets[2] >= 6 * 2_048
