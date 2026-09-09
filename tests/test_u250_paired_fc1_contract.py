from __future__ import annotations

import copy

import pytest

from tools.enable_u250_paired_fc1_contract import enable_paired_fc1


def fixtures():
    encoder = []
    cases = []
    for layer in range(12):
        singles = [f"mlp_fc1_l{layer:02d}_c{chunk:02d}" for chunk in range(6)]
        pairs = [f"mlp_fc1_pair_l{layer:02d}_p{pair:02d}" for pair in range(3)]
        encoder.append({"layer": layer, "mlp": {"fc1_kernels": singles}})
        cases.extend({"name": name} for name in pairs)
    contract = {
        "encoder": encoder,
        "execution_totals": {
            "encoder_npu_calls": 348,
            "npu_calls_per_inference": 443,
            "resident_kernel_variants": 262,
        },
        "bank": {},
    }
    bank = {
        "cases": cases,
        "bank_size_bytes": 100,
        "bank_sha256": "a" * 64,
        "required_fm_io_bytes": 20,
        "shared_fm_workspace_bytes": 30,
    }
    return contract, bank


def test_enable_paired_fc1_updates_order_and_dispatch_totals():
    contract, bank = fixtures()
    result = enable_paired_fc1(contract, bank)
    assert result["encoder"][0]["mlp"]["fc1_kernels"] == [
        "mlp_fc1_pair_l00_p00", "mlp_fc1_pair_l00_p01", "mlp_fc1_pair_l00_p02",
    ]
    assert result["encoder"][0]["mlp"]["fc1_outputs_per_kernel"] == 2
    assert result["execution_totals"]["encoder_npu_calls"] == 312
    assert result["execution_totals"]["npu_calls_per_inference"] == 407
    assert result["execution_totals"]["resident_kernel_variants"] == 36
    assert result["encoder_fc1_dispatch_policy"]["physical_dispatch_reduction"] == 36
    assert contract["encoder"][0]["mlp"]["fc1_kernels"][0] == "mlp_fc1_l00_c00"


def test_enable_paired_fc1_rejects_missing_pair():
    contract, bank = fixtures()
    bank = copy.deepcopy(bank)
    bank["cases"].pop()
    with pytest.raises(ValueError, match="resident bank is missing"):
        enable_paired_fc1(contract, bank)


def test_enable_paired_fc1_rejects_reordered_single_outputs():
    contract, bank = fixtures()
    contract = copy.deepcopy(contract)
    contract["encoder"][0]["mlp"]["fc1_kernels"].reverse()
    with pytest.raises(ValueError, match="unexpected FC1 kernel order"):
        enable_paired_fc1(contract, bank)
