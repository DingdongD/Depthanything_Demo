from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.analyze_u250_encoder_residency import analyze, storage_abi


ROOT = Path("/root/demo/depthanything/build/depthanything_u250_resident_kernel_bank_r43_lnfold_decoderretuned")


@pytest.fixture(scope="module")
def deployment():
    if not ROOT.exists():
        pytest.skip("local r43 deployment evidence is unavailable")
    return (
        json.loads((ROOT / "resident_kernel_bank_manifest.json").read_text()),
        json.loads((ROOT / "depthanything_u250_runtime_contract.json").read_text()),
    )


def test_r43_encoder_residency_plan_is_in_bounds_and_exact(deployment):
    report = analyze(*deployment)
    intervals = report["address_plan"]["intervals"]
    assert [(item["begin_units"], item["end_units"]) for item in intervals] == [
        (0, 8256), (8256, 12384), (12384, 16512), (16512, 20640)
    ]
    assert report["workspace"]["units_per_bank"] == 65536
    assert report["workspace"]["headroom_units_per_bank"] == 44896
    assert report["projected_steady_frame_savings"] == {
        "h2c_bytes": 25362432,
        "native_pack_calls": 24,
        "native_pack_logical_bytes": 50503680,
        "native_pack_physical_bytes": 25362432,
        "c2h_bytes": 0,
        "python_submission_groups": 12,
        "npu_dispatches": 0,
    }
    assert all(
        item["residual_x_norm1_to_post_storage_abi"]
        and item["post_to_norm2_storage_abi"]
        for item in report["layer_checks"]
    )


def test_storage_abi_ignores_address_and_direction_only():
    output = {
        "address": 6192, "layout": "NDWC", "dims": [1, 1, 1370, 384],
        "bitdepth": 16, "size_per_bank": 1056768,
    }
    consumer = {**output, "address": 0, "direction": "input"}
    assert storage_abi(output) == storage_abi(consumer)
    consumer["bitdepth"] = 8
    assert storage_abi(output) != storage_abi(consumer)


def test_incompatible_post_to_norm_fails_closed(deployment):
    manifest, contract = copy.deepcopy(deployment)
    records = {item["name"]: item for item in manifest["cases"]}
    records["post_attention_l00"]["outputs"][0]["bitdepth"] = 8
    with pytest.raises(ValueError, match="required resident storage ABI"):
        analyze(manifest, contract)


def test_workspace_overflow_fails_closed(deployment):
    manifest, contract = copy.deepcopy(deployment)
    manifest["shared_fm_workspace_bytes"] = 2 * 20000 * 128
    with pytest.raises(ValueError, match="exceeds shared FM workspace"):
        analyze(manifest, contract)
