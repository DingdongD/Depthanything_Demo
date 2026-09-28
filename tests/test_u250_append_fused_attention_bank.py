from __future__ import annotations

import json

import pytest

from tools.append_u250_fused_qkv_attention_bank import (
    append_programs,
    install_qualified_codec_report,
)


def test_append_programs_moves_suffix_and_rebinds_every_fm_base():
    prefix = bytes(range(256)) * 4
    suffix = b"w" * 512
    manifest = {
        "shared_fm_placement": "suffix",
        "shared_fm_base_units": 4,
        "shared_fm_workspace_bytes": len(suffix),
        "cases": [{
            "name": "old",
            "base_addresses": [1, 2, 3, 4, 4, 6],
        }],
    }
    metadata = {
        "base_addresses": [0, 1, 2, 3, 9, 5],
        "isa_ranges": [5, 1],
        "inputs": [{"bitdepth": 8}],
        "outputs": [{"bitdepth": 16}] * 12,
    }
    updated, image = append_programs(manifest, prefix + suffix, [{
        "name": "fused",
        "layer": 0,
        "payload": b"p" * 256,
        "metadata": metadata,
        "cfg": "fused_cfg.txt",
        "binary": "fused_ddr.bin",
    }], 512)

    assert updated["shared_fm_base_units"] == 6
    assert updated["bank_size_bytes"] == 2048
    assert image[:1024] == prefix
    assert image[1024:1280] == b"p" * 256
    assert image[1536:] == suffix
    assert updated["cases"][0]["base_addresses"][4] == 6
    fused = updated["cases"][1]
    assert fused["offset_bytes"] == 1024
    assert fused["base_addresses"] == [4, 5, 6, 7, 6, 9]


def test_append_programs_rejects_duplicate_kernel_name():
    manifest = {
        "shared_fm_placement": "suffix",
        "shared_fm_base_units": 1,
        "shared_fm_workspace_bytes": 256,
        "cases": [{"name": "same", "base_addresses": [0] * 6}],
    }
    program = {
        "name": "same", "layer": 0, "payload": bytes(256),
        "metadata": {}, "cfg": "x", "binary": "y",
    }
    try:
        append_programs(manifest, bytes(512), [program], 256)
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("duplicate resident name was accepted")


def test_install_qualified_codec_report_rejects_wrong_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": []}) + "\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "qualified": True,
        "manifest_sha256": "0" * 64,
        "descriptors": [],
    }) + "\n")

    with pytest.raises(ValueError, match="does not match appended manifest"):
        install_qualified_codec_report(
            report, manifest, tmp_path, tmp_path / "active.json"
        )
