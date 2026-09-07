from __future__ import annotations

import pytest

from tools.u250_layout_descriptors import (
    TensorLayoutDescriptor,
    build_case_descriptors,
)
from tools.run_u250_depthanything_hybrid import CfgCodecRegistry


def ndwc_tensor(**overrides):
    tensor = {
        "address": 0,
        "layout": "NDWC",
        "dims": [1, 1, 256, 64],
        "bitdepth": 8,
        "c_align": 4,
        "w_align": 64,
        "size_per_bank": 16384,
    }
    tensor.update(overrides)
    return tensor


def test_descriptor_identity_ignores_relocated_address():
    left = TensorLayoutDescriptor.from_tensor(
        "qkv_projection_l00", ndwc_tensor(address=0), "input", 0
    )
    right = TensorLayoutDescriptor.from_tensor(
        "qkv_projection_l00", ndwc_tensor(address=9000), "input", 0
    )
    assert left.identity() == right.identity()


def test_descriptor_rejects_invalid_extent():
    tensor = {
        "layout": "NCHW",
        "dims": [1, 64, 37, 37],
        "bitdepth": 16,
        "c_align": 8,
        "w_align": 24,
        "size_per_bank": 227327,
    }
    with pytest.raises(ValueError, match="256-byte aligned"):
        TensorLayoutDescriptor.from_tensor("decoder_conv_00", tensor, "output", 0)


def test_descriptor_assigns_matrix_roles_from_layout_direction_and_case():
    attention = TensorLayoutDescriptor.from_tensor(
        "attention2_l00_h00", ndwc_tensor(), "input", 1
    )
    other_input = TensorLayoutDescriptor.from_tensor(
        "qkv_projection_l00", ndwc_tensor(), "input", 1
    )
    output = TensorLayoutDescriptor.from_tensor(
        "attention2_l00_h00", ndwc_tensor(), "output", 0
    )
    nchw = TensorLayoutDescriptor.from_tensor(
        "decoder_conv_00",
        ndwc_tensor(layout="NCHW", dims=[1, 64, 1, 256]),
        "input",
        0,
    )

    assert attention.matrix_role == "right"
    assert other_input.matrix_role == "left"
    assert output.matrix_role == "output"
    assert nchw.matrix_role == "netio"


def test_build_case_descriptors_keeps_input_and_output_order():
    records = {
        "attention2_l00_h00": {
            "inputs": [ndwc_tensor(), ndwc_tensor(address=64)],
            "outputs": [ndwc_tensor(bitdepth=16, c_align=8, w_align=128,
                                      size_per_bank=32768)],
        }
    }

    descriptors = build_case_descriptors(records)

    assert [item.index for item in descriptors["attention2_l00_h00"]["input"]] == [0, 1]
    assert descriptors["attention2_l00_h00"]["input"][1].matrix_role == "right"
    assert descriptors["attention2_l00_h00"]["output"][0].matrix_role == "output"


def test_registry_enriches_copied_manifest_tensors_from_cfg(tmp_path):
    name = "attention2_l00_h00"
    (tmp_path / f"{name}_cfg.txt").write_text(
        "Address: 0 (0x0) Size: 16384 Layout: NDWC Dims: [1, 1, 256, 64] "
        "c_align: 4 w_align: 64 bitdepth: 8\n"
        "Output Address: 64 (0x40) Size: 32768 Layout: NDWC Dims: [1, 1, 256, 64] "
        "c_align: 8 w_align: 128 bitdepth: 16\n"
    )
    records = {
        name: {
            "inputs": [{"address": 0, "layout": "NDWC", "dims": [1, 1, 256, 64],
                        "bitdepth": 8, "size_per_bank": 16384}],
            "outputs": [{"address": 64, "layout": "NDWC", "dims": [1, 1, 256, 64],
                         "bitdepth": 16, "size_per_bank": 32768}],
        }
    }

    registry = CfgCodecRegistry(tmp_path, records, object(), quiet=True)

    assert "c_align" not in records[name]["inputs"][0]
    assert registry.descriptors[name]["input"][0].c_align == 4
    assert registry.descriptors[name]["output"][0].w_align == 128
    assert registry.signature(name) == tuple(
        item.identity()
        for direction in ("input", "output")
        for item in registry.descriptors[name][direction]
    )


def test_registry_rejects_cfg_tensor_mismatch(tmp_path):
    name = "qkv_projection_l00"
    (tmp_path / f"{name}_cfg.txt").write_text(
        "Address: 0 (0x0) Size: 16384 Layout: NDWC Dims: [1, 1, 256, 64] "
        "c_align: 4 w_align: 64 bitdepth: 16\n"
    )
    records = {
        name: {
            "inputs": [{"address": 0, "layout": "NDWC", "dims": [1, 1, 256, 64],
                        "bitdepth": 8, "size_per_bank": 16384}],
            "outputs": [],
        }
    }

    with pytest.raises(ValueError, match="qkv_projection_l00: input 0 bitdepth mismatch"):
        CfgCodecRegistry(tmp_path, records, object(), quiet=True)
