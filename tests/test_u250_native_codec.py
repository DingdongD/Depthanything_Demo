from __future__ import annotations

from dataclasses import asdict
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from tools.u250_layout_descriptors import TensorLayoutDescriptor


@pytest.fixture(scope="module")
def codec():
    extension_path = os.environ.get("FPGA_DMA_BATCH_SO")
    if extension_path is None:
        pytest.skip("FPGA_DMA_BATCH_SO is not set")

    spec = importlib.util.spec_from_file_location(
        "fpgaDmaBatch", Path(extension_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FPGA DMA extension from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(**overrides):
    result = {
        "layout": "NDWC",
        "dims": [1, 1, 1370, 384],
        "bitdepth": 16,
        "c_align": 48,
        "w_align": 4128,
        "combined_bytes": 1056768,
        "direction": "output",
        "index": 0,
    }
    result.update(overrides)
    return result


def test_validate_descriptor_normalizes_fields(codec):
    result = codec.DmaBatch.validate_descriptor(descriptor())

    assert result["half_bytes"] == 528384
    assert result["elements"] == 526080
    assert result["dims"] == [1, 1, 1370, 384]


def test_validate_descriptor_retains_task_one_matrix_role(codec):
    task_one_descriptor = TensorLayoutDescriptor.from_tensor(
        "attention2_l00_h00",
        {
            "layout": "NDWC",
            "dims": [1, 1, 1370, 64],
            "bitdepth": 8,
            "c_align": 8,
            "w_align": 4128,
            "size_per_bank": 88064,
        },
        "input",
        1,
    )

    result = codec.DmaBatch.validate_descriptor(asdict(task_one_descriptor))

    assert result["matrix_role"] == "right"


@pytest.mark.parametrize(
    "matrix_role,match",
    [
        ("diagonal", "unsupported matrix_role"),
        ("left", "does not match layout and direction"),
    ],
)
def test_validate_descriptor_rejects_invalid_matrix_role(codec, matrix_role, match):
    with pytest.raises(ValueError, match=match):
        codec.DmaBatch.validate_descriptor(descriptor(matrix_role=matrix_role))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("dims", [1, 1370, 384], "rank four"),
        ("layout", "NHWC", "unsupported layout"),
        ("bitdepth", 32, "unsupported bitdepth"),
        ("c_align", 0, "alignment must be positive"),
        ("c_align", -1, "alignment must be positive"),
        ("w_align", 0, "alignment must be positive"),
        ("direction", "sideways", "invalid tensor direction"),
        ("combined_bytes", 1056767, "256-byte aligned"),
        ("combined_bytes", 1051904, "smaller than logical tensor"),
    ],
)
def test_validate_descriptor_rejects_invalid_common_fields(
    codec, field, value, match
):
    with pytest.raises(ValueError, match=match):
        codec.DmaBatch.validate_descriptor(descriptor(**{field: value}))


@pytest.mark.parametrize(
    "value,bits",
    [
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3F80),
        (-2.0, 0xC000),
        (1.00390625, 0x3F80),
        (1.01171875, 0x3F82),
    ],
)
def test_bf16_known_values_and_ties_round_to_even(codec, value, bits):
    assert codec._test_fp32_to_bf16(np.float32(value)) == bits


@pytest.mark.parametrize("value", [np.float32(np.inf), np.float32(np.nan)])
def test_bf16_rejects_nonfinite_values(codec, value):
    with pytest.raises(ValueError, match="finite"):
        codec._test_fp32_to_bf16(value)


def test_bf16_test_hook_requires_float32_scalar(codec):
    with pytest.raises(TypeError, match="float32"):
        codec._test_fp32_to_bf16(np.float64(1.0))
