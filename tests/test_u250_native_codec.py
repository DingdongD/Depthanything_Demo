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


def nchw_descriptor(shape, bitdepth, compact=False, **overrides):
    n, c, h, w = shape
    ca, wa = (4, 64) if compact else (16, 16)
    extent = n * h * ((c + ca - 1) // ca * ca) * ((w + wa - 1) // wa * wa)
    c_stride = ((c + ca - 1) // ca) * (bitdepth // 8)
    w_stride = ((w + wa - 1) // wa) * c_stride
    return descriptor(layout="NCHW", dims=list(shape), bitdepth=bitdepth,
                      c_align=c_stride, w_align=w_stride, combined_bytes=extent * (bitdepth // 8),
                      **overrides)


def bf16_reference(value):
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


@pytest.mark.parametrize("shape,bitdepth,compact", [
    ((1, 64, 37, 37), 8, False), ((1, 48, 76, 148), 8, False),
    ((1, 64, 74, 148), 16, False), ((1, 1, 74, 518), 16, False),
    ((2, 17, 3, 19), 8, False), ((2, 3, 3, 65), 16, True),
])
def test_nchw_round_trip(codec, shape, bitdepth, compact):
    values = ((np.arange(np.prod(shape), dtype=np.int64) * 73 + 19) % 256 - 128)
    logical = values.astype(np.int8 if bitdepth == 8 else np.float32).reshape(shape)
    if bitdepth == 16:
        logical *= np.float32(1.00390625)
    desc = nchw_descriptor(shape, bitdepth, compact)
    even, odd = codec.DmaBatch.pack_tensor(logical, desc)
    assert even.dtype == odd.dtype == np.dtype(np.uint8)
    assert even.shape == odd.shape == (desc["combined_bytes"] // 2,)
    actual = codec.DmaBatch.unpack_tensor(even, odd, desc)
    expected = bf16_reference(logical) if bitdepth == 16 else logical
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize("bitdepth,compact,shape,offsets", [
    (8, False, (1, 17, 1, 19), [0, 16, 112, 256, 288, 384]),
    (16, False, (1, 17, 1, 19), [0, 32, 224, 512, 576, 768]),
    (8, True, (1, 3, 1, 65), [0, 16, 112, 4, 8, 128]),
    (16, True, (1, 3, 1, 65), [0, 32, 224, 8, 16, 256]),
])
def test_nchw_known_lanes_and_zero_padding(codec, bitdepth, compact, shape, offsets):
    logical = np.zeros(shape, dtype=np.int8 if bitdepth == 8 else np.float32)
    coords = [(0, 0), (0, 1), (0, 7), (0, 16),
              (0, 32) if compact else (0, 18),
              (0, 64) if compact else (16, 16)]
    # Explicit hand-derived bank lane fixture, including odd-bank width 8.
    values = [-128, 127, -3, 5, 9, 11] if bitdepth == 8 else [1., -2., .5, 4., -8., 16.]
    for (c, x), value in zip(coords, values):
        logical[0, c, 0, x] = value
    logical[0, 0, 0, 8] = 1
    desc = nchw_descriptor(shape, bitdepth, compact)
    expected_even = np.zeros(desc["combined_bytes"] // 2, np.uint8)
    expected_odd = expected_even.copy()
    bits = [128, 127, 253, 5, 9, 11] if bitdepth == 8 else [0x3F80, 0xC000, 0x3F00, 0x4080, 0xC100, 0x4180]
    for offset, value in zip(offsets, bits):
        target = expected_even
        if bitdepth == 16 and offset == 224:
            target, offset = expected_odd, 96
        target[offset] = value & 255
        if bitdepth == 16:
            target[offset + 1] = value >> 8
    if bitdepth == 8:
        expected_odd[0] = 1
    if bitdepth == 16:
        expected_even[128:130] = [128, 63]
    even, odd = codec.DmaBatch.pack_tensor(logical, desc)
    np.testing.assert_array_equal(even, expected_even)
    np.testing.assert_array_equal(odd, expected_odd)


@pytest.mark.parametrize("extent", [2304, 2816])
def test_nchw_descriptor_rejects_inexact_padded_extent(codec, extent):
    desc = nchw_descriptor((1, 17, 5, 16), 8)
    desc["combined_bytes"] = extent  # Exact physical extent is 2560.
    with pytest.raises(ValueError, match="physical extent"):
        codec.DmaBatch.validate_descriptor(desc)


@pytest.mark.parametrize("field,value", [("c_align", 8), ("w_align", 32)])
def test_nchw_descriptor_rejects_unsupported_alignment(codec, field, value):
    desc = nchw_descriptor((1, 17, 5, 16), 8)
    desc[field] = value
    with pytest.raises(ValueError, match="alignment"):
        codec.DmaBatch.validate_descriptor(desc)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float64])
def test_nchw_pack_rejects_incompatible_dtype(codec, dtype):
    desc = nchw_descriptor((1, 16, 1, 16), 8)
    with pytest.raises((TypeError, ValueError), match="dtype"):
        codec.DmaBatch.pack_tensor(np.zeros(desc["dims"], dtype), desc)


def test_nchw_pack_rejects_shape_and_noncontiguous_input(codec):
    desc = nchw_descriptor((1, 16, 2, 16), 8)
    with pytest.raises(ValueError, match="shape"):
        codec.DmaBatch.pack_tensor(np.zeros((1, 16, 1, 16), np.int8), desc)
    with pytest.raises(ValueError, match="contiguous"):
        codec.DmaBatch.pack_tensor(np.zeros(desc["dims"], np.int8)[..., ::-1], desc)


def test_nchw_pack_rejects_nonfinite_bf16_before_parallel_work(codec):
    desc = nchw_descriptor((1, 64, 37, 37), 16)
    logical = np.zeros(desc["dims"], np.float32)
    logical[0, 0, -1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        codec.DmaBatch.pack_tensor(logical, desc)


@pytest.mark.parametrize("even_size,odd_size", [(127, 128), (127, 127), (256, 256)])
def test_nchw_unpack_rejects_incorrect_bank_extents(codec, even_size, odd_size):
    desc = nchw_descriptor((1, 16, 1, 16), 8)
    with pytest.raises(ValueError, match="bank.*size"):
        codec.DmaBatch.unpack_tensor(np.zeros(even_size, np.uint8),
                                    np.zeros(odd_size, np.uint8), desc)


def test_nchw_unpack_rejects_incompatible_bank_dtype(codec):
    desc = nchw_descriptor((1, 16, 1, 16), 8)
    with pytest.raises((TypeError, ValueError), match="dtype"):
        codec.DmaBatch.unpack_tensor(np.zeros(128, np.int8), np.zeros(128, np.uint8), desc)


def test_oracle_mismatch_records_first_physical_byte():
    from tools.validate_u250_native_codecs import first_mismatch
    assert first_mismatch(np.array([0, 9, 4], np.uint8),
                          np.array([0, 8, 4], np.uint8)) == {
        "kind": "value", "flat_index": 1, "coordinate": [1],
        "actual": 9, "expected": 8,
    }


def test_oracle_rejects_shape_dtype_and_signed_zero_mismatches():
    from tools.validate_u250_native_codecs import first_mismatch
    assert first_mismatch(np.zeros(2, np.uint8), np.zeros(3, np.uint8))["kind"] == "shape"
    assert first_mismatch(np.zeros(2, np.int8), np.zeros(2, np.uint8))["kind"] == "dtype"
    mismatch = first_mismatch(np.array([-0.0], np.float32), np.array([0.0], np.float32))
    assert mismatch["flat_index"] == 0
    assert mismatch["actual_bits"] == "80000000"
    assert mismatch["expected_bits"] == "00000000"


def test_oracle_probes_exercise_coordinate_tails_and_bf16_boundaries():
    from tools.validate_u250_native_codecs import deterministic_tensor
    shape = (2, 17, 3, 19)
    signed = deterministic_tensor(shape, 8, "coordinates")
    assert signed.dtype == np.int8 and signed.shape == shape
    assert set(signed.ravel().tolist()) == set(range(-128, 128))
    boundary = deterministic_tensor(shape, 16, "boundaries")
    assert np.isfinite(boundary).all()
    assert {0, 0x80000000, 0x3F808000, 0x3F818000, 0x007FFFFF, 0x00800000}.issubset(
        set(boundary.view(np.uint32).ravel().tolist()))


def test_oracle_permutation_probe_breaks_int8_coordinate_period():
    from tools.validate_u250_native_codecs import deterministic_tensor
    shape = (1, 588, 3, 19)
    value = deterministic_tensor(shape, 8, "permutation")
    assert not np.array_equal(value[:, :256], value[:, 256:512])
    np.testing.assert_array_equal(value, deterministic_tensor(shape, 8, "permutation"))
