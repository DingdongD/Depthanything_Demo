from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import pytest

from tools.run_u250_depthanything_hybrid import gelu, resize_align_corners
from tools.run_u250_depthanything_hybrid import layer_norm


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def extension(tmp_path_factory):
    pybind11 = pytest.importorskip("pybind11")
    directory = tmp_path_factory.mktemp("host_graph_extension")
    source = (ROOT / "tools/fpga_dma_batch.cpp").read_text().replace(
        "PYBIND11_MODULE(fpgaDmaBatch,",
        "PYBIND11_MODULE(fpgaDmaBatchHostGraphTest,",
    )
    cpp = directory / "fpga_dma_batch_host_graph_test.cpp"
    cpp.write_text(source)
    output = directory / (
        "fpgaDmaBatchHostGraphTest" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [
            "g++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-ffp-contract=off", "-pthread", "-shared", "-fPIC",
            "-I" + pybind11.get_include(),
            "-I" + sysconfig.get_path("include"),
            "-I" + str(ROOT / "tools"), str(cpp), "-o", str(output),
        ],
        check=True,
        capture_output=True,
    )
    spec = importlib.util.spec_from_file_location(
        "fpgaDmaBatchHostGraphTest", output
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_quantize(value, scale):
    return np.clip(np.rint(np.asarray(value, np.float32) / scale),
                   -128, 127).astype(np.int8)


def ndwc_descriptor(dims, bitdepth, direction, index=0):
    element_bytes = bitdepth // 8
    c_align = dims[1] * ((dims[3] + 15) // 16) * element_bytes
    w_align = ((dims[2] + 15) // 16) * c_align
    return {
        "layout": "NDWC", "dims": list(dims), "bitdepth": bitdepth,
        "c_align": c_align, "w_align": w_align,
        "combined_bytes": w_align * 256, "direction": direction,
        "index": index,
        "matrix_role": "output" if direction == "output" else "left",
    }


def nchw_descriptor(dims, bitdepth, direction, index=0):
    element_bytes = bitdepth // 8
    c_align = ((dims[1] + 15) // 16) * element_bytes
    w_align = ((dims[3] + 15) // 16) * c_align
    combined = (dims[0] * dims[2] * ((dims[3] + 15) // 16)
                * ((dims[1] + 15) // 16) * 256 * element_bytes)
    return {
        "layout": "NCHW", "dims": list(dims), "bitdepth": bitdepth,
        "c_align": c_align, "w_align": w_align,
        "combined_bytes": combined, "direction": direction,
        "index": index, "matrix_role": "netio",
    }


def test_host_executor_construction_does_not_construct_dma(extension):
    """Catch accidental composition with DmaBatch, whose constructor opens XDMA."""
    executor = extension.HostGraphExecutor()
    assert executor.stats()["host_calls"] == 0


@pytest.mark.parametrize(
    "scale", [0.00390625, 0.03993530943989754, 0.10580708831548691]
)
def test_quantize_is_bit_exact_at_rounding_and_saturation_boundaries(
    extension, scale
):
    """Catch truncation, non-even rounding, wrong scale precision, or wrapping."""
    values = np.array(
        [-1000, -127.5 * scale, -2.5 * scale, -0.5 * scale, -0.0,
         0.5 * scale, 1.5 * scale, 126.5 * scale, 127.5 * scale, 1000],
        np.float32,
    )
    actual = extension.HostGraphExecutor().quantize(values, scale)
    assert actual.dtype == np.int8
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, python_quantize(values, scale))


def test_gelu_quantize_matches_the_current_runtime_boundary(extension):
    """Catch FP contraction or approximation changes that alter FC2 INT8 codes."""
    values = np.linspace(-128, 12, 262145, dtype=np.float32).reshape(
        1, 1, 1, -1
    )
    scale = 0.03993530943989754
    expected = python_quantize(gelu(values), scale)
    actual = extension.HostGraphExecutor().gelu_quantize(values, scale)
    assert np.array_equal(actual, expected)


def test_gelu_quantize_caches_exact_bf16_domain_by_scale(extension):
    bits = np.arange(65536, dtype=np.uint32) << np.uint32(16)
    values = bits.view(np.float32)
    values = np.ascontiguousarray(values[np.isfinite(values)])
    scale = 0.03993530943989754
    with np.errstate(over="ignore", invalid="ignore"):
        expected = python_quantize(gelu(values), scale)
    first_executor = extension.HostGraphExecutor()
    second_executor = extension.HostGraphExecutor()
    first = first_executor.gelu_quantize(values, scale)
    second = second_executor.gelu_quantize(values, scale)
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert first_executor.stats()["gelu_lut_misses"] == 1
    assert second_executor.stats()["gelu_lut_hits"] == 1
    assert first_executor.stats()["gelu_lut_elements"] == values.size
    assert second_executor.stats()["gelu_lut_elements"] == values.size


def test_physical_bf16_gelu_concatenate_packs_fc2_input_bit_exact(extension):
    rng = np.random.default_rng(650)
    channels = (16, 32, 16, 48, 16, 32)
    sources = []
    logical = []
    descriptors = []
    for index, count in enumerate(channels):
        descriptor = ndwc_descriptor((1, 1, 37, count), 16, "output", index)
        value = rng.standard_normal(descriptor["dims"], dtype=np.float32)
        physical = extension.DmaBatch.pack_tensor(value, descriptor)
        sources.append(physical)
        logical.append(extension.DmaBatch.unpack_tensor(*physical, descriptor))
        descriptors.append(descriptor)
    target = ndwc_descriptor((1, 1, 37, sum(channels)), 8, "input")
    scale = 0.03993530943989754
    expected = extension.DmaBatch.pack_tensor(
        python_quantize(gelu(np.concatenate(logical, axis=3)), scale), target
    )
    executor = extension.HostGraphExecutor()
    actual = executor.gelu_pack_bf16_concatenate(
        sources, descriptors, target, scale
    )
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    assert executor.stats()["gelu_pack_bf16_concatenate_calls"] == 1


def test_physical_bf16_gelu_rejects_nonfinite_source(extension):
    source = ndwc_descriptor((1, 1, 16, 16), 16, "output")
    target = ndwc_descriptor((1, 1, 16, 16), 8, "input")
    values = np.ones(source["dims"], np.float32)
    values[0, 0, 0, 0] = np.inf
    # Construct the BF16 bank pair directly because the generic packer rightly
    # rejects non-finite logical BF16 input before physical layout conversion.
    physical = list(extension.DmaBatch.pack_tensor(
        np.ones(source["dims"], np.float32), source
    ))
    physical[0][0:2] = np.array([0x80, 0x7f], np.uint8)
    with pytest.raises(ValueError, match="finite"):
        extension.HostGraphExecutor().gelu_pack_bf16_concatenate(
            [tuple(physical)], [source], target, 0.125
        )


def test_physical_attention_chunks_pack_heads_bit_exact(extension):
    rng = np.random.default_rng(660)
    valid_widths = [4, 4, 2] * 2
    physical, descriptors, logical = [], [], []
    for index in range(6):
        descriptor = ndwc_descriptor((1, 1, 4, 16), 16, "output", index % 2)
        value = rng.standard_normal(descriptor["dims"], dtype=np.float32)
        banks = extension.DmaBatch.pack_tensor(value, descriptor)
        physical.append(banks)
        descriptors.append(descriptor)
        logical.append(extension.DmaBatch.unpack_tensor(*banks, descriptor))
    target = ndwc_descriptor((1, 1, 10, 32), 8, "input")
    assembled = np.concatenate([
        np.concatenate([
            logical[head * 3 + chunk][:, :, :valid_widths[head * 3 + chunk]]
            for chunk in range(3)
        ], axis=2)
        for head in range(2)
    ], axis=3)
    scale = 0.10580708831548691
    expected = extension.DmaBatch.pack_tensor(python_quantize(assembled, scale), target)
    executor = extension.HostGraphExecutor()
    actual = executor.attention_pack_bf16_heads(
        physical, descriptors, valid_widths, target, scale, 2
    )
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    stats = executor.stats()
    assert stats["attention_pack_bf16_heads_calls"] == 1
    assert stats["quantize_lut_misses"] == 1


@pytest.mark.parametrize(
    "valid_widths,heads,match",
    [([4, 4, 1] * 2, 2, "cover"), ([4, 4, 3] * 2, 2, "exceed"),
     ([4, 4, 2] * 2, 3, "head-aligned")],
)
def test_physical_attention_rejects_incomplete_geometry(
    extension, valid_widths, heads, match
):
    descriptor = ndwc_descriptor((1, 1, 4, 16), 16, "output")
    banks = extension.DmaBatch.pack_tensor(
        np.ones(descriptor["dims"], np.float32), descriptor
    )
    target = ndwc_descriptor((1, 1, 10, 32), 8, "input")
    with pytest.raises(ValueError, match=match):
        extension.HostGraphExecutor().attention_pack_bf16_heads(
            [banks] * 6, [descriptor] * 6, valid_widths, target, 0.125, heads
        )


def test_decoder_capture_physical_bridge_matches_logical_boundary(extension):
    rng = np.random.default_rng(681)
    channels, height, width = 32, 3, 5
    source = ndwc_descriptor(
        (1, 1, height * width + 1, channels), 16, "output"
    )
    target = nchw_descriptor((1, channels, height, width), 8, "input")
    capture = rng.standard_normal(source["dims"], dtype=np.float32)
    physical = extension.DmaBatch.pack_tensor(capture, source)
    logical_capture = extension.DmaBatch.unpack_tensor(*physical, source)
    gamma = rng.standard_normal(channels, dtype=np.float32)
    beta = rng.standard_normal(channels, dtype=np.float32)
    scale = 0.04125
    normalized = layer_norm(logical_capture[:, 0], gamma, beta, -1, 1.0e-6)
    image = np.ascontiguousarray(
        normalized[:, 1:].transpose(0, 2, 1).reshape(
            1, channels, height, width
        )
    )
    expected = extension.DmaBatch.pack_tensor(
        python_quantize(image, scale), target
    )
    executor = extension.HostGraphExecutor()
    actual = executor.decoder_capture_pack_bf16(
        physical, source, gamma, beta, target, scale, 1.0e-6
    )
    assert all(np.array_equal(left, right)
               for left, right in zip(actual, expected))
    assert executor.stats()["decoder_capture_pack_bf16_calls"] == 1


def test_add_and_add_quantize_preserve_shape_and_values(extension):
    """Catch mismatched shapes, incorrect FP32 addition, or a second rounding rule."""
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4)
    executor = extension.HostGraphExecutor()
    added = executor.add(left, right)
    assert added.shape == left.shape
    assert np.array_equal(added, (left + right).astype(np.float32))
    assert np.array_equal(
        executor.add_quantize(left, right, 0.125),
        python_quantize((left + right).astype(np.float32), 0.125),
    )


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_concatenate_matches_numpy_for_every_axis_form(extension, axis):
    """Catch incorrect outer/inner-stride copying or negative-axis handling."""
    left = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
    right = (np.arange(16, dtype=np.int8) + 40).reshape(2, 2, 4)
    if axis not in (1,):
        if axis in (0,):
            right = (np.arange(24, dtype=np.int8) + 40).reshape(2, 3, 4)
        else:
            right = (np.arange(12, dtype=np.int8) + 40).reshape(2, 3, 2)
    actual = extension.HostGraphExecutor().concatenate([left, right], axis)
    assert np.array_equal(actual, np.concatenate([left, right], axis=axis))


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ((1, 3, 2, 3), (1, 3, 1, 1)),
        ((1, 2, 19, 37), (1, 2, 37, 74)),
        ((1, 2, 37, 74), (1, 2, 74, 148)),
        ((1, 1, 74, 148), (1, 1, 148, 296)),
        ((1, 1, 148, 296), (1, 1, 296, 518)),
        ((1, 2, 19, 37), (1, 2, 75, 518)),
    ],
)
def test_resize_align_corners_is_bit_exact_for_decoder_extents(
    extension, input_shape, output_shape
):
    rng = np.random.default_rng(sum(input_shape) + sum(output_shape))
    value = rng.standard_normal(input_shape, dtype=np.float32)
    expected = resize_align_corners(value, np.asarray(output_shape, np.int64))
    actual = extension.HostGraphExecutor().resize_align_corners(
        value, output_shape[2], output_shape[3]
    )
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "operation,match",
    [
        (lambda e: e.quantize(np.arange(4, dtype=np.float64), 1.0), "float32"),
        (lambda e: e.quantize(np.arange(4, dtype=np.float32)[::2], 1.0),
         "C-contiguous"),
        (lambda e: e.quantize(np.array([np.nan], np.float32), 1.0), "finite"),
        (lambda e: e.quantize(np.ones(1, np.float32), 0.0), "positive"),
        (lambda e: e.add(np.ones(2, np.float32), np.ones(3, np.float32)),
         "shape"),
        (lambda e: e.concatenate([], 0), "non-empty"),
        (lambda e: e.resize_align_corners(
            np.ones((1, 2, 3), np.float32), 4, 4), "rank-4"),
        (lambda e: e.resize_align_corners(
            np.ones((1, 1, 2, 2), np.float64), 4, 4), "float32"),
        (lambda e: e.resize_align_corners(
            np.ones((1, 1, 2, 2), np.float32), 0, 4), "positive"),
    ],
)
def test_host_executor_rejects_unsafe_or_ambiguous_inputs(extension, operation, match):
    """Catch implicit dtype/layout conversion and malformed execution contracts."""
    with pytest.raises((TypeError, ValueError), match=match):
        operation(extension.HostGraphExecutor())


def test_stats_reset_clears_counters_without_reconstructing_executor(extension):
    """Catch resident-frame statistics leaking across requests."""
    executor = extension.HostGraphExecutor()
    executor.quantize(np.ones(8, np.float32), 0.5)
    executor.gelu_quantize(np.ones(8, np.float32), 0.5)
    executor.resize_align_corners(np.ones((1, 1, 2, 2), np.float32), 3, 3)
    stats = executor.stats()
    assert stats["host_calls"] == 3
    assert stats["quantize_calls"] == 1
    assert stats["gelu_quantize_calls"] == 1
    assert stats["resize_align_corners_calls"] == 1
    executor.reset_stats()
    assert executor.stats()["host_calls"] == 0
