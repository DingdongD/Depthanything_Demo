from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import pytest

from tools.run_u250_depthanything_hybrid import gelu


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
    stats = executor.stats()
    assert stats["host_calls"] == 2
    assert stats["quantize_calls"] == 1
    assert stats["gelu_quantize_calls"] == 1
    executor.reset_stats()
    assert executor.stats()["host_calls"] == 0
