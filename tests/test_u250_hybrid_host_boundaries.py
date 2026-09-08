from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import pytest

from tools.u250_host_executor import CppHostExecutor, PythonHostExecutor


ROOT = Path(__file__).resolve().parents[1]
QKV_SCALES = (
    0.0430610254406929, 0.06889763474464417, 0.09153543412685394,
    0.05437992140650749, 0.05373821407556534, 0.04650590568780899,
    0.05733267590403557, 0.059151601046323776, 0.09202755987644196,
    0.11466535180807114, 0.11761811375617981, 0.12696850299835205,
)
FC1_SCALES = (
    0.10580708831548691, 0.08858267962932587, 0.06053149700164795,
    0.0664370059967041, 0.05930118262767792, 0.057578738778829575,
    0.06594488024711609, 0.10334645956754684, 0.12598425149917603,
    0.12795275449752808, 0.12992125749588013, 0.11023622006177902,
)
FC2_SCALES = (
    0.03993530943989754, 0.041932038962841034, 0.017172586172819138,
    0.011247577145695686, 0.010979774408042431, 0.014516622759401798,
    0.013587468303740025, 0.016185909509658813, 0.015444914810359478,
    0.012788776308298111, 0.010787553153932095, 0.016942273825407028,
)


def deterministic_fp32(shape):
    index = np.arange(np.prod(shape), dtype=np.float32)
    return (np.sin(index * np.float32(0.017)) * np.float32(7.25)).reshape(shape)


@pytest.fixture(scope="module")
def executors(tmp_path_factory):
    pybind11 = pytest.importorskip("pybind11")
    directory = tmp_path_factory.mktemp("hybrid_boundaries")
    source = (ROOT / "tools/fpga_dma_batch.cpp").read_text().replace(
        "PYBIND11_MODULE(fpgaDmaBatch,",
        "PYBIND11_MODULE(fpgaDmaBatchHybridBoundaries,",
    )
    cpp = directory / "hybrid_boundaries.cpp"
    cpp.write_text(source)
    output = directory / (
        "fpgaDmaBatchHybridBoundaries" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    subprocess.run(
        [
            "g++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-ffp-contract=off", "-pthread", "-shared", "-fPIC",
            "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
            "-I" + str(ROOT / "tools"), str(cpp), "-o", str(output),
        ], check=True, capture_output=True,
    )
    spec = importlib.util.spec_from_file_location("fpgaDmaBatchHybridBoundaries", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return PythonHostExecutor(), CppHostExecutor(module)


@pytest.mark.parametrize("scale", QKV_SCALES + FC1_SCALES)
def test_all_encoder_input_quantized_boundaries_are_exact(executors, scale):
    python, cpp = executors
    value = deterministic_fp32((1, 1370, 384))
    assert np.array_equal(cpp.quantize(value, scale), python.quantize(value, scale))


@pytest.mark.parametrize("scale", FC2_SCALES)
def test_all_gelu_fc2_boundaries_are_exact(executors, scale):
    python, cpp = executors
    hidden = deterministic_fp32((1, 1, 1370, 1536))
    assert np.array_equal(
        cpp.gelu_quantize(hidden, scale), python.gelu_quantize(hidden, scale)
    )


def test_attention_six_head_three_piece_assembly_is_exact(executors):
    python, cpp = executors
    heads = []
    for head in range(6):
        pieces = [np.full((1, 1, rows, 64), head * 3 + part, np.float32)
                  for part, rows in enumerate((512, 512, 346))]
        heads.append(cpp.concatenate(pieces, axis=2))
    actual = cpp.concatenate(heads, axis=3)
    expected_heads = [python.concatenate([
        np.full((1, 1, rows, 64), head * 3 + part, np.float32)
        for part, rows in enumerate((512, 512, 346))
    ], axis=2) for head in range(6)]
    expected = python.concatenate(expected_heads, axis=3)
    assert np.array_equal(actual, expected)


def test_decoder_first_middle_last_tile_assembly_is_exact(executors):
    python, cpp = executors
    source = deterministic_fp32((1, 32, 48, 64))
    parts = [np.ascontiguousarray(source[:, :, :17]),
             np.ascontiguousarray(source[:, :, 17:33]),
             np.ascontiguousarray(source[:, :, 33:])]
    assert np.array_equal(
        cpp.concatenate(parts, axis=2), python.concatenate(parts, axis=2)
    )


def test_runner_routes_hot_boundaries_through_selected_executor():
    source = (ROOT / "tools/run_u250_depthanything_hybrid.py").read_text()
    assert "host_executor.gelu_quantize(" in source
    assert "host_executor.add(post, fc2)" in source
    assert source.count("host_executor.concatenate(") >= 6
