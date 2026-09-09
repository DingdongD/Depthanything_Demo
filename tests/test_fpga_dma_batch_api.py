from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import pytest


EXPECTED = {
    "h2c_batch", "h2c_batch_safe", "c2h_batch", "c2h_batch_safe",
    "run_npu_chain", "run_resident_transaction", "run_frame_graph",
    "run_decoder_capture_stems",
    "run_cbam_fused_pool",
    "pack_int8_nchw",
    "pack_int8_nchw_segments", "unpack_int8_nchw",
    "interleave_polyphase_normal16", "requantize_normal16",
    "crop_normal16_tiles", "scatter_normal16_tiles", "stats", "reset_stats",
    "validate_descriptor",
}


@pytest.fixture(scope="module")
def extension_type():
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
    return module.DmaBatch


def test_extension_preserves_qualified_api(extension_type):
    assert EXPECTED <= set(dir(extension_type))


def test_stale_event_stats_and_reset_without_device_access(tmp_path):
    """Replace only hardware construction; execute the real C++ stats/reset API."""
    pybind11 = pytest.importorskip("pybind11")
    source = (Path(__file__).resolve().parents[1] / "tools/fpga_dma_batch.cpp").read_text()
    begin = source.index("  DmaBatch() {")
    end = source.index("  ~DmaBatch()", begin)
    source = (source[:begin] + "  DmaBatch() { stale_events_ = 7; }\n\n" + source[end:])
    source = source.replace("PYBIND11_MODULE(fpgaDmaBatch,", "PYBIND11_MODULE(fpgaDmaBatchStatsTest,")
    cpp = tmp_path / "stats.cpp"
    cpp.write_text(source)
    extension = tmp_path / ("fpgaDmaBatchStatsTest" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["g++", "-O0", "-std=c++17", "-shared", "-fPIC", "-pthread",
                    "-I" + pybind11.get_include(), "-I" + sysconfig.get_path("include"),
                    "-I" + str(Path(__file__).resolve().parents[1] / "tools"),
                    str(cpp), "-o", str(extension)], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("fpgaDmaBatchStatsTest", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.DmaBatch()
    stale = runtime.stats()["stale_events"]
    assert type(stale) is int and stale == 7
    runtime.reset_stats()
    assert runtime.stats()["stale_events"] == 0

    initial = {
        "left": np.array([1.0, -2.0], np.float32),
        "right": np.array([0.5, 1.0], np.float32),
    }
    result = runtime.run_frame_graph(
        initial,
        [
            {"op": "add", "left": "left", "right": "right", "output": "sum"},
            {"op": "quantize", "input": "sum", "output": "q", "scale": 0.5},
        ],
        ["q"], 1000, False,
    )
    assert set(initial) == {"left", "right"}
    assert np.array_equal(result["outputs"]["q"], np.array([3, -2], np.int8))
    assert result["nodes"] == 2
    assert result["programs"] == 0
    assert result["host_graph"]["host_calls"] == 2
    assert runtime.stats()["frame_graph_calls"] == 1

    runtime.reset_stats()
    with pytest.raises(ValueError, match="undefined tensor missing"):
        runtime.run_frame_graph(
            {},
            [
                {"op": "device_read", "outputs": ["would_touch_board"],
                 "requests": [[(0, 0, 128), (1, 0x400000000, 128)]]},
                {"op": "quantize", "input": "missing", "output": "q",
                 "scale": 1.0},
            ],
            ["q"], 1000, False,
        )
    assert runtime.stats()["c2h_batches"] == 0
