from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


EXPECTED = {
    "h2c_batch", "h2c_batch_safe", "c2h_batch", "c2h_batch_safe",
    "run_npu_chain", "run_cbam_fused_pool", "pack_int8_nchw",
    "pack_int8_nchw_segments", "unpack_int8_nchw",
    "interleave_polyphase_normal16", "requantize_normal16",
    "crop_normal16_tiles", "scatter_normal16_tiles", "stats", "reset_stats",
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
