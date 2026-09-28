from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from tools.analyze_u250_encoder_block_boundaries import main


def test_encoder_block_boundary_report_supports_split_fp32_archives(
    tmp_path, monkeypatch
) -> None:
    trace_root = tmp_path / "trace"
    reference_root = tmp_path / "reference"
    block_root = tmp_path / "block_reference"
    for root in (trace_root, reference_root, block_root):
        (root / "holdout").mkdir(parents=True)
    actual = np.asarray([1.0, 2.0], dtype=np.float32)
    reference = np.asarray([1.0, 1.0], dtype=np.float32)
    np.savez(
        trace_root / "holdout/sample.npz",
        q_l00=actual, k_l00=actual, v_l00=actual,
        attention_l00=actual, block_l00=actual,
    )
    np.savez(
        reference_root / "holdout/sample.npz",
        encoder_l00_q=reference, encoder_l00_k=reference,
        encoder_l00_v=reference, encoder_l00_attention=reference,
    )
    np.savez(block_root / "holdout/sample.npz", block_l00=reference)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_u250_encoder_block_boundaries.py",
        "--trace-dir", str(trace_root),
        "--reference-dir", str(reference_root),
        "--block-reference-dir", str(block_root),
        "--layer", "0", "--output", str(output),
    ])
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["sample_count"] == 1
    expected = np.sqrt(1.0 / 2.0)
    assert report["aggregate"]["raw_attention"]["relative_l2"] == pytest.approx(
        expected
    )
    assert report["aggregate"]["block_output"]["relative_l2"] == pytest.approx(
        expected
    )
