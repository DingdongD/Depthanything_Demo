from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_u250_multi_a8_probability import (  # noqa: E402
    chunk_id_for_rows,
    dynamic_row_max_codes,
    key_ranges,
    radix_codes,
)


def test_dynamic_row_max_avoids_positive_saturation() -> None:
    probability = np.asarray([[0.5, 0.25, 0.125, 0.001]], dtype=np.float32)
    codes, scales = dynamic_row_max_codes(probability)
    np.testing.assert_array_equal(codes, [[127, 64, 32, 0]])
    assert np.count_nonzero(codes == 127) <= 1
    assert scales.shape == (1, 1)
    assert float(scales[0, 0]) > 0.0


def test_radix_routes_encode_residual_instead_of_duplicate_probability() -> None:
    probability = np.asarray([[0.0, 1e-5, 7e-4, 0.01, 0.3, 1.0]], dtype=np.float32)
    codes, scales = radix_codes(probability, 2)
    dequantized = sum(
        code.astype(np.float32) * np.float32(scale)
        for code, scale in zip(codes, scales)
    )
    assert np.all(dequantized <= probability + np.float32(1e-7))
    assert np.max(probability - dequantized) <= scales[-1] * 1.01
    # A second full quantization would count the 0.3 entry twice.  The residual
    # digit must instead be strictly bounded by one coarse step.
    assert codes[1][0, 4] < 127


def test_more_radix_routes_monotonically_reduce_reconstruction_error() -> None:
    probability = np.linspace(0.0, 1.0, 1001, dtype=np.float32)[None]
    errors = []
    for routes in (1, 2, 3):
        codes, scales = radix_codes(probability, routes)
        reconstructed = sum(
            code.astype(np.float32) * np.float32(scale)
            for code, scale in zip(codes, scales)
        )
        errors.append(float(np.linalg.norm(reconstructed - probability)))
    assert errors[2] < errors[1] < errors[0]


def test_key_partitions_cover_each_token_once() -> None:
    coverage = np.zeros(1370, dtype=np.int32)
    for start, stop in key_ranges(1370, 4):
        coverage[start:stop] += 1
    np.testing.assert_array_equal(coverage, np.ones(1370, dtype=np.int32))


def test_query_chunk_mapping_matches_six_hardware_chunks() -> None:
    rows = np.asarray([0, 255, 256, 511, 512, 767, 768, 1023,
                       1024, 1279, 1280, 1369])
    np.testing.assert_array_equal(
        chunk_id_for_rows(rows), np.repeat(np.arange(6), 2)
    )
