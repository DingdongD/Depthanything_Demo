from __future__ import annotations

import numpy as np
import pytest

from tools.run_u250_depthanything_hybrid import (
    attention_head_inputs,
    bf16_quantization_surrogate,
    crop_nchw_width,
    fp32_attention_head,
    pad_nchw_width,
)


def head(index: int = 2) -> dict:
    return {
        "head": index,
        "scales_bf16": {"q": 0.25, "k": 0.5, "v": 0.125},
    }


def test_int8_qkv_is_sliced_without_requantization() -> None:
    values = np.arange(1370 * 384, dtype=np.int64).reshape(1, 1, 1370, 384)
    q = values.astype(np.int8)
    k = (values + 1).astype(np.int8)
    v = (values + 2).astype(np.int8)

    got = attention_head_inputs(q, k, v, head())

    for source, actual in zip((q, k, v), got):
        np.testing.assert_array_equal(actual, source[0, 0, :, 128:192])


def test_bf16_decoded_qkv_uses_independent_per_head_scales() -> None:
    q = np.full((1, 1, 1370, 384), 0.5, np.float32)
    k = np.full_like(q, 1.0)
    v = np.full_like(q, -0.25)
    calls: list[float] = []

    def quantizer(value: np.ndarray, scale: float) -> np.ndarray:
        calls.append(scale)
        return np.rint(value / scale).astype(np.int8)

    qh, kh, vh = attention_head_inputs(q, k, v, head(), quantizer)

    assert calls == [0.25, 0.5, 0.125]
    np.testing.assert_array_equal(qh, np.full((1370, 64), 2, np.int8))
    np.testing.assert_array_equal(kh, np.full((1370, 64), 2, np.int8))
    np.testing.assert_array_equal(vh, np.full((1370, 64), -2, np.int8))


def test_mixed_qkv_abi_is_rejected() -> None:
    integer = np.zeros((1, 1, 1370, 384), np.int8)
    floating = np.zeros((1, 1, 1370, 384), np.float32)
    with pytest.raises(ValueError, match="uniformly INT8 or BF16"):
        attention_head_inputs(integer, floating, floating, head())


def test_bf16_qkv_requires_every_scale() -> None:
    floating = np.zeros((1, 1, 1370, 384), np.float32)
    record = head()
    del record["scales_bf16"]["k"]
    with pytest.raises(ValueError, match="positive per-head scales"):
        attention_head_inputs(floating, floating, floating, record)


def test_decoder_width_pad_and_crop_preserve_logical_tensor() -> None:
    source = np.arange(2 * 3 * 37, dtype=np.float32).reshape(1, 2, 3, 37)
    padded = pad_nchw_width(source, 64)

    assert padded.shape == (1, 2, 3, 64)
    assert padded.flags.c_contiguous
    np.testing.assert_array_equal(padded[:, :, :, :37], source)
    np.testing.assert_array_equal(padded[:, :, :, 37:], 0)
    np.testing.assert_array_equal(crop_nchw_width(padded, 37), source)


def test_decoder_width_pad_rejects_non_nchw_or_shrink() -> None:
    with pytest.raises(ValueError, match="rank 4"):
        pad_nchw_width(np.zeros((1, 2, 3), np.float32), 16)
    with pytest.raises(ValueError, match="smaller"):
        pad_nchw_width(np.zeros((1, 2, 3, 19), np.float32), 16)
    with pytest.raises(ValueError, match="invalid NCHW"):
        crop_nchw_width(np.zeros((1, 2, 3, 19), np.float32), 20)


def test_fp32_attention_head_matches_direct_formula() -> None:
    rng = np.random.default_rng(13)
    q = rng.normal(size=(19, 8)).astype(np.float32)
    k = rng.normal(size=(19, 8)).astype(np.float32)
    v = rng.normal(size=(19, 8)).astype(np.float32)
    logits = q @ k.T
    probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)

    actual = fp32_attention_head(q, k, v, rows_per_chunk=7)

    np.testing.assert_allclose(actual[0, 0], probability @ v, rtol=2e-6, atol=2e-6)


def test_fp32_attention_head_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="identical rank-2"):
        fp32_attention_head(np.zeros((2, 3), np.float32),
                            np.zeros((3, 2), np.float32),
                            np.zeros((2, 3), np.float32))


def test_bf16_surrogate_preserves_direct_int8_quantization() -> None:
    rng = np.random.default_rng(79)
    value = rng.normal(size=(3, 17, 64)).astype(np.float32)
    scale = 0.01372891

    def quantize(array: np.ndarray, step: float) -> np.ndarray:
        return np.clip(np.rint(array / step), -128, 127).astype(np.int8)

    expected = quantize(value, scale)
    surrogate = bf16_quantization_surrogate(value, scale, quantize)
    bits = surrogate.view(np.uint32)
    bf16 = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1))
            & np.uint32(0xFFFF0000)).view(np.float32)

    np.testing.assert_array_equal(quantize(bf16, scale), expected)


def test_bf16_surrogate_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        bf16_quantization_surrogate(
            np.ones((2, 2), np.float32), 0.0,
            lambda value, scale: value.astype(np.int8),
        )
