"""EPU LUT channel chunk planning used by compiler-side validation."""

from __future__ import annotations


def plan_bf16_lut_chunks(channels: int, *, lut_depth: int = 256,
                         max_encoded_size: int = 4095) -> list[tuple[int, int]]:
    """Return ``(start, size)`` chunks whose BF16 LUT fits the EPU field.

    The EPU uses one 256-entry table for each group of 16 channels.  The
    encoded size is strictly less than 4096, hence at most 15 groups (240
    channels) per instruction.
    """
    if channels < 0:
        raise ValueError("channels must be non-negative")
    if channels == 0:
        return []
    max_groups = (max_encoded_size) // lut_depth
    if max_groups < 1:
        raise ValueError("LUT field cannot encode one channel group")
    max_channels = max_groups * 16
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < channels:
        size = min(max_channels, channels - start)
        chunks.append((start, size))
        start += size
    return chunks


def bf16_lut_encoded_size(channels: int, *, lut_depth: int = 256) -> int:
    """Encoded LUT size for a channel-wise BF16 EPU instruction."""
    if channels < 0:
        raise ValueError("channels must be non-negative")
    return ((max(channels, 1) - 1) // 16 + 1) * lut_depth
