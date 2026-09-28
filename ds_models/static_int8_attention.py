"""Static-INT8 attention contract shared by calibration and U250 runtime.

The U250 CTC MatMul is reliable only when the left/query row count is at most
256.  Depth Anything V2-S has 1370 tokens, so every attention head is executed
as five full 256-row calls and one 90-row call padded to 256 rows.  K and V
always retain all 1370 tokens, which makes each slice mathematically independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class QueryChunk:
    index: int
    start: int
    stop: int
    padded_rows: int

    @property
    def valid_rows(self) -> int:
        return self.stop - self.start


def plan_query_chunks(tokens: int, rows_per_chunk: int = 256) -> tuple[QueryChunk, ...]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if rows_per_chunk <= 0 or rows_per_chunk > 256:
        raise ValueError("U250 query chunks must contain 1..256 rows")
    return tuple(
        QueryChunk(index, start, min(start + rows_per_chunk, tokens), rows_per_chunk)
        for index, start in enumerate(range(0, tokens, rows_per_chunk))
    )


def bf16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def quantize_symmetric(value: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(bf16_round(value) / scale).clamp(-127, 127).to(torch.int8)


def quantize_probability(value: torch.Tensor, scale: float) -> torch.Tensor:
    # SPU output uses the non-negative half of signed INT8.
    return torch.floor(value / scale).clamp(0, 127).to(torch.int8)


@dataclass(frozen=True)
class HeadScale:
    q: float
    k: float
    v: float
    probability: float
    av_output_gain: float = 1.0
    av_value_accumulator_scale: float | None = None

    @property
    def effective_av_value_scale(self) -> float:
        if self.av_value_accumulator_scale is not None:
            return self.av_value_accumulator_scale
        return self.v * self.av_output_gain


class StaticAttentionProfile:
    """Validated access to the calibration JSON consumed by runtime/codegen."""

    def __init__(self, document: dict):
        if document.get("schema_version") != 2:
            raise ValueError("expected static attention profile schema_version=2")
        self.document = document
        self.tokens = int(document["model"]["tokens"])
        self.heads = int(document["model"]["heads"])
        self.head_dimension = int(document["model"]["head_dimension"])
        self.layers = document["layers"]
        if len(self.layers) != int(document["model"]["attention_layers"]):
            raise ValueError("profile layer count does not match model metadata")

    @classmethod
    def load(cls, path: str | Path) -> "StaticAttentionProfile":
        return cls(json.loads(Path(path).read_text()))

    def scale(self, layer: int, head: int) -> HeadScale:
        values = self.layers[str(layer)]["heads"][head]["selected_scales_bf16"]
        return HeadScale(
            q=float(values["q"]),
            k=float(values["k"]),
            v=float(values["v"]),
            probability=float(values["probability"]),
            av_output_gain=float(values.get("av_output_gain", 1.0)),
            av_value_accumulator_scale=(
                float(values["av_value_accumulator_scale"])
                if "av_value_accumulator_scale" in values else None
            ),
        )


# A hardware runner receives already-quantized physical values.  It must return
# BF16-dequantized output with shape [1, padded_rows, head_dimension].
HardwareRunner = Callable[
    [int, int, QueryChunk, np.ndarray, np.ndarray, np.ndarray, HeadScale],
    np.ndarray,
]


class StaticInt8AttentionRuntime:
    """Execute calibrated attention using either a surrogate or a U250 callback."""

    def __init__(
        self,
        profile: StaticAttentionProfile,
        runner: HardwareRunner | None = None,
        rows_per_chunk: int = 256,
    ) -> None:
        self.profile = profile
        self.runner = runner
        self.chunks = plan_query_chunks(profile.tokens, rows_per_chunk)

    def execute(self, layer: int, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """Run tensors shaped [1, heads, tokens, head_dimension]."""

        expected = (1, self.profile.heads, self.profile.tokens,
                    self.profile.head_dimension)
        if tuple(q.shape) != expected or tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"Q/K/V must all have shape {expected}")
        per_head = []
        for head in range(self.profile.heads):
            scale = self.profile.scale(layer, head)
            qi = quantize_symmetric(q[:, head], scale.q)
            ki = quantize_symmetric(k[:, head], scale.k)
            vi = quantize_symmetric(v[:, head], scale.v)
            kt = ki.transpose(-2, -1).contiguous()
            outputs = []
            for chunk in self.chunks:
                query = torch.zeros(
                    1, chunk.padded_rows, self.profile.head_dimension,
                    dtype=torch.int8, device=q.device,
                )
                query[:, :chunk.valid_rows] = qi[:, chunk.start:chunk.stop]
                if self.runner is None:
                    output = self._surrogate(query, kt, vi, scale)
                else:
                    output_np = self.runner(
                        layer,
                        head,
                        chunk,
                        query.cpu().numpy(),
                        kt.cpu().numpy(),
                        vi.cpu().numpy(),
                        scale,
                    )
                    expected_output = (1, chunk.padded_rows, self.profile.head_dimension)
                    if tuple(output_np.shape) != expected_output:
                        raise ValueError(
                            f"hardware runner returned {output_np.shape}, expected {expected_output}"
                        )
                    output = torch.from_numpy(np.asarray(output_np, dtype=np.float32)).to(q.device)
                outputs.append(output[:, :chunk.valid_rows])
            per_head.append(torch.cat(outputs, dim=1))
        return torch.stack(per_head, dim=1)

    @staticmethod
    def _surrogate(query: torch.Tensor, key_transposed: torch.Tensor,
                   value: torch.Tensor, scale: HeadScale) -> torch.Tensor:
        logits_i32 = torch.matmul(query.to(torch.int32), key_transposed.to(torch.int32))
        logits = bf16_round(logits_i32.float() * (scale.q * scale.k))
        probability = bf16_round(torch.softmax(logits, dim=-1))
        probability_i8 = quantize_probability(probability, scale.probability)
        output_i32 = torch.matmul(probability_i8.to(torch.int32), value.to(torch.int32))
        return bf16_round(
            output_i32.float() * (scale.probability * scale.effective_av_value_scale)
        )


def chunked_float_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            rows_per_chunk: int = 256) -> torch.Tensor:
    """Exact floating reference with the same query slicing as U250 runtime."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Q/K/V must have identical [B,H,N,D] shapes")
    key_transposed = k.transpose(-2, -1)
    return torch.cat(
        [torch.softmax(q[:, :, c.start:c.stop] @ key_transposed, dim=-1) @ v
         for c in plan_query_chunks(q.shape[-2], rows_per_chunk)],
        dim=-2,
    )
