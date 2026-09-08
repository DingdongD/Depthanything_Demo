"""Fail-closed selection and adapters for U250 host-graph execution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np


SCHEMA = "u250-host-executor-qualification-v1"
OPERATIONS = (
    "quantize", "gelu_quantize", "add", "add_quantize", "concatenate",
    "resize_align_corners",
)
PHYSICAL_FUSIONS = (
    "gelu_pack_bf16_concatenate", "attention_pack_bf16_heads",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE = Path(__file__).with_name("u250_host_graph.hpp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gelu_reference(value: np.ndarray) -> np.ndarray:
    """Match the deployed NumPy GELU approximation operation for operation."""
    value = np.asarray(value, dtype=np.float32)
    x = value / np.float32(np.sqrt(2.0))
    sign = np.sign(x)
    absolute = np.abs(x)
    factor = 1.0 / (1.0 + 0.3275911 * absolute)
    polynomial = (((((1.061405429 * factor - 1.453152027) * factor)
                    + 1.421413741) * factor - 0.284496736) * factor
                  + 0.254829592) * factor
    erf = sign * (1.0 - polynomial * np.exp(-(absolute * absolute)))
    return (0.5 * value * (1.0 + erf)).astype(np.float32)


def resize_align_corners_reference(
    value: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Match the decoder's NumPy NCHW align-corners interpolation exactly."""
    value = np.asarray(value)
    output_shape = tuple(int(item) for item in np.asarray(sizes).reshape(-1))
    if value.ndim != 4 or len(output_shape) != 4:
        raise ValueError(
            f"only NCHW Resize is supported: {value.shape}, {output_shape}"
        )
    out_h, out_w = output_shape[2:]
    in_h, in_w = value.shape[2:]
    ys = (np.linspace(0.0, in_h - 1, out_h, dtype=np.float32)
          if out_h > 1 else np.zeros(1))
    xs = (np.linspace(0.0, in_w - 1, out_w, dtype=np.float32)
          if out_w > 1 else np.zeros(1))
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).reshape(1, 1, out_h, 1)
    wx = (xs - x0).reshape(1, 1, 1, out_w)
    vertical = value[:, :, y0, :] * (1.0 - wy) + value[:, :, y1, :] * wy
    return (vertical[:, :, :, x0] * (1.0 - wx)
            + vertical[:, :, :, x1] * wx).astype(np.float32)


class PythonHostExecutor:
    backend = "python"

    def __init__(self) -> None:
        self.reset_stats()

    def _record(self, operation: str, started: float, elements: int) -> None:
        self._calls[operation] += 1
        self._elements += int(elements)
        self._seconds += time.perf_counter() - started

    def quantize(self, value: np.ndarray, scale: float) -> np.ndarray:
        started = time.perf_counter()
        result = np.clip(np.rint(np.asarray(value, np.float32) / scale),
                         -128, 127).astype(np.int8)
        self._record("quantize", started, result.size)
        return np.ascontiguousarray(result)

    def gelu_quantize(self, value: np.ndarray, scale: float) -> np.ndarray:
        started = time.perf_counter()
        activated = gelu_reference(value)
        result = np.clip(np.rint(activated / scale), -128, 127).astype(np.int8)
        self._record("gelu_quantize", started, result.size)
        return np.ascontiguousarray(result)

    def add(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        result = np.ascontiguousarray(
            np.asarray(left, np.float32) + np.asarray(right, np.float32),
            dtype=np.float32,
        )
        self._record("add", started, result.size)
        return result

    def add_quantize(
        self, left: np.ndarray, right: np.ndarray, scale: float
    ) -> np.ndarray:
        started = time.perf_counter()
        summed = np.asarray(left, np.float32) + np.asarray(right, np.float32)
        result = np.clip(np.rint(summed / scale), -128, 127).astype(np.int8)
        self._record("add_quantize", started, result.size)
        return np.ascontiguousarray(result)

    def concatenate(self, values: Iterable[np.ndarray], axis: int) -> np.ndarray:
        started = time.perf_counter()
        result = np.ascontiguousarray(np.concatenate(tuple(values), axis=axis))
        self._record("concatenate", started, result.size)
        return result

    def resize_align_corners(
        self, value: np.ndarray, sizes: np.ndarray
    ) -> np.ndarray:
        started = time.perf_counter()
        result = np.ascontiguousarray(
            resize_align_corners_reference(value, sizes), dtype=np.float32
        )
        self._record("resize_align_corners", started, result.size)
        return result

    def stats(self) -> dict[str, int | float]:
        return {
            "host_calls": sum(self._calls.values()),
            **{f"{name}_calls": self._calls[name] for name in OPERATIONS},
            "host_elements": self._elements,
            "host_seconds": self._seconds,
        }

    def reset_stats(self) -> None:
        self._calls = {name: 0 for name in OPERATIONS}
        self._elements = 0
        self._seconds = 0.0


class CppHostExecutor:
    backend = "cpp"

    def __init__(self, extension: Any) -> None:
        self.native = extension.HostGraphExecutor()

    def quantize(self, value: np.ndarray, scale: float) -> np.ndarray:
        return np.ascontiguousarray(self.native.quantize(value, scale), dtype=np.int8)

    def gelu_quantize(self, value: np.ndarray, scale: float) -> np.ndarray:
        return np.ascontiguousarray(
            self.native.gelu_quantize(value, scale), dtype=np.int8
        )

    def gelu_pack_bf16_concatenate(
        self, physical_inputs: Iterable[tuple[np.ndarray, np.ndarray]],
        source_descriptors: Iterable[Any], target_descriptor: Any, scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = self.native.gelu_pack_bf16_concatenate(
            list(physical_inputs),
            [asdict(descriptor) for descriptor in source_descriptors],
            asdict(target_descriptor), scale,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("native physical GELU fusion returned an invalid bank pair")
        return tuple(np.ascontiguousarray(bank, dtype=np.uint8)
                     for bank in result)  # type: ignore[return-value]

    def attention_pack_bf16_heads(
        self, physical_inputs: Iterable[tuple[np.ndarray, np.ndarray]],
        source_descriptors: Iterable[Any], valid_widths: Iterable[int],
        target_descriptor: Any, scale: float, heads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = self.native.attention_pack_bf16_heads(
            list(physical_inputs),
            [asdict(descriptor) for descriptor in source_descriptors],
            list(valid_widths), asdict(target_descriptor), scale, heads,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("native physical attention fusion returned invalid banks")
        return tuple(np.ascontiguousarray(bank, dtype=np.uint8)
                     for bank in result)  # type: ignore[return-value]

    def add(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.native.add(left, right), dtype=np.float32)

    def add_quantize(
        self, left: np.ndarray, right: np.ndarray, scale: float
    ) -> np.ndarray:
        return np.ascontiguousarray(
            self.native.add_quantize(left, right, scale), dtype=np.int8
        )

    def concatenate(self, values: Iterable[np.ndarray], axis: int) -> np.ndarray:
        return np.ascontiguousarray(self.native.concatenate(list(values), axis))

    def resize_align_corners(
        self, value: np.ndarray, sizes: np.ndarray
    ) -> np.ndarray:
        output_shape = tuple(int(item) for item in np.asarray(sizes).reshape(-1))
        if np.asarray(value).ndim != 4 or len(output_shape) != 4:
            raise ValueError(
                f"only NCHW Resize is supported: "
                f"{np.asarray(value).shape}, {output_shape}"
            )
        return np.ascontiguousarray(
            self.native.resize_align_corners(
                value, output_shape[2], output_shape[3]
            ),
            dtype=np.float32,
        )

    def stats(self) -> dict:
        return dict(self.native.stats())

    def reset_stats(self) -> None:
        self.native.reset_stats()


class HostExecutorSelection:
    """Validate qualification evidence before a native host API can be used."""

    def __init__(self, mode: str, report: Path | None,
                 extension_path: Path | None, extension_sha256: str | None):
        if mode not in {"python", "auto", "cpp"}:
            raise ValueError(f"invalid host executor mode: {mode}")
        self.mode = mode
        self.extension_path = (Path(extension_path).resolve()
                               if extension_path is not None else None)
        self.extension_sha256 = extension_sha256
        self.report_path = Path(report).resolve() if report is not None else None
        self.report_sha256: str | None = None
        self.fallback_reason: str | None = None
        self.backend = "python"
        if mode == "python":
            return
        error = self._validate_report()
        if error is not None:
            if mode == "cpp":
                raise RuntimeError(f"qualification report: {error}")
            self.fallback_reason = error
            return
        self.backend = "cpp"

    def _validate_report(self) -> str | None:
        if self.report_path is None:
            return "qualification report is required"
        try:
            raw = self.report_path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError) as error:
            return f"cannot read qualification report: {error}"
        self.report_sha256 = hashlib.sha256(raw).hexdigest()
        if not isinstance(value, dict):
            return "invalid qualification report object"
        if value.get("schema") != SCHEMA:
            return "unknown qualification schema"
        if value.get("qualified") is not True:
            return "report is not qualified"
        source_digest = value.get("source_sha256")
        if (not isinstance(source_digest, str)
                or _SHA256.fullmatch(source_digest) is None
                or source_digest != _sha256_file(_SOURCE)):
            return "source SHA-256 mismatch"
        if self.extension_path is None:
            return "extension path is unavailable"
        try:
            report_path = Path(value.get("extension_path", "")).resolve()
        except (OSError, TypeError, ValueError):
            return "extension path is invalid"
        if report_path != self.extension_path:
            return "extension path mismatch"
        if (not isinstance(self.extension_sha256, str)
                or _SHA256.fullmatch(self.extension_sha256) is None
                or value.get("extension_sha256") != self.extension_sha256):
            return "extension SHA-256 mismatch"
        operations = value.get("operations")
        if not isinstance(operations, dict):
            return "operations are missing"
        for name in OPERATIONS:
            item = operations.get(name)
            if not isinstance(item, dict):
                return f"operation {name} is missing"
            if item.get("exact") is not True:
                return f"operation {name} is not exact"
            if type(item.get("cases")) is not int or item["cases"] <= 0:
                return f"operation {name} has invalid case count"
        if set(operations) != set(OPERATIONS):
            return "qualification report has unknown operations"
        physical_fusions = value.get("physical_fusions")
        if not isinstance(physical_fusions, dict):
            return "physical fusions are missing"
        for name in PHYSICAL_FUSIONS:
            item = physical_fusions.get(name)
            if not isinstance(item, dict):
                return f"physical fusion {name} is missing"
            if item.get("exact") is not True:
                return f"physical fusion {name} is not exact"
            if type(item.get("cases")) is not int or item["cases"] <= 0:
                return f"physical fusion {name} has invalid case count"
        if set(physical_fusions) != set(PHYSICAL_FUSIONS):
            return "qualification report has unknown physical fusions"
        return None

    def create(self, extension: Any) -> PythonHostExecutor | CppHostExecutor:
        if self.backend == "python":
            return PythonHostExecutor()
        loaded_path = getattr(extension, "__file__", None)
        if loaded_path is None or Path(loaded_path).resolve() != self.extension_path:
            raise RuntimeError("loaded extension path does not match qualification")
        if getattr(extension, "_u250_extension_sha256", None) != self.extension_sha256:
            raise RuntimeError("loaded extension SHA-256 does not match qualification")
        if not callable(getattr(extension, "HostGraphExecutor", None)):
            raise RuntimeError("loaded extension has no HostGraphExecutor API")
        return CppHostExecutor(extension)
