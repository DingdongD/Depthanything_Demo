"""Fine-grained, reconciled timing for DepthAnything host graph work."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Iterator


_TOLERANCE_MS = 0.001


@dataclass
class HostProfiler:
    """Accumulate named host intervals without overlapping external timers."""

    operations: dict[str, dict[str, int | float]] = field(default_factory=dict)

    @contextmanager
    def measure(
        self, name: str, *, elements: int = 0, nbytes: int = 0
    ) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name,
                (time.perf_counter() - started) * 1000.0,
                elements=elements,
                nbytes=nbytes,
            )

    def record(
        self,
        name: str,
        elapsed_ms: float,
        *,
        elements: int = 0,
        nbytes: int = 0,
    ) -> None:
        if (
            not isinstance(name, str)
            or not name
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
            or type(elements) is not int
            or type(nbytes) is not int
            or elements < 0
            or nbytes < 0
        ):
            raise ValueError("invalid host timing")
        item = self.operations.setdefault(
            name, {"calls": 0, "ms": 0.0, "elements": 0, "bytes": 0}
        )
        item["calls"] += 1
        item["ms"] += float(elapsed_ms)
        item["elements"] += elements
        item["bytes"] += nbytes

    def summary(
        self, *, process_wall_ms: float, externally_accounted_ms: float
    ) -> dict:
        if (
            not math.isfinite(process_wall_ms)
            or not math.isfinite(externally_accounted_ms)
            or process_wall_ms < 0
            or externally_accounted_ms < 0
        ):
            raise ValueError("invalid host timing totals")
        host_total = sum(float(item["ms"]) for item in self.operations.values())
        residual = float(process_wall_ms) - float(externally_accounted_ms) - host_total
        if residual < -_TOLERANCE_MS:
            raise ValueError(
                "explicit timing exceeds process wall by "
                f"{-residual:.6f} ms"
            )
        return {
            "host_profile": {
                name: dict(self.operations[name]) for name in sorted(self.operations)
            },
            "host_profile_ms_total": host_total,
            "unattributed_host_residual_ms": max(0.0, residual),
        }
