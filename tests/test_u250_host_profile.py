from __future__ import annotations

import pytest

from tools.u250_host_profile import HostProfiler


def test_profiler_aggregates_calls_work_and_time(monkeypatch):
    """Catch lost calls/bytes or a timer that overwrites rather than accumulates."""
    ticks = iter((10.000, 10.002, 20.000, 20.003))
    monkeypatch.setattr(
        "tools.u250_host_profile.time.perf_counter", lambda: next(ticks)
    )
    profile = HostProfiler()
    with profile.measure("encoder.quantize", elements=4, nbytes=16):
        pass
    with profile.measure("encoder.quantize", elements=6, nbytes=24):
        pass

    result = profile.operations["encoder.quantize"]
    assert result["calls"] == 2
    assert result["ms"] == pytest.approx(5.0)
    assert result["elements"] == 10
    assert result["bytes"] == 40


def test_summary_reconciles_process_wall_without_hiding_negative_time():
    """Catch clamping that would conceal double-counted measurement scopes."""
    profile = HostProfiler()
    profile.record("encoder.gelu", 25.0, elements=100, nbytes=400)
    result = profile.summary(
        process_wall_ms=100.0, externally_accounted_ms=60.0
    )
    assert result["host_profile_ms_total"] == 25.0
    assert result["unattributed_host_residual_ms"] == 15.0

    with pytest.raises(ValueError, match="exceeds process wall"):
        profile.summary(process_wall_ms=50.0, externally_accounted_ms=40.0)


@pytest.mark.parametrize(
    "name,elapsed,elements,nbytes",
    [
        ("", 1.0, 1, 4),
        ("encoder.gelu", -1.0, 1, 4),
        ("encoder.gelu", float("nan"), 1, 4),
        ("encoder.gelu", 1.0, -1, 4),
        ("encoder.gelu", 1.0, 1, -4),
    ],
)
def test_profiler_rejects_invalid_measurements(name, elapsed, elements, nbytes):
    """Catch malformed measurements before they can enter board evidence."""
    with pytest.raises(ValueError, match="invalid host timing"):
        HostProfiler().record(
            name, elapsed, elements=elements, nbytes=nbytes
        )
