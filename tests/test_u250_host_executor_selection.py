from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.u250_host_executor import HostExecutorSelection, PythonHostExecutor
from tools.qualify_u250_host_executor import qualify_host_executor


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("quantize", "gelu_quantize", "add", "add_quantize", "concatenate")


@pytest.fixture
def extension_identity(tmp_path):
    path = tmp_path / "fpgaDmaBatch.so"
    path.write_bytes(b"qualified host executor extension")
    return path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def report(extension_identity):
    path, digest = extension_identity
    return {
        "schema": "u250-host-executor-qualification-v1",
        "qualified": True,
        "source_sha256": hashlib.sha256(
            (ROOT / "tools/u250_host_graph.hpp").read_bytes()
        ).hexdigest(),
        "extension_path": str(path),
        "extension_sha256": digest,
        "operations": {name: {"exact": True, "cases": 2} for name in OPERATIONS},
    }


@pytest.fixture
def report_path(tmp_path, report):
    path = tmp_path / "host_executor.json"
    path.write_text(json.dumps(report))
    return path


def test_python_mode_requires_neither_extension_nor_report():
    selection = HostExecutorSelection("python", None, None, None)
    assert selection.backend == "python"
    assert selection.fallback_reason is None
    assert isinstance(selection.create(None), PythonHostExecutor)


def test_cpp_mode_requires_matching_complete_report(extension_identity):
    with pytest.raises(RuntimeError, match="qualification report"):
        HostExecutorSelection("cpp", None, *extension_identity)


def test_auto_falls_back_with_reason_for_digest_mismatch(
    tmp_path, report, extension_identity
):
    report["extension_sha256"] = "0" * 64
    path = tmp_path / "host_executor.json"
    path.write_text(json.dumps(report))
    selection = HostExecutorSelection("auto", path, *extension_identity)
    assert selection.backend == "python"
    assert selection.fallback_reason == "extension SHA-256 mismatch"


def test_cpp_accepts_only_all_exact_cases(report_path, extension_identity):
    selection = HostExecutorSelection("cpp", report_path, *extension_identity)
    assert selection.backend == "cpp"
    assert selection.fallback_reason is None
    assert len(selection.report_sha256) == 64


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda value: value.update(schema="unknown"), "schema"),
        (lambda value: value.update(qualified=False), "qualified"),
        (lambda value: value.update(source_sha256="0" * 64), "source SHA-256"),
        (lambda value: value.update(extension_path="/tmp/replaced.so"), "extension path"),
        (lambda value: value["operations"].pop("add"), "operation add"),
        (lambda value: value["operations"]["quantize"].update(exact=False), "not exact"),
        (lambda value: value["operations"]["concatenate"].update(cases=0), "case count"),
    ],
)
def test_cpp_rejects_mutated_qualification(
    tmp_path, report, extension_identity, mutation, reason
):
    mutation(report)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match=reason):
        HostExecutorSelection("cpp", path, *extension_identity)


class FakeNativeExecutor:
    def quantize(self, value, scale):
        return np.zeros(np.asarray(value).shape, dtype=np.int8)

    gelu_quantize = quantize

    def add(self, left, right):
        return np.asarray(left, np.float32) + np.asarray(right, np.float32)

    def add_quantize(self, left, right, scale):
        return np.zeros(np.asarray(left).shape, dtype=np.int8)

    def concatenate(self, values, axis):
        return np.concatenate(values, axis=axis)

    def stats(self):
        return {"host_calls": 0}

    def reset_stats(self):
        pass


def test_create_rejects_replaced_loaded_module(report_path, extension_identity):
    path, digest = extension_identity
    selection = HostExecutorSelection("cpp", report_path, path, digest)
    extension = SimpleNamespace(
        __file__=str(path.with_name("other.so")),
        _u250_extension_sha256=digest,
        HostGraphExecutor=FakeNativeExecutor,
    )
    with pytest.raises(RuntimeError, match="loaded extension path"):
        selection.create(extension)


def test_create_cpp_adapter_only_for_bound_loaded_module(
    report_path, extension_identity
):
    path, digest = extension_identity
    selection = HostExecutorSelection("cpp", report_path, path, digest)
    extension = SimpleNamespace(
        __file__=str(path),
        _u250_extension_sha256=digest,
        HostGraphExecutor=FakeNativeExecutor,
    )
    executor = selection.create(extension)
    assert executor.backend == "cpp"
    assert executor.stats()["host_calls"] == 0


def test_python_adapter_matches_runtime_boundaries():
    executor = PythonHostExecutor()
    values = np.linspace(-8, 8, 1025, dtype=np.float32)
    quantized = executor.gelu_quantize(values, 0.03993530943989754)
    assert quantized.dtype == np.int8
    assert quantized.flags.c_contiguous
    assert executor.stats()["gelu_quantize_calls"] == 1
    executor.reset_stats()
    assert executor.stats()["host_calls"] == 0


class ExactNativeExecutor:
    def __init__(self):
        self.reference = PythonHostExecutor()

    def quantize(self, value, scale):
        return self.reference.quantize(value, scale)

    def gelu_quantize(self, value, scale):
        return self.reference.gelu_quantize(value, scale)

    def add(self, left, right):
        return self.reference.add(left, right)

    def add_quantize(self, left, right, scale):
        return self.reference.add_quantize(left, right, scale)

    def concatenate(self, values, axis):
        return self.reference.concatenate(values, axis)


def test_qualifier_records_exact_deterministic_and_trace_cases(
    tmp_path, extension_identity
):
    extension_path, extension_sha256 = extension_identity
    trace_path = tmp_path / "demo05_holdout_trace_r52.npz"
    np.savez(trace_path, hidden=np.linspace(-4, 4, 96, dtype=np.float32).reshape(2, 3, 16))
    extension = SimpleNamespace(HostGraphExecutor=ExactNativeExecutor)
    result = qualify_host_executor(extension, extension_path, trace_path)
    assert result["qualified"] is True
    assert result["extension_sha256"] == extension_sha256
    assert result["trace_sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert all(result["operations"][name]["cases"] >= 2 for name in OPERATIONS)
    assert all(result["operations"][name]["exact"] for name in OPERATIONS)


def test_qualifier_cannot_mark_changed_native_result_exact(
    tmp_path, extension_identity
):
    class ChangedNativeExecutor(ExactNativeExecutor):
        def quantize(self, value, scale):
            result = super().quantize(value, scale)
            result.flat[0] = np.int8(int(result.flat[0]) + 1)
            return result

    extension_path, _ = extension_identity
    result = qualify_host_executor(
        SimpleNamespace(HostGraphExecutor=ChangedNativeExecutor),
        extension_path,
        None,
    )
    assert result["qualified"] is False
    assert result["operations"]["quantize"]["exact"] is False
