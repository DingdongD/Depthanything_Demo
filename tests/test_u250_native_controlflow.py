"""Control-flow evidence must fail closed before a board gate can run."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def controlflow():
    path = ROOT / "tools/run_u250_native_codec_controlflow.py"
    assert path.is_file(), "native CPU control-flow gate is missing"
    spec = importlib.util.spec_from_file_location("native_controlflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_summary():
    # Reuse authentic remote input metadata; bind this test fixture to current
    # source bytes independently of the gate's hashing implementation.
    provenance = json.loads((ROOT / "artifacts/u250_native_codec/"
                            "full_controlflow.summary.json").read_text())["provenance"]
    provenance["source_sha256"] = {
        name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest()
        for name in ("run_u250_native_codec_controlflow.py", "run_u250_depthanything_hybrid.py",
                     "u250_cpp_mapped_runtime.py", "u250_layout_descriptors.py", "fpga_dma_batch.cpp")
    }
    return {
        "provenance": provenance,
        "output_sha256": "b8db8d36a400c7fa0bc135cb6eb992c5cd350dac7e1657fa9029dee343af463b",
        "npu_calls": 443, "submission_groups": 248,
        "submission_group_dispatches": 443,
        "native_pack_calls": 1103, "native_unpack_calls": 683,
        "vendor_pack_calls": 0, "vendor_unpack_calls": 0,
        "layout_codec": "native", "finite": True,
        "native_pack_ms": 110., "native_unpack_ms": 68.,
        "codec_pack_ms_total": 110., "codec_unpack_ms_total": 68.,
        "fallback_reasons": {}, "c2h_exact_half_size": True,
        "codec_by_layout_dtype": {"NDWC_INT8": {
            "native_pack_calls": 1103, "native_unpack_calls": 683,
            "native_pack_ms": 110., "native_unpack_ms": 68.,
        }},
        "cpu_controlflow": {
            "fake_transport": True, "device_open_attempts": 0,
            "runtime_lock_open_attempts": 0, "open_trace_checked": True,
            "native_api_calls": {"pack": 1103, "unpack": 683},
            "transport_dispatches": 443, "transport_groups": 248,
            "traced_open_calls": 100, "open_trace_sha256": "1" * 64,
            "open_trace_fd_decoding": True, "protected_writes_checked": True,
            "protected_write_open_attempts": 0,
        },
    }


def test_recorded_r58_is_rejected_for_missing_native_counters():
    gate = controlflow().assert_native_controlflow
    summary = json.loads((ROOT / "artifacts/u250_mapped_runtime_r58/"
                          "full_controlflow_fake_transport.summary.json").read_text())
    with pytest.raises(AssertionError, match="native_pack_calls"):
        gate(summary)


@pytest.mark.parametrize("field,value", [
    ("npu_calls", 442), ("submission_groups", 247),
    ("submission_group_dispatches", 442),
    ("native_pack_calls", 1102), ("native_unpack_calls", 682),
    ("vendor_pack_calls", 1), ("vendor_unpack_calls", 1),
    ("native_pack_ms", float("nan")), ("native_unpack_ms", -1.),
    ("native_pack_ms", 12841.621), ("native_unpack_ms", 0.),
    ("finite", False), ("layout_codec", "auto"),
    ("fallback_reasons", {"tensor": "unsupported"}),
    ("codec_by_layout_dtype", {}), ("c2h_exact_half_size", False),
    ("codec_pack_ms_total", 109.),
])
def test_gate_rejects_incomplete_or_inconsistent_evidence(field, value):
    gate = controlflow().assert_native_controlflow
    summary = qualified_summary()
    summary[field] = value
    with pytest.raises(AssertionError):
        gate(summary)


@pytest.mark.parametrize("field,value", [
    ("device_open_attempts", 1), ("runtime_lock_open_attempts", 1),
    ("open_trace_checked", False), ("fake_transport", False),
    ("native_api_calls", {"pack": 0, "unpack": 683}),
    ("transport_dispatches", 442), ("transport_groups", 247),
])
def test_gate_requires_independent_transport_codec_and_device_evidence(field, value):
    gate = controlflow().assert_native_controlflow
    summary = qualified_summary()
    summary["cpu_controlflow"][field] = value
    with pytest.raises(AssertionError):
        gate(summary)


def test_gate_accepts_complete_evidence_and_strictly_rejects_timing_boundary():
    summary = qualified_summary()
    controlflow().assert_native_controlflow(summary)
    summary["native_pack_ms"] = 12841.621 - 68.
    summary["codec_pack_ms_total"] = summary["native_pack_ms"]
    summary["codec_by_layout_dtype"]["NDWC_INT8"]["native_pack_ms"] = summary["native_pack_ms"]
    with pytest.raises(AssertionError, match="12841.621"):
        controlflow().assert_native_controlflow(summary)


def test_shell_cpu_check_exits_before_board_work(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(qualified_summary()))
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_u250_mapped_r58_gate.sh"),
         "--check-native-controlflow", str(path)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "native CPU control-flow gate passed" in result.stdout


def test_open_trace_detects_devices_and_runtime_lock(tmp_path):
    trace = tmp_path / "opens.log"
    trace.write_text('42 openat(AT_FDCWD, "/dev/null", O_WRONLY) = 3\n'
                     '42 openat(AT_FDCWD, "/tmp/input.npy", O_RDONLY) = 4\n')
    assert controlflow().inspect_open_trace(trace)["device_open_attempts"] == 0
    trace.write_text('42 openat(AT_FDCWD, "/dev/xdma0_user", O_RDWR) = -1 EACCES\n'
                     '42 openat(AT_FDCWD, "/tmp/ds-u250-runtime.lock", O_RDWR) = 4\n')
    result = controlflow().inspect_open_trace(trace)
    assert result["device_open_attempts"] == 1
    assert result["runtime_lock_open_attempts"] == 1


@pytest.mark.parametrize("contents", ["", "not an open trace\n"])
def test_empty_or_unrelated_trace_cannot_prove_no_device_access(tmp_path, contents):
    module = controlflow()
    path = tmp_path / "trace.log"
    path.write_text(contents)
    with pytest.raises(AssertionError):
        module.inspect_open_trace(path)


@pytest.mark.parametrize("line,field", [
    ('openat(3</dev>, "xdma0_user", O_RDWR) = -1 EACCES', "device_open_attempts"),
    ('openat(3</tmp>, "ds-u250-runtime.lock", O_RDWR) = -1 EACCES', "runtime_lock_open_attempts"),
    ('openat(3, "xdma0_user", O_RDWR) = -1 EACCES', "device_open_attempts"),
    ('openat(3, "ds-u250-runtime.lock", O_RDWR) = -1 EACCES', "runtime_lock_open_attempts"),
    ('openat(3, "uio0", O_RDWR) = -1 EACCES', "device_open_attempts"),
])
def test_dirfd_or_unresolved_suspicious_open_cannot_bypass_trace_gate(tmp_path, line, field):
    path = tmp_path / "trace.log"
    path.write_text("42 " + line + "\n")
    assert controlflow().inspect_open_trace(path)[field] == 1


def test_legitimate_relative_opens_remain_accepted(tmp_path):
    path = tmp_path / "trace.log"
    path.write_text('42 openat(3</tmp>, "weights.bin", O_RDONLY) = 4</tmp/weights.bin>\n'
                    '42 openat(3, "config.json", O_RDONLY) = -1 ENOENT\n')
    result = controlflow().inspect_open_trace(path)
    assert result["device_open_attempts"] == result["runtime_lock_open_attempts"] == 0


def test_trace_detects_protected_package_writes_for_absolute_and_dirfd_paths(tmp_path):
    path = tmp_path / "trace.log"
    path.write_text('42 openat(3</opt/package>, "cfg.txt", O_WRONLY|O_TRUNC) = 4\n'
                    '42 openat(AT_FDCWD, "/opt/package/params.npz", O_RDWR) = 4\n'
                    '42 openat(3</opt/package>, "input.npy", O_RDONLY) = 4\n'
                    '42 openat(3</tmp>, "output.json", O_CREAT|O_WRONLY, 0600) = 4\n')
    result = controlflow().inspect_open_trace(path, protected_dirs=(Path("/opt/package"),))
    assert result["protected_writes_checked"] is True
    assert result["protected_write_open_attempts"] == 2


@pytest.mark.parametrize("mutation", [
    lambda s: s.pop("provenance"),
    lambda s: s.update(output_sha256="0" * 64),
    lambda s: s["cpu_controlflow"].update(traced_open_calls=0),
    lambda s: s["cpu_controlflow"].pop("open_trace_sha256"),
    lambda s: s["cpu_controlflow"].update(open_trace_sha256=""),
    lambda s: s["cpu_controlflow"].update(open_trace_fd_decoding=False),
    lambda s: s["cpu_controlflow"].update(protected_write_open_attempts=1),
    lambda s: s["provenance"].update(source_sha256={}),
    lambda s: s["provenance"]["source_sha256"].update(fpga_dma_batch_cpp="0" * 64),
    lambda s: s["provenance"]["source_sha256"].update({"u250_cpp_mapped_runtime.py": "0" * 64}),
    lambda s: s["provenance"].pop("extension_sha256"),
    lambda s: s["provenance"].update(extension_sha256="invalid"),
    lambda s: s["provenance"].update(extension_sha256="0" * 64),
    lambda s: s["provenance"].update(input_sha256={}),
    lambda s: s["provenance"]["input_sha256"].update({"layout-codec-report": "0" * 64}),
    lambda s: s["provenance"].update(cfg_sha256={}),
    lambda s: s["provenance"]["cfg_sha256"].pop(next(iter(s["provenance"]["cfg_sha256"]))),
])
def test_gate_rejects_tampered_provenance_or_equivalence(mutation):
    module = controlflow()
    summary = qualified_summary()
    mutation(summary)
    with pytest.raises(AssertionError):
        module.assert_native_controlflow(summary)


def test_cli_rejects_deleted_provenance_under_python_optimization(tmp_path):
    summary = qualified_summary()
    summary.pop("provenance")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    result = subprocess.run(["python", "-O", str(ROOT / "tools/run_u250_native_codec_controlflow.py"),
                             "--check-summary", str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "provenance" in result.stderr


def test_committed_summary_qualifies():
    path = ROOT / "artifacts/u250_native_codec/full_controlflow.summary.json"
    assert path.is_file(), "authentic native full control-flow evidence is missing"
    module = controlflow()
    summary = json.loads(path.read_text())
    module.assert_native_controlflow(summary)
    assert len(summary["provenance"]["cfg_sha256"]) == 262
    for name, expected in summary["provenance"]["source_sha256"].items():
        assert module.sha256_file(ROOT / "tools" / name) == expected
    assert summary["cpp_runtime"]["mapped_bar"] is False
    assert summary["cpp_runtime"]["locked_host_buffers"] is False
