"""CPU regressions for the board gate's acceptance boundary; no device access."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725"


def checker():
    path = ROOT / "tools/check_u250_native_board_gate.py"
    assert path.is_file(), "board evidence checker must exist"
    spec = importlib.util.spec_from_file_location("board_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame():
    return {
        "output_sha256": EXPECTED,
        "resident_bank_sha256": "9d01d1fd4ecae67755a4314e98a2f9d4cbe7182f3985d7573113de579b7f9577",
        "finite": True, "metrics": {"relative_l2": 0.0, "rmse": 0.0},
        "layout_codec": "native", "fallback_reasons": {},
        "vendor_pack_calls": 0, "vendor_unpack_calls": 0,
        "native_pack_calls": 1103, "native_unpack_calls": 683,
        "npu_calls": 443, "submission_groups": 248,
        "submission_group_dispatches": 443,
        "resident_bank_reused": True, "cpp_runtime_reused": True,
        "codec_yaml_reused": True, "load_ms": 0.0,
        "wall_ms": 1000.0, "process_wall_ms": 1050.0,
        "cpp_runtime": {"safe_dma": True, "stale_events": 0,
                        "physical_npu_dispatches": 443,
                        "python_submission_groups": 248},
        "latency_breakdown": {
            "resident_bank_load_ms": 0.0, "cfg_preparse_ms": 0.0,
            "cfg_vendor_activation_ms": 0.0, "input_pack_ms": 300.0,
            "output_unpack_ms": 150.0, "h2c_ms": 100.0,
            "npu_ms": 200.0, "c2h_ms": 100.0,
            "decoder_host_ops_ms": 100.0,
            "host_graph_and_python_residual_ms": 100.0,
        },
    }


def frame_v2():
    report = frame()
    report["summary_schema_version"] = 2
    report["host_profile"] = {
        "encoder.gelu": {
            "calls": 12, "ms": 60.0, "elements": 100, "bytes": 400,
        }
    }
    report["latency_breakdown"].update(
        host_profile_ms_total=60.0,
        unattributed_host_residual_ms=40.0,
    )
    report["host_executor"] = {
        "requested": "cpp", "backend": "cpp", "fallback_reason": None,
        "qualification_sha256": "1" * 64, "extension_sha256": "2" * 64,
        "host_calls": 64, "quantize_calls": 36,
        "gelu_quantize_calls": 12, "add_calls": 12,
        "add_quantize_calls": 0, "concatenate_calls": 4,
        "host_elements": 100, "host_seconds": 0.05,
    }
    report["cpp_runtime"]["extension_sha256"] = "2" * 64
    return report


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r.update(output_sha256="wrong"), "output_sha256"),
    (lambda r: r.update(resident_bank_sha256="wrong"), "resident_bank_sha256"),
    (lambda r: r["metrics"].update(relative_l2=1e-12), "relative_l2"),
    (lambda r: r["metrics"].update(rmse=float("nan")), "rmse"),
    (lambda r: r.pop("metrics"), "metrics"),
    (lambda r: r.update(finite=False), "finite"),
    (lambda r: r.update(layout_codec="auto"), "layout_codec"),
    (lambda r: r.update(vendor_pack_calls=1), "vendor_pack_calls"),
    (lambda r: r.update(vendor_unpack_calls=False), "vendor_unpack_calls"),
    (lambda r: r.update(fallback_reasons={"tensor": "missing"}), "fallback_reasons"),
    (lambda r: r.update(npu_calls=442), "npu_calls"),
    (lambda r: r.update(submission_groups=249), "submission_groups"),
    (lambda r: r.update(native_pack_calls=1102), "native_pack_calls"),
    (lambda r: r.update(resident_bank_reused=False), "resident_bank_reused"),
    (lambda r: r.update(codec_yaml_reused=False), "codec_yaml_reused"),
    (lambda r: r.update(load_ms=0.1), "load_ms"),
    (lambda r: r["cpp_runtime"].update(safe_dma=False), "safe_dma"),
    (lambda r: r["cpp_runtime"].update(stale_events=1), "stale_events"),
    (lambda r: r["cpp_runtime"].pop("stale_events"), "stale_events"),
    (lambda r: r.update(wall_ms=float("inf")), "wall_ms"),
    (lambda r: r["latency_breakdown"].pop("input_pack_ms"), "input_pack_ms"),
])
def test_rejects_inaccurate_fallback_or_incomplete_resident_evidence(mutate, reason):
    gate = checker()
    report = frame()
    mutate(report)
    with pytest.raises(AssertionError, match=reason):
        gate.check_frame("demo05_full_resident", report)


def test_accepts_exact_safe_resident_frame():
    checker().check_frame("demo05_full_resident", frame())


def test_accepts_reconciled_schema_v2_host_profile():
    checker().check_frame("demo05_full_resident", frame_v2())


def test_schema_v2_rejects_python_host_executor():
    report = frame_v2()
    report["host_executor"]["backend"] = "python"
    with pytest.raises(AssertionError, match="host executor"):
        checker().check_frame("demo05_full_resident", report)


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.pop("host_profile"), "host profile"),
    (lambda r: r["host_profile"]["encoder.gelu"].update(ms=-1.0),
     "encoder.gelu.ms"),
    (lambda r: r["latency_breakdown"].update(host_profile_ms_total=59.0),
     "host profile total"),
    (lambda r: r["latency_breakdown"].update(
        unattributed_host_residual_ms=42.0), "reconcile"),
])
def test_schema_v2_rejects_missing_or_unreconciled_host_profile(mutation, reason):
    report = frame_v2()
    mutation(report)
    with pytest.raises(AssertionError, match=reason):
        checker().check_frame("demo05_full_resident", report)


@pytest.mark.parametrize("log", ["NPU timeout status=0x0", "stale event consumed", "event poll failed"])
def test_rejects_device_errors_even_with_valid_summary(log):
    gate = checker()
    with pytest.raises(AssertionError, match="device error"):
        gate.check_log(log)


def test_success_summary_log_with_zero_stale_events_is_accepted():
    checker().check_log("HYBRID_SUMMARY=" + json.dumps(frame()))


def test_shell_passes_native_qualification_and_stops_after_failed_decoder(tmp_path):
    import os
    package = tmp_path / "package"
    package.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env python3\nimport json, pathlib, sys\n"
                          "if '--verify-package' in sys.argv:\n print('{}')\n sys.exit(0)\n"
                          "pathlib.Path(" + repr(str(tmp_path / "argv.json")) + ").write_text(json.dumps(sys.argv))\n"
                          "sys.exit(42)\n")
    fake_python.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "tools/run_u250_mapped_r58_gate.sh")],
                            env={**os.environ, "U250_NATIVE_PACKAGE": str(package),
                                 "PYTHON": str(fake_python), "U250_DEPLOYMENT_SHA256": "fixture"},
                            capture_output=True, text=True)
    assert result.returncode == 42, result.stderr
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--layout-codec") + 1] == "native"
    assert argv[argv.index("--layout-codec-report") + 1] == str(package / "artifacts/u250_native_codec/all_oracle.json")
    assert argv[argv.index("--host-executor") + 1] == "cpp"
    assert argv[argv.index("--host-executor-report") + 1] == str(
        package / "artifacts/u250_host_graph_r61/host_executor_qualification.json"
    )
    assert "--encoder-captures" in argv
    assert "--cpp-persistent-dma" not in argv


def test_python_optimization_does_not_remove_accuracy_assertions(tmp_path):
    checker()
    report = frame()
    report["metrics"]["rmse"] = 1.0
    path = tmp_path / "demo05_full_resident.summary.json"
    path.write_text(json.dumps(report))
    result = subprocess.run([sys.executable, "-O", str(ROOT / "tools/check_u250_native_board_gate.py"),
                             "--frame", str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "rmse" in result.stderr


def test_busy_board_lock_exits_75_without_package_writes(tmp_path):
    import os
    package = tmp_path / "package"
    with open("/tmp/ds-u250-runtime.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.skip("local runtime lock already occupied")
        result = subprocess.run(["bash", str(ROOT / "tools/run_u250_mapped_r58_gate.sh")],
                                env={**os.environ, "U250_NATIVE_PACKAGE": str(package)},
                                capture_output=True, text=True)
    assert result.returncode == 75
    assert "no device access" in result.stderr
    assert not package.exists()


def run_evidence(tmp_path):
    for name, counts in (("demo05_decoder_only", (89, 38)),
                         ("demo05_resume_l11", (118, 55)),
                         ("demo05_full_first", (443, 248)),
                         ("demo05_full_resident", (443, 248))):
        report = frame()
        report.update(npu_calls=counts[0], submission_groups=counts[1],
                      submission_group_dispatches=counts[0])
        report["cpp_runtime"].update(physical_npu_dispatches=counts[0], python_submission_groups=counts[1])
        (tmp_path / f"{name}.summary.json").write_text(json.dumps(report))
    for name in ("demo05_decoder_only.log", "demo05_resume_l11.log", "resident_server.stderr"):
        (tmp_path / name).write_text("")
    events = [{"ready": True, "pid": 123}]
    for name in ("demo05_full_first", "demo05_full_resident"):
        events.append({"ok": True, "code": 0, "output_sha256": EXPECTED,
                       "summary": str(tmp_path / f"{name}.summary.json")})
    events.append({"shutdown": True})
    (tmp_path / "resident_server.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    (tmp_path / "deployment_verified.json").write_text('{}')
    return events


def test_gate_cannot_accept_unverified_deployment_even_with_exact_frames(tmp_path):
    gate = checker()
    run_evidence(tmp_path)
    with pytest.raises(AssertionError, match="deployment provenance"):
        gate.check_run(tmp_path)


def test_failed_resident_request_cannot_be_hidden_by_exact_summary(tmp_path):
    gate = checker()
    events = run_evidence(tmp_path)
    events[1]["ok"] = False
    (tmp_path / "resident_server.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(AssertionError, match="request failed"):
        gate.check_run(tmp_path)


def test_retained_r60_board_evidence_is_internally_valid_but_sources_are_stale(tmp_path):
    import shutil
    gate = checker()
    root = ROOT / "artifacts/u250_native_codec"
    recorded = json.loads((root / "gate_summary.json").read_text())
    for name, report in recorded["reports"].items():
        path = root / f"{name}.summary.json"
        assert gate.digest(path) == report["summary_sha256"]
        assert json.loads(path.read_text()) == {k: v for k, v in report.items() if k != "summary_sha256"}
        shutil.copy2(path, tmp_path)
    for name, expected in recorded["log_sha256"].items():
        assert gate.digest(root / name) == expected
        shutil.copy2(root / name, tmp_path)
    shutil.copy2(root / "deployment_verified.json", tmp_path)
    verified = gate.check_run(tmp_path)
    assert verified["reports"] == recorded["reports"]
    assert verified["steady_comparisons"] == recorded["steady_comparisons"]
    assert gate.digest(root / "deployment_sha256.json") == recorded["provenance"]["inventory_sha256"]
    changed = []
    for name, expected in recorded["provenance"]["files"].items():
        if name.startswith(("tools/", "artifacts/")) and (ROOT / name).is_file():
            if gate.digest(ROOT / name) != expected:
                changed.append(name)
    assert "tools/fpga_dma_batch.cpp" in changed
    assert "tools/run_u250_depthanything_hybrid.py" in changed


def test_retained_r61_board_and_repeated_latency_evidence(tmp_path):
    import shutil

    gate = checker()
    root = ROOT / "artifacts/u250_host_graph_r61"
    for name in gate.STAGES:
        shutil.copy2(root / f"{name}.summary.json", tmp_path)
    for name in ("demo05_decoder_only.log", "demo05_resume_l11.log",
                 "resident_server.jsonl", "resident_server.stderr",
                 "deployment_verified.json"):
        shutil.copy2(root / name, tmp_path)
    verified = gate.check_run(tmp_path)
    assert verified["passed"] is True
    for report in verified["reports"].values():
        assert report["summary_schema_version"] == 2
        assert report["host_executor"]["backend"] == "cpp"
        assert report["metrics"]["relative_l2"] == 0
        assert report["metrics"]["rmse"] == 0

    repeated = json.loads(
        (root / "repeated_latency/repeated_latency.json").read_text()
    )
    assert repeated["passed"] is True
    assert repeated["safe_dma"] is True
    assert repeated["warmup_frames"] == 1
    assert repeated["measured_frames"] == 5
    assert repeated["latency"]["wall_ms"]["count"] == 5
    assert all(frame["output_sha256"] == gate.EXPECTED
               and frame["relative_l2"] == 0
               and frame["rmse"] == 0
               and frame["stale_events"] == 0
               for frame in repeated["frames"])

    output_check = json.loads((root / "r61_output_verification.json").read_text())
    assert output_check["passed"] is True
    assert output_check["expected_array_sha256"] == gate.EXPECTED
    assert all(item["exact"] is True and item["array_sha256"] == gate.EXPECTED
               for item in output_check["reports"].values())


def test_retained_r62_board_and_lut_latency_evidence(tmp_path):
    import shutil

    gate = checker()
    root = ROOT / "artifacts/u250_host_graph_r62"
    for name in gate.STAGES:
        shutil.copy2(root / f"{name}.summary.json", tmp_path)
    for name in ("demo05_decoder_only.log", "demo05_resume_l11.log",
                 "resident_server.jsonl", "resident_server.stderr",
                 "deployment_verified.json"):
        shutil.copy2(root / name, tmp_path)
    verified = gate.check_run(tmp_path)
    first = verified["reports"]["demo05_full_first"]
    resident = verified["reports"]["demo05_full_resident"]
    assert first["host_executor"]["gelu_lut_misses"] == 12
    assert first["host_executor"]["gelu_lut_hits"] == 0
    assert resident["host_executor"]["gelu_lut_misses"] == 0
    assert resident["host_executor"]["gelu_lut_hits"] == 12

    repeated = json.loads(
        (root / "repeated_latency/repeated_latency.json").read_text()
    )
    assert repeated["passed"] is True
    assert repeated["measured_frames"] == 5
    assert all(frame["lut_hits"] == 12 and frame["lut_misses"] == 0
               and frame["output_sha256"] == gate.EXPECTED
               and frame["relative_l2"] == 0 and frame["rmse"] == 0
               for frame in repeated["frames"])
    output_check = json.loads((root / "r62_output_verification.json").read_text())
    assert output_check["passed"] is True
    assert all(item["exact"] is True for item in output_check["reports"].values())
