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
    inventory = json.loads((ROOT / "artifacts/u250_native_codec/controlflow_input_inventory.json").read_text())
    provenance["runtime_input_sha256"] = {
        role: entry["sha256"] for role, entry in inventory["runtime_inputs"].items()
    }
    return {
        "provenance": provenance,
        "output_sha256": "b8db8d36a400c7fa0bc135cb6eb992c5cd350dac7e1657fa9029dee343af463b",
        "resident_bank_sha256": inventory["resident_bank"]["sha256"],
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


def qualified_r61_host_summary():
    summary = qualified_summary()
    summary["summary_schema_version"] = 2
    summary["process_wall_ms"] = 1000.0
    summary["host_profile"] = {
        "encoder.gelu_quantize": {
            "calls": 12, "ms": 60.0, "elements": 100, "bytes": 400,
        }
    }
    summary["latency_breakdown"] = {
        "resident_bank_load_ms": 0.0, "cfg_preparse_ms": 0.0,
        "cfg_vendor_activation_ms": 0.0, "input_pack_ms": 110.0,
        "output_unpack_ms": 68.0, "h2c_ms": 100.0, "npu_ms": 200.0,
        "c2h_ms": 100.0, "decoder_host_ops_ms": 100.0,
        "host_profile_ms_total": 60.0,
        "unattributed_host_residual_ms": 262.0,
        "host_graph_and_python_residual_ms": 322.0,
    }
    summary["host_executor"] = {
        "requested": "cpp", "backend": "cpp", "fallback_reason": None,
        "qualification_sha256": "1" * 64, "extension_sha256": "2" * 64,
        "host_calls": 60, "quantize_calls": 36,
        "gelu_quantize_calls": 12, "add_calls": 12,
        "add_quantize_calls": 0, "concatenate_calls": 4,
        "host_elements": 100, "host_seconds": 0.05,
    }
    return summary


def test_schema_v3_host_execution_requires_five_native_resize_calls():
    summary = qualified_r61_host_summary()
    summary["summary_schema_version"] = 3
    summary["host_executor"]["resize_align_corners_calls"] = 5
    summary["host_executor"]["host_calls"] = 69
    controlflow().validate_host_execution(summary)
    summary["host_executor"]["resize_align_corners_calls"] = 4
    with pytest.raises(AssertionError, match="5 align-corners Resize"):
        controlflow().validate_host_execution(summary)


def test_schema_v4_controlflow_accepts_only_exact_pack_reuse_counts():
    summary = qualified_r61_host_summary()
    summary["summary_schema_version"] = 4
    summary.update(
        native_pack_calls=721,
        native_pack_cache_hits=382,
        native_pack_cache_logical_bytes_saved=69055500,
        native_pack_cache_physical_bytes_saved=72978432,
    )
    summary["cpu_controlflow"]["native_api_calls"]["pack"] = 721
    summary["codec_by_layout_dtype"]["NDWC_INT8"]["native_pack_calls"] = 721
    summary["host_executor"]["resize_align_corners_calls"] = 5
    summary["host_executor"]["host_calls"] = 69
    # Provenance is intentionally out of scope for this counter boundary.
    original = controlflow().validate_provenance
    module = controlflow()
    module.validate_provenance = lambda *args, **kwargs: None
    try:
        module.assert_native_controlflow(summary)
    finally:
        module.validate_provenance = original


def test_schema_v5_controlflow_requires_physical_fc1_fusion_counts():
    summary = qualified_r61_host_summary()
    summary["summary_schema_version"] = 5
    summary.update(
        native_pack_calls=709, native_unpack_calls=611,
        native_pack_cache_hits=382,
        native_pack_cache_logical_bytes_saved=69055500,
        native_pack_cache_physical_bytes_saved=72978432,
        native_prepacked_input_calls=12,
        native_prepacked_input_physical_bytes=25362432,
    )
    summary["cpu_controlflow"]["native_api_calls"] = {"pack": 709, "unpack": 611}
    layout = summary["codec_by_layout_dtype"]["NDWC_INT8"]
    layout["native_pack_calls"] = 709
    layout["native_unpack_calls"] = 611
    summary["host_executor"].update(
        gelu_quantize_calls=0,
        gelu_pack_bf16_concatenate_calls=12,
        resize_align_corners_calls=5,
        host_calls=69,
    )
    module = controlflow()
    original = module.validate_provenance
    module.validate_provenance = lambda *args, **kwargs: None
    try:
        module.assert_native_controlflow(summary)
        summary["native_prepacked_input_calls"] = 11
        with pytest.raises(AssertionError, match="native_prepacked_input_calls"):
            module.assert_native_controlflow(summary)
    finally:
        module.validate_provenance = original


def test_schema_v6_controlflow_requires_physical_attention_fusion_counts():
    summary = qualified_r61_host_summary()
    summary["summary_schema_version"] = 6
    summary.update(
        native_pack_calls=697, native_unpack_calls=179,
        native_pack_cache_hits=382,
        native_pack_cache_logical_bytes_saved=69055500,
        native_pack_cache_physical_bytes_saved=72978432,
        native_prepacked_input_calls=24,
        native_prepacked_input_physical_bytes=31703040,
    )
    summary["cpu_controlflow"]["native_api_calls"] = {"pack": 697, "unpack": 179}
    layout = summary["codec_by_layout_dtype"]["NDWC_INT8"]
    layout["native_pack_calls"] = 697
    layout["native_unpack_calls"] = 179
    summary["host_executor"].update(
        quantize_calls=24, gelu_quantize_calls=0,
        gelu_pack_bf16_concatenate_calls=12,
        attention_pack_bf16_heads_calls=12,
        resize_align_corners_calls=5, host_calls=69,
    )
    module = controlflow()
    original = module.validate_provenance
    module.validate_provenance = lambda *args, **kwargs: None
    try:
        module.assert_native_controlflow(summary)
        summary["native_prepacked_input_calls"] = 23
        with pytest.raises(AssertionError, match="native_prepacked_input_calls"):
            module.assert_native_controlflow(summary)
    finally:
        module.validate_provenance = original
    summary["native_pack_cache_hits"] = 381
    module.validate_provenance = lambda *args, **kwargs: None
    try:
        with pytest.raises(AssertionError, match="native_pack_cache_hits"):
            module.assert_native_controlflow(summary)
    finally:
        module.validate_provenance = original


def test_schema_v7_controlflow_requires_device_handle_lifetime_counts():
    summary = qualified_r61_host_summary()
    summary["summary_schema_version"] = 7
    summary.update(
        native_pack_calls=673, native_unpack_calls=179,
        submission_groups=236,
        native_pack_cache_hits=382,
        native_pack_cache_logical_bytes_saved=69055500,
        native_pack_cache_physical_bytes_saved=72978432,
        native_prepacked_input_calls=24,
        native_prepacked_input_physical_bytes=31703040,
        encoder_resident_intermediates=True,
        h2c_skipped_bytes=25362432,
    )
    summary["cpu_controlflow"].update(
        native_api_calls={"pack": 673, "unpack": 179},
        transport_groups=236,
    )
    layout = summary["codec_by_layout_dtype"]["NDWC_INT8"]
    layout["native_pack_calls"] = 673
    layout["native_unpack_calls"] = 179
    summary["host_executor"].update(
        quantize_calls=24, gelu_quantize_calls=0,
        gelu_pack_bf16_concatenate_calls=12,
        attention_pack_bf16_heads_calls=12,
        resize_align_corners_calls=5, host_calls=69,
    )
    summary["cpp_runtime"] = {
        "python_submission_groups": 236,
        "physical_npu_dispatches": 443,
        "device_tensor_handle_creations": 60,
        "device_tensor_handle_invalidations": 60,
        "device_tensor_live_handles": 0,
        "device_tensor_forwarded_inputs": 12,
        "device_tensor_connections": 12,
    }
    module = controlflow()
    original = module.validate_provenance
    module.validate_provenance = lambda *args, **kwargs: None
    try:
        module.assert_native_controlflow(summary)
        summary["cpp_runtime"]["device_tensor_live_handles"] = 1
        with pytest.raises(AssertionError, match="device_tensor_live_handles"):
            module.assert_native_controlflow(summary)
    finally:
        module.validate_provenance = original


def committed_current_evidence():
    root = ROOT / "artifacts/u250_cpp_transaction_paired_fc1_r73"
    return json.loads((root / "full_controlflow.summary.json").read_text()), {
        "report_path": root / "native_codec_all_oracle.json",
        "input_inventory_path": root / "controlflow_input_inventory.json",
        "host_executor_report_path": root / "host_executor_qualification.json",
    }


def test_r61_gate_rejects_python_host_executor():
    summary = qualified_r61_host_summary()
    summary["host_executor"]["backend"] = "python"
    with pytest.raises(AssertionError, match="host executor"):
        controlflow().assert_native_controlflow(summary)


def test_r61_gate_rejects_unreconciled_host_timing():
    summary = qualified_r61_host_summary()
    summary["latency_breakdown"]["unattributed_host_residual_ms"] += 2.0
    with pytest.raises(AssertionError, match="reconcile"):
        controlflow().assert_native_controlflow(summary)


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
    summary, evidence = committed_current_evidence()
    controlflow().assert_native_controlflow(summary, **evidence)
    summary["native_pack_ms"] = 12841.621 - 68.
    summary["codec_pack_ms_total"] = summary["native_pack_ms"]
    for item in summary["codec_by_layout_dtype"].values():
        item["native_pack_ms"] = 0.0
    summary["codec_by_layout_dtype"]["NDWC_INT8"]["native_pack_ms"] = summary["native_pack_ms"]
    with pytest.raises(AssertionError, match="12841.621"):
        controlflow().assert_native_controlflow(summary, **evidence)


def test_shell_rejects_stale_r60_evidence_before_board_work(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(qualified_summary()))
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_u250_mapped_r58_gate.sh"),
         "--check-native-controlflow", str(path)], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "native source does not match qualification report" in result.stderr


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


def test_open_trace_allows_cpu_runtime_shared_memory(tmp_path):
    trace = tmp_path / "opens.log"
    trace.write_text(
        '42 openat(AT_FDCWD, "/dev/shm", O_RDONLY|O_DIRECTORY) = 3\n'
        '42 openat(AT_FDCWD, "/dev/shm/__KMP_REGISTERED_LIB_42", '
        'O_RDWR|O_CREAT, 0600) = 4\n'
    )
    assert controlflow().inspect_open_trace(trace)["device_open_attempts"] == 0


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


@pytest.mark.parametrize("target", ["cfg", "contract", "host-plan", "host-params", "input",
                                   "resident-bank", "helper", "runtime-yaml"])
def test_portable_gate_rejects_input_digest_changes_without_remote_files(tmp_path, target):
    module = controlflow()
    summary, evidence = committed_current_evidence()
    provenance = summary["provenance"]
    for index, token in enumerate(provenance["invocation"]):
        if token.startswith("--") and index + 1 < len(provenance["invocation"]):
            if not provenance["invocation"][index + 1].startswith("--"):
                provenance["invocation"][index + 1] = str(tmp_path / "missing" / token[2:])
    provenance["extension_path"] = str(tmp_path / "missing" / "extension.so")
    if target == "cfg":
        provenance["cfg_sha256"][next(iter(provenance["cfg_sha256"]))] = "0" * 64
    elif target == "resident-bank":
        summary["resident_bank_sha256"] = "0" * 64
    elif target == "helper":
        provenance["helper_sha256"] = "0" * 64
    elif target == "runtime-yaml":
        provenance["runtime_input_sha256"]["architecture-16"] = "0" * 64
    else:
        provenance["input_sha256"][target] = "0" * 64
    with pytest.raises(AssertionError, match="canonical inventory"):
        module.assert_native_controlflow(summary, **evidence)


@pytest.mark.parametrize("exists", [False, True])
def test_gate_rejects_missing_or_tampered_canonical_inventory(tmp_path, monkeypatch, exists):
    module = controlflow()
    inventory = tmp_path / "controlflow_input_inventory.json"
    if exists:
        inventory.write_text('{"schema_version": 1, "cfg_sha256": {}}\n')
    monkeypatch.setattr(module, "INPUT_INVENTORY_PATH", inventory, raising=False)
    with pytest.raises(AssertionError, match="canonical inventory"):
        module.assert_native_controlflow(qualified_summary())


def test_committed_r60_summary_is_stale_against_current_sources():
    path = ROOT / "artifacts/u250_native_codec/full_controlflow.summary.json"
    assert path.is_file(), "authentic native full control-flow evidence is missing"
    module = controlflow()
    summary = json.loads(path.read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        module.assert_native_controlflow(summary)


def test_committed_r61_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r61"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary,
            report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json",
        )


def test_committed_r62_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r62"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary,
            report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json",
        )


def test_committed_r63_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r63"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary,
            report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json",
        )


def test_committed_r64_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r64"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary,
            report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json",
        )


def test_committed_r65_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r65"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary, report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json")


def test_committed_r66_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_host_graph_r66"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary, report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json")


def test_committed_r67_summary_is_stale_against_current_sources():
    root = ROOT / "artifacts/u250_encoder_residency_r67"
    summary = json.loads((root / "full_controlflow.summary.json").read_text())
    with pytest.raises(AssertionError, match="source_sha256 mismatch"):
        controlflow().assert_native_controlflow(
            summary, report_path=root / "native_codec_all_oracle.json",
            input_inventory_path=root / "controlflow_input_inventory.json",
            host_executor_report_path=root / "host_executor_qualification.json")


def test_committed_r73_summary_qualifies_with_paired_fc1_transactions():
    summary, evidence = committed_current_evidence()
    controlflow().assert_native_controlflow(summary, **evidence)
    host = summary["host_executor"]
    assert host["gelu_lut_misses"] == 12
    assert host["gelu_lut_hits"] == 0
    assert host["gelu_lut_elements"] == 25251840
    assert host["resize_align_corners_calls"] == 5
    assert summary["decoder_host_ops"]["Resize"]["calls"] == 5
    assert host["gelu_quantize_calls"] == 0
    assert host["gelu_pack_bf16_concatenate_calls"] == 12
    assert host["attention_pack_bf16_heads_calls"] == 12
    assert summary["native_pack_calls"] == 673
    assert summary["native_unpack_calls"] == 179
    assert summary["native_pack_cache_hits"] == 346
    assert summary["native_prepacked_input_calls"] == 24
    assert summary["encoder_resident_intermediates"] is True
    assert summary["h2c_skipped_bytes"] == 25362432
    assert summary["submission_groups"] == 200
    runtime = summary["cpp_runtime"]
    assert runtime["device_tensor_handle_creations"] == 60
    assert runtime["device_tensor_handle_invalidations"] == 60
    assert runtime["device_tensor_live_handles"] == 0
    assert runtime["device_tensor_forwarded_inputs"] == 12
    assert runtime["device_tensor_connections"] == 12
    assert runtime["physical_npu_dispatches"] == 407
    assert runtime["python_transport_api_calls"] == 200
    assert runtime["cpp_resident_transaction_calls"] == 200
