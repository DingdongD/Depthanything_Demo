#!/usr/bin/env python3
"""Replay the full r43 graph with real native codecs and zero-output CPU DMA.

The parent runs the worker under strace and checks every open before accepting
the summary. Only the external transport is replaced: hybrid.main, cfg
preflight, CppMappedRuntime.run_group, and all native conversions stay real.
This qualifies CPU control flow, not FPGA output accuracy or board latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


VENDOR_CODEC_BASELINE_MS = 12841.621
FAKE_OUTPUT_SHA256 = "b8db8d36a400c7fa0bc135cb6eb992c5cd350dac7e1657fa9029dee343af463b"
SOURCE_NAMES = {
    "run_u250_native_codec_controlflow.py", "run_u250_depthanything_hybrid.py",
    "u250_cpp_mapped_runtime.py", "u250_layout_descriptors.py", "fpga_dma_batch.cpp",
}
HOST_SOURCE_NAMES = {
    "u250_host_profile.py", "u250_host_executor.py", "u250_host_graph.hpp",
    "qualify_u250_host_executor.py",
}
INPUT_NAMES = {"manifest", "contract", "host-plan", "host-params", "input", "layout-codec-report"}
HOST_INPUT_NAMES = {"host-executor-report"}
INPUT_INVENTORY_PATH = (Path(__file__).resolve().parent.parent
                        / "artifacts/u250_native_codec/controlflow_input_inventory.json")
# Independently collected from the authentic remote package; never learned
# from the summary being checked. Changing the inventory requires review.
INPUT_INVENTORY_SHA256 = "7c2432aa0890c6de104294c0fccea0260d1d027275d5495d0360dacfcc9f72fa"
HOST_BREAKDOWN = (
    "resident_bank_load_ms", "cfg_preparse_ms", "cfg_vendor_activation_ms",
    "input_pack_ms", "output_unpack_ms", "h2c_ms", "npu_ms", "c2h_ms",
    "decoder_host_ops_ms",
)


def require(condition, message):
    # An evidence gate must still run under python -O.
    if not condition:
        raise AssertionError(message)


def assert_native_controlflow(summary, report_path=None, input_inventory_path=None,
                              host_executor_report_path=None):
    """Reject incomplete counters, device evidence, or an unmet timing gate."""
    schema = summary.get("summary_schema_version")
    paired_fc1 = schema == 10
    resident_intermediates = schema in (7, 10)
    reusable_pack = schema in (4, 5, 6, 7, 10)
    physical_fc1_fusion = schema in (5, 6, 7, 10)
    physical_attention_fusion = schema in (6, 7, 10)
    expected = {
        "native_pack_calls": (673 if resident_intermediates else
                              697 if physical_attention_fusion else
                              709 if physical_fc1_fusion else
                              721 if reusable_pack else 1103),
        "native_unpack_calls": (179 if physical_attention_fusion else
                                611 if physical_fc1_fusion else 683),
        "vendor_pack_calls": 0, "vendor_unpack_calls": 0,
        "npu_calls": 407 if paired_fc1 else 443,
        "submission_groups": (200 if paired_fc1 else
                               236 if resident_intermediates else 248),
        "submission_group_dispatches": 407 if paired_fc1 else 443,
    }
    for key, value in expected.items():
        require(type(summary.get(key)) is int and summary[key] == value,
                f"{key}: expected {value}, got {summary.get(key)!r}")
    for key, value in (("layout_codec", "native"), ("finite", True),
                       ("c2h_exact_half_size", True), ("fallback_reasons", {})):
        require(summary.get(key) == value, f"invalid {key}")
    layouts = summary.get("codec_by_layout_dtype")
    require(isinstance(layouts, dict) and layouts, "missing per-layout timings")
    for operation in ("pack", "unpack"):
        key = f"native_{operation}_ms"
        elapsed = summary.get(key)
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                f"invalid {key}")
        require(summary.get(f"codec_{operation}_ms_total") == elapsed,
                f"inconsistent {operation} timing total")
        for field in ("calls", "ms"):
            key = f"native_{operation}_{field}"
            values = [entry.get(key, 0) for entry in layouts.values()]
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                        for v in values), f"invalid per-layout {key}")
            require(math.isclose(sum(values), summary[key], rel_tol=1e-9, abs_tol=1e-6),
                    f"inconsistent per-layout {key}")
    total = summary["native_pack_ms"] + summary["native_unpack_ms"]
    require(total < VENDOR_CODEC_BASELINE_MS,
            f"native pack+unpack {total:.6f} ms must be below {VENDOR_CODEC_BASELINE_MS} ms")
    evidence = summary.get("cpu_controlflow", {})
    for key, value in (("fake_transport", True), ("open_trace_checked", True),
                       ("device_open_attempts", 0), ("runtime_lock_open_attempts", 0),
                       ("open_trace_fd_decoding", True), ("protected_writes_checked", True),
                       ("protected_write_open_attempts", 0),
                       ("transport_dispatches", 407 if paired_fc1 else 443),
                       ("transport_groups", 200 if paired_fc1 else
                        236 if resident_intermediates else 248),
                       ("native_api_calls", {
                           "pack": (673 if resident_intermediates else
                                    697 if physical_attention_fusion else
                                    709 if physical_fc1_fusion else
                                    721 if reusable_pack else 1103),
                           "unpack": (179 if physical_attention_fusion else
                                      611 if physical_fc1_fusion else 683),
                       })):
        require(evidence.get(key) == value, f"invalid CPU evidence: {key}")
    require(type(evidence.get("traced_open_calls")) is int and evidence["traced_open_calls"] > 0,
            "trace evidence requires positive traced_open_calls")
    require(valid_sha256(evidence.get("open_trace_sha256")), "invalid open_trace_sha256")
    require(summary.get("output_sha256") == FAKE_OUTPUT_SHA256,
            "output_sha256 does not match historical r58 fake output")
    if schema in (2, 3, 4, 5, 6, 7, 10):
        validate_host_execution(summary)
    if reusable_pack:
        for field, expected_value in (
            ("native_pack_cache_hits", 346 if paired_fc1 else 382),
            ("native_pack_cache_logical_bytes_saved",
             50116620 if paired_fc1 else 69055500),
            ("native_pack_cache_physical_bytes_saved",
             53956608 if paired_fc1 else 72978432),
        ):
            require(summary.get(field) == expected_value,
                    f"invalid {field}")
    if physical_fc1_fusion:
        for field, expected_value in (
            ("native_prepacked_input_calls", 24 if physical_attention_fusion else 12),
            ("native_prepacked_input_physical_bytes",
             31703040 if physical_attention_fusion else 25362432),
        ):
            require(summary.get(field) == expected_value,
                    f"invalid {field}")
    if resident_intermediates:
        require(summary.get("encoder_resident_intermediates") is True,
                "encoder resident intermediates flag is missing")
        require(summary.get("h2c_skipped_bytes") == 25362432,
                "resident H2C savings do not match the qualified plan")
        runtime = summary.get("cpp_runtime")
        require(isinstance(runtime, dict), "resident runtime counters are missing")
        for field, expected_value in (
            ("python_submission_groups", 200 if paired_fc1 else 236),
            ("physical_npu_dispatches", 407 if paired_fc1 else 443),
            ("device_tensor_handle_creations", 60),
            ("device_tensor_handle_invalidations", 60),
            ("device_tensor_live_handles", 0),
            ("device_tensor_forwarded_inputs", 12),
            ("device_tensor_connections", 12),
        ):
            require(runtime.get(field) == expected_value,
                    f"invalid resident runtime counter: {field}")
        if paired_fc1:
            for field, expected_value in (
                ("python_transport_api_calls", 200),
                ("cpp_resident_transaction_calls", 200),
            ):
                require(runtime.get(field) == expected_value,
                        f"invalid resident runtime counter: {field}")
    validate_provenance(summary, report_path, input_inventory_path,
                        host_executor_report_path)


def validate_host_execution(summary):
    host = summary.get("host_executor")
    require(isinstance(host, dict)
            and host.get("requested") == "cpp"
            and host.get("backend") == "cpp"
            and host.get("fallback_reason") is None,
            "qualified C++ host executor is required")
    require(valid_sha256(host.get("qualification_sha256")),
            "invalid host executor qualification SHA-256")
    require(valid_sha256(host.get("extension_sha256")),
            "invalid host executor extension SHA-256")
    fused = summary.get("summary_schema_version") in (5, 6, 7, 10)
    require(type(host.get("gelu_quantize_calls")) is int
            and host["gelu_quantize_calls"] == (0 if fused else 12),
            "host executor has invalid GELU-quantize call count")
    counter_fields = ["quantize_calls", "gelu_quantize_calls", "add_calls",
                      "add_quantize_calls", "concatenate_calls"]
    if fused:
        counter_fields.append("gelu_pack_bf16_concatenate_calls")
        require(host.get("gelu_pack_bf16_concatenate_calls") == 12,
                "host executor must execute 12 physical FC1 GELU-pack fusions")
    if summary.get("summary_schema_version") in (6, 7, 10):
        counter_fields.append("attention_pack_bf16_heads_calls")
        require(host.get("attention_pack_bf16_heads_calls") == 12,
                "host executor must execute 12 physical attention-pack fusions")
    if summary.get("summary_schema_version") in (3, 4, 5, 6, 7, 10):
        counter_fields.append("resize_align_corners_calls")
        require(host.get("resize_align_corners_calls") == 5,
                "host executor must execute 5 align-corners Resize calls")
    for field in ("host_calls", *counter_fields, "host_elements"):
        require(type(host.get(field)) is int and host[field] >= 0,
                f"invalid host executor {field}")
    require(host["host_calls"] == sum(host.get(field, -1)
                                      for field in counter_fields),
            "host executor call counters do not reconcile")
    profile = summary.get("host_profile")
    breakdown = summary.get("latency_breakdown")
    require(isinstance(profile, dict) and profile, "missing host profile")
    require(isinstance(breakdown, dict), "missing latency breakdown")
    for name, item in profile.items():
        require(isinstance(name, str) and name and isinstance(item, dict),
                "malformed host profile")
        require(type(item.get("calls")) is int and item["calls"] > 0,
                f"invalid host profile {name}.calls")
        for field in ("ms", "elements", "bytes"):
            value = item.get(field)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    f"invalid host profile {name}.{field}")
    measured = sum(float(item["ms"]) for item in profile.values())
    require(abs(measured - breakdown.get("host_profile_ms_total", -1)) <= 0.5,
            "host profile total does not reconcile")
    combined = (breakdown.get("host_profile_ms_total", -1)
                + breakdown.get("unattributed_host_residual_ms", -1))
    require(abs(combined - breakdown.get("host_graph_and_python_residual_ms", -1)) <= 0.5,
            "host timing compatibility residual does not reconcile")
    require(all(type(breakdown.get(field)) in (int, float)
                and math.isfinite(breakdown[field]) and breakdown[field] >= 0
                for field in HOST_BREAKDOWN),
            "invalid host timing breakdown")
    reconciled = sum(float(breakdown[field]) for field in HOST_BREAKDOWN) + combined
    require(abs(reconciled - summary.get("process_wall_ms", -1)) <= 0.5,
            "host timing does not reconcile with process wall")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value):
    return (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and value != hashlib.sha256(b"").hexdigest())


def validate_provenance(summary, report_path=None, input_inventory_path=None,
                        host_executor_report_path=None):
    """Bind recorded evidence to current source and the actual qualification report.

    Every input/cfg digest is compared with a separately retained, pinned
    inventory, including on hosts without the original remote files. Files
    that are present are additionally rehashed.
    """
    provenance = summary.get("provenance")
    require(isinstance(provenance, dict), "missing provenance")
    modern = summary.get("summary_schema_version") in (2, 3, 4, 5, 6, 7, 10)
    inventory_path = (Path(input_inventory_path) if input_inventory_path else
                      (Path(__file__).resolve().parent.parent
                       / "artifacts/u250_host_graph_r61/controlflow_input_inventory.json")
                      if modern else INPUT_INVENTORY_PATH)
    require(inventory_path.is_file(), "canonical inventory is missing")
    inventory_digest = sha256_file(inventory_path)
    if modern:
        require(provenance.get("input_inventory_sha256") == inventory_digest,
                "canonical inventory SHA-256 mismatch")
    else:
        require(inventory_digest == INPUT_INVENTORY_SHA256,
                "canonical inventory SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text())
    source_dir = Path(__file__).resolve().parent
    sources = provenance.get("source_sha256")
    expected_sources = SOURCE_NAMES | HOST_SOURCE_NAMES if modern else SOURCE_NAMES
    require(isinstance(sources, dict) and set(sources) == expected_sources,
            "provenance requires complete source_sha256 fields")
    for name, digest in sources.items():
        require(valid_sha256(digest) and digest == sha256_file(source_dir / name),
                f"provenance source_sha256 mismatch: {name}")
    inputs = provenance.get("input_sha256")
    expected_inputs = INPUT_NAMES | HOST_INPUT_NAMES if modern else INPUT_NAMES
    require(isinstance(inputs, dict) and set(inputs) == expected_inputs
            and all(valid_sha256(digest) for digest in inputs.values()),
            "provenance requires complete input/report SHA-256 fields")
    require(inputs == {role: item["sha256"] for role, item in inventory["inputs"].items()},
            "input digest values do not match canonical inventory")
    require(summary.get("resident_bank_sha256") == inventory["resident_bank"]["sha256"],
            "resident bank digest does not match canonical inventory")
    require(provenance.get("helper_sha256") == inventory["helper"]["sha256"],
            "helper digest does not match canonical inventory")
    runtime_inputs = {role: item["sha256"] for role, item in inventory["runtime_inputs"].items()}
    require(provenance.get("runtime_input_sha256") == runtime_inputs,
            "runtime input digest values do not match canonical inventory")
    report_path = (Path(report_path) if report_path else
                   source_dir.parent / "artifacts/u250_native_codec/all_oracle.json")
    require(report_path.is_file(), f"qualification report is missing: {report_path}")
    require(inputs["layout-codec-report"] == sha256_file(report_path),
            "provenance layout-codec-report SHA-256 mismatch")
    report = json.loads(report_path.read_text())
    require(inputs["manifest"] == report.get("manifest_sha256"), "provenance manifest mismatch")
    require(sources["fpga_dma_batch.cpp"] == report.get("native_source_sha256"),
            "provenance native source does not match qualification report")
    for key in ("extension_sha256", "helper_sha256", "worker_log_sha256"):
        require(valid_sha256(provenance.get(key)), f"invalid provenance {key}")
    require(provenance["extension_sha256"] == report.get("extension_sha256"),
            "provenance extension does not match qualification report")
    if modern:
        host_path = (Path(host_executor_report_path)
                     if host_executor_report_path is not None else
                     Path(report_path).with_name("host_executor_qualification.json"))
        if host_executor_report_path is None:
            invocation = provenance.get("invocation", [])
            flag = "--host-executor-report"
            if flag in invocation and invocation.index(flag) + 1 < len(invocation):
                host_path = Path(invocation[invocation.index(flag) + 1])
        require(host_path.is_file(), "host executor qualification report is missing")
        host_report_digest = sha256_file(host_path)
        require(inputs["host-executor-report"] == host_report_digest
                and summary["host_executor"]["qualification_sha256"] == host_report_digest,
                "host executor qualification SHA-256 mismatch")
        host_report = json.loads(host_path.read_text())
        require(host_report.get("schema") == "u250-host-executor-qualification-v1"
                and host_report.get("qualified") is True,
                "invalid host executor qualification report")
        require(host_report.get("source_sha256") == sources["u250_host_graph.hpp"],
                "host executor source SHA-256 mismatch")
        require(host_report.get("extension_sha256") == provenance["extension_sha256"]
                == summary["host_executor"]["extension_sha256"],
                "host executor extension SHA-256 mismatch")
        operations = host_report.get("operations")
        required_ops = {
            "quantize", "gelu_quantize", "add", "add_quantize", "concatenate"
        }
        if summary.get("summary_schema_version") in (3, 4, 5, 6, 7, 10):
            required_ops.add("resize_align_corners")
        require(isinstance(operations, dict) and set(operations) == required_ops
                and all(isinstance(item, dict) and item.get("exact") is True
                        and type(item.get("cases")) is int and item["cases"] > 0
                        for item in operations.values()),
                "host executor operations are not completely exact")
        if summary.get("summary_schema_version") == 5:
            physical = host_report.get("physical_fusions")
            require(isinstance(physical, dict)
                    and set(physical) == {"gelu_pack_bf16_concatenate"}
                    and physical["gelu_pack_bf16_concatenate"].get("exact") is True
                    and type(physical["gelu_pack_bf16_concatenate"].get("cases")) is int
                    and physical["gelu_pack_bf16_concatenate"]["cases"] > 0,
                    "host executor physical fusion is not completely exact")
        if summary.get("summary_schema_version") in (6, 7, 10):
            physical = host_report.get("physical_fusions")
            required = {"gelu_pack_bf16_concatenate", "attention_pack_bf16_heads"}
            require(isinstance(physical, dict) and set(physical) == required
                    and all(item.get("exact") is True
                            and type(item.get("cases")) is int and item["cases"] > 0
                            for item in physical.values()),
                    "host executor physical fusions are not completely exact")
    cfg = provenance.get("cfg_sha256")
    cfg_names = {user["case"] + "_cfg.txt" for entry in report["descriptors"]
                 for user in entry["users"]}
    expected_cfg_count = 226 if summary.get("summary_schema_version") == 10 else 262
    require(isinstance(cfg, dict) and len(cfg) == expected_cfg_count and set(cfg) == cfg_names
            and all(valid_sha256(digest) for digest in cfg.values()),
            f"provenance requires all {expected_cfg_count} qualified cfg names and hashes")
    require(cfg == inventory["cfg_sha256"], "cfg digest values do not match canonical inventory")
    invocation = provenance.get("invocation")
    require(isinstance(invocation, list) and all(isinstance(v, str) for v in invocation),
            "provenance invocation is missing")
    recorded = {}
    for key in expected_inputs | {"case-dir", "cfg-dir", "runtime-dir"}:
        flag = "--" + key
        require(invocation.count(flag) == 1 and invocation.index(flag) + 1 < len(invocation),
                f"provenance invocation is missing {flag}")
        recorded[key] = Path(invocation[invocation.index(flag) + 1])
    for key, path in recorded.items():
        if key in inputs and path.is_file():
            require(sha256_file(path) == inputs[key], f"recorded input changed: {key}")
    if recorded["cfg-dir"].is_dir():
        for name, digest in cfg.items():
            path = recorded["cfg-dir"] / name
            require(path.is_file() and sha256_file(path) == digest, f"recorded cfg changed: {name}")
    extension_path = provenance.get("extension_path")
    require(isinstance(extension_path, str) and extension_path, "missing provenance extension_path")
    if Path(extension_path).is_file():
        require(sha256_file(extension_path) == provenance["extension_sha256"], "recorded extension changed")
    helper = recorded["case-dir"] / "run_u250_resident_compiled_case.py"
    if helper.is_file():
        require(sha256_file(helper) == provenance["helper_sha256"], "recorded helper changed")
    bank = recorded["case-dir"] / inventory["resident_bank"]["name"]
    if bank.is_file():
        require(sha256_file(bank) == inventory["resident_bank"]["sha256"], "recorded bank changed")
    for role, item in inventory["runtime_inputs"].items():
        path = recorded[item["base"]] / item["name"]
        if path.is_file():
            require(sha256_file(path) == runtime_inputs[role], f"recorded runtime input changed: {role}")


def forbidden_path(path):
    resolved = os.path.realpath(os.fsdecode(path))
    if resolved == "/tmp/ds-u250-runtime.lock":
        return "runtime_lock"
    if (resolved.startswith("/dev/")
            and resolved not in {"/dev/null", "/dev/urandom", "/dev/random"}
            and resolved != "/dev/shm" and not resolved.startswith("/dev/shm/")):
        return "device"
    return None


def inspect_open_trace(path, *, protected_dirs=(), initial_cwd=None):
    """Count attempts, including failed opens, in strace's open-family log."""
    text = Path(path).read_text()
    require(bool(text.strip()), "empty open trace")
    attempts = {"device": 0, "runtime_lock": 0}
    protected = [Path(p).resolve() for p in protected_dirs]
    writes = []
    opens = 0
    for line in text.splitlines():
        call = re.search(r"\b(open|openat|openat2|creat)\(", line)
        if not call:
            continue
        match = re.search(r'"((?:[^"\\]|\\.)*)"', line)
        require(match is not None, f"cannot parse open trace: {line}")
        name = json.loads('"' + match.group(1) + '"')
        opens += 1
        resolved = name if os.path.isabs(name) else None
        if resolved is None:
            prefix = line[call.end():match.start()].strip().rstrip(",").strip()
            annotation = re.fullmatch(r"(?:\d+|AT_FDCWD)<(/[^>]*)>", prefix)
            directory = annotation.group(1) if annotation else None
            if (prefix == "AT_FDCWD" or call.group(1) in {"open", "creat"}) and initial_cwd:
                directory = str(initial_cwd)
            if directory:
                resolved = os.path.normpath(os.path.join(directory, name))
        category = forbidden_path(resolved) if resolved else None
        if resolved is None:
            # A missing dirfd annotation cannot turn a suspicious hardware or
            # lock basename into a harmless file under this checker's cwd.
            parts = Path(name).parts
            if "ds-u250-runtime.lock" in parts:
                category = "runtime_lock"
            elif any(re.match(r"^(?:xdma|fpga|uio|vfio|tty|nvme|dri|renderD|nvidia|kfd|card\d|mem$|kmem$|port$)",
                              part) for part in parts):
                category = "device"
        if category:
            attempts[category] += 1
        writing = (call.group(1) == "creat" or
                   re.search(r"\bO_(?:WRONLY|RDWR|CREAT|TRUNC|TMPFILE)\b", line[match.end():]))
        if writing and protected:
            require(resolved is not None, f"cannot resolve write open against protected packages: {line}")
            target = Path(resolved).resolve()
            if any(target == root or root in target.parents for root in protected):
                writes.append(str(target))
    require(opens > 0, "trace contains no open calls")
    return {"open_trace_checked": True,
            "traced_open_calls": opens,
            "device_open_attempts": attempts["device"],
            "runtime_lock_open_attempts": attempts["runtime_lock"],
            "protected_writes_checked": bool(protected),
            "protected_write_open_attempts": len(writes),
            "protected_write_paths": writes,
            "open_trace_sha256": sha256_file(path)}


def run_worker(args):
    sys.dont_write_bytecode = True

    def audit(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            require(forbidden_path(values[0]) is None,
                    f"CPU control-flow worker forbids opening {values[0]}")

    sys.addaudithook(audit)
    import numpy as np
    if __package__:
        from . import run_u250_depthanything_hybrid as hybrid
        from . import u250_cpp_mapped_runtime as mapped
    else:
        import run_u250_depthanything_hybrid as hybrid
        import u250_cpp_mapped_runtime as mapped

    # Loading the extension is CPU-only. Never instantiate its DMA class.
    extension = mapped.load_fpga_dma_batch(args.fpga_dma_batch)
    codec = extension.DmaBatch
    counts = {"pack": 0, "unpack": 0}
    transports = []

    class ZeroOutputDma:
        """Same fake boundary as test_u250_cpp_mapped_runtime.FakeDmaBatch."""
        validate_descriptor = staticmethod(codec.validate_descriptor)

        @staticmethod
        def pack_tensor(array, descriptor):
            result = codec.pack_tensor(array, descriptor)
            counts["pack"] += 1
            return result

        @staticmethod
        def unpack_tensor(even, odd, descriptor):
            result = codec.unpack_tensor(even, odd, descriptor)
            counts["unpack"] += 1
            return result

        def __init__(self):
            self.reset_stats()
            transports.append(self)

        def reset_stats(self):
            self.groups = 0
            self.dispatches = 0
            self.upload_bytes = 0
            self.download_bytes = 0
            self.schedule = hashlib.sha256()

        def h2c_batch_safe(self, requests):
            for bank, address, array in requests:
                require(bank in (0, 1) and address >= 0, "invalid fake H2C address")
                require(array.dtype == np.uint8 and array.flags.c_contiguous,
                        "fake H2C requires contiguous byte buffer")
                self.upload_bytes += array.nbytes

        def c2h_batch_safe(self, requests):
            values = []
            for bank, address, size in requests:
                require(bank in (0, 1) and address >= 0 and size > 0 and size % 128 == 0,
                        "invalid fake C2H extent")
                values.append(np.zeros(size, dtype=np.uint8))
                self.download_bytes += size
            return values

        def run_npu_chain(self, programs, timeout_ms):
            require(bool(programs) and timeout_ms > 0, "invalid fake NPU group")
            self.groups += 1
            self.dispatches += len(programs)
            self.schedule.update(json.dumps(programs, sort_keys=True).encode() + b"\n")
            return [0.] * len(programs)

        def run_resident_transaction(self, h2c, programs, c2h, timeout_ms,
                                     reopen_each_segment):
            require(reopen_each_segment is True,
                    "CPU control flow must retain safe-DMA semantics")
            require(bool(programs) and timeout_ms > 0,
                    "invalid fake resident transaction")
            self.h2c_batch_safe(h2c)
            self.groups += 1
            self.dispatches += len(programs)
            self.schedule.update(json.dumps(programs, sort_keys=True).encode() + b"\n")
            return {
                "outputs": self.c2h_batch_safe(c2h),
                "npu_seconds": [0.] * len(programs),
                "h2c_seconds": 0.,
                "c2h_seconds": 0.,
            }

        def stats(self):
            return {"fake_transport": True, "transport_groups": self.groups,
                    "transport_dispatches": self.dispatches,
                    "h2c_bytes": self.upload_bytes, "c2h_bytes": self.download_bytes,
                    "schedule_sha256": self.schedule.hexdigest()}

    argv = [str(hybrid.__file__)]
    inputs = {
        "case-dir": args.case_dir, "runtime-dir": args.runtime_dir,
        "manifest": args.case_dir / "resident_kernel_bank_manifest.json",
        "contract": args.case_dir / "depthanything_u250_runtime_contract.json",
        "host-plan": args.case_dir / "depthanything_u250_host_plan.json",
        "host-params": args.case_dir / "depthanything_u250_host_params.npz",
        "cfg-dir": args.case_dir / "cfg", "input": args.case_dir / "demo05.npy",
        "layout-codec-report": args.layout_codec_report,
        "fpga-dma-batch": args.fpga_dma_batch,
        "output": args.output.with_suffix(".npz"),
    }
    if args.host_executor_report is not None:
        inputs["host-executor-report"] = args.host_executor_report
    for key, value in inputs.items():
        argv.extend(["--" + key, str(value)])
    argv.extend(["--dma-runtime", "cpp_mapped", "--layout-codec", "native",
                 "--host-executor", args.host_executor,
                 "--attention-launch-group", "3", "--decoder-launch-group", "32",
                 "--depth-only"])
    if args.encoder_resident_intermediates:
        argv.append("--encoder-resident-intermediates")
    fake_extension = SimpleNamespace(
        DmaBatch=ZeroOutputDma,
        HostGraphExecutor=getattr(extension, "HostGraphExecutor", None),
        __file__=extension.__file__,
        _u250_extension_sha256=extension._u250_extension_sha256)
    with patch.object(mapped, "load_fpga_dma_batch", return_value=fake_extension), \
            patch.object(sys, "argv", argv):
        require(hybrid.main() == 0, "hybrid runner failed")
    require(len(transports) == 1, "expected exactly one CPU transport")
    summary = json.loads(args.output.with_suffix(".summary.json").read_text())
    # The runtime's hardware capability flags describe its usual transport;
    # this worker actually allocates ordinary arrays and maps no BAR.
    summary["cpp_runtime"].update(mapped_bar=False, locked_host_buffers=False)
    summary["cpu_controlflow"] = {
        **transports[0].stats(), "native_api_calls": counts,
        "scope": "full r43 host graph and real C++ codecs; zero NPU output; no board accuracy claim",
        "native_dma_constructed": False,
        "python_open_guard": True,
    }
    sources = [Path(__file__), Path(hybrid.__file__), Path(mapped.__file__),
               Path(__file__).with_name("u250_layout_descriptors.py"),
               Path(__file__).with_name("fpga_dma_batch.cpp")]
    if args.host_executor != "python":
        sources.extend(Path(__file__).with_name(name) for name in HOST_SOURCE_NAMES)
    runtime_sources = {
        "architecture-16": args.runtime_dir / "arch_16_mono.yaml",
        "architecture-256": args.runtime_dir / "arch_256_mono.yaml",
        "vendor-codec": Path(sys.modules["npz2bin"].__file__),
        "vendor-tensor-helper": Path(sys.modules["npz_util"].__file__),
    }
    summary["provenance"] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__, "extension_sha256": sha256_file(extension.__file__),
        "extension_path": str(extension.__file__),
        "source_sha256": {p.name: sha256_file(p) for p in sources},
        "input_sha256": {key: sha256_file(value) for key, value in inputs.items()
                         if key not in {"case-dir", "runtime-dir", "cfg-dir", "output", "fpga-dma-batch"}},
        "cfg_sha256": {p.name: sha256_file(p) for p in sorted(inputs["cfg-dir"].glob("*_cfg.txt"))},
        "helper_sha256": sha256_file(args.case_dir / "run_u250_resident_compiled_case.py"),
        "runtime_input_sha256": {role: sha256_file(path) for role, path in runtime_sources.items()},
        "invocation": argv,
    }
    if args.input_inventory is not None:
        summary["provenance"]["input_inventory_sha256"] = sha256_file(
            args.input_inventory
        )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-summary", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--runtime-dir", type=Path,
                        default=Path("/home/visitor/Documents/nn_inference"))
    parser.add_argument("--layout-codec-report", type=Path)
    parser.add_argument("--fpga-dma-batch", type=Path)
    parser.add_argument("--host-executor", choices=("python", "auto", "cpp"),
                        default="python")
    parser.add_argument("--host-executor-report", type=Path)
    parser.add_argument("--input-inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder-resident-intermediates", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_summary:
        assert_native_controlflow(
            json.loads(args.check_summary.read_text()), args.layout_codec_report,
            args.input_inventory, args.host_executor_report,
        )
        print("native CPU control-flow gate passed")
        return 0
    for key in ("case_dir", "runtime_dir", "layout_codec_report", "fpga_dma_batch", "output"):
        if getattr(args, key) is None:
            parser.error("--" + key.replace("_", "-") + " is required")
        setattr(args, key, getattr(args, key).resolve())
    if args.host_executor != "python" and args.host_executor_report is None:
        parser.error("--host-executor auto/cpp requires --host-executor-report")
    if args.host_executor_report is not None:
        args.host_executor_report = args.host_executor_report.resolve()
    if args.host_executor != "python" and args.input_inventory is None:
        parser.error("--host-executor auto/cpp requires --input-inventory")
    if args.input_inventory is not None:
        args.input_inventory = args.input_inventory.resolve()
    for protected in (args.case_dir, args.runtime_dir):
        require(args.output != protected and protected not in args.output.parents,
                "CPU qualification output must be outside the existing package/runtime")
    if args.worker:
        run_worker(args)
        return 0
    strace = shutil.which("strace")
    require(strace is not None, "strace is required to prove zero device/lock opens")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".opens.log")
    log = args.output.with_suffix(".worker.log")
    command = [strace, "-f", "-qq", "-yy", "-s", "4096", "-e", "trace=open,openat,openat2,creat",
               "-o", str(trace), sys.executable, str(Path(__file__).resolve()),
               *sys.argv[1:], "--worker"]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode:
        raise RuntimeError(f"CPU control-flow worker failed ({result.returncode}); see {log}")
    summary = json.loads(args.output.read_text())
    summary["cpu_controlflow"].update(inspect_open_trace(
        trace, protected_dirs=(args.case_dir, args.runtime_dir), initial_cwd=Path.cwd()))
    summary["cpu_controlflow"]["open_trace_fd_decoding"] = True
    summary["provenance"]["worker_log_sha256"] = sha256_file(log)
    summary["native_codec_total_ms"] = summary["native_pack_ms"] + summary["native_unpack_ms"]
    summary["vendor_codec_baseline_ms"] = VENDOR_CODEC_BASELINE_MS
    assert_native_controlflow(summary, args.layout_codec_report, args.input_inventory,
                              args.host_executor_report)
    summary["cpu_controlflow"]["passed"] = True
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("native CPU control-flow gate passed: " + json.dumps({
        key: summary[key] for key in ("npu_calls", "submission_groups", "native_pack_calls",
                                     "native_unpack_calls", "vendor_pack_calls",
                                     "vendor_unpack_calls", "native_codec_total_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
