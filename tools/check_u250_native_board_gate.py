#!/usr/bin/env python3
"""Fail-closed, CPU-only checks for native board evidence and deployment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


EXPECTED = "2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725"
BANK = "9d01d1fd4ecae67755a4314e98a2f9d4cbe7182f3985d7573113de579b7f9577"
STAGES = {"demo05_decoder_only": (89, 38), "demo05_resume_l11": (118, 55),
          "demo05_full_first": (443, 248), "demo05_full_resident": (443, 248)}
BREAKDOWN = ("resident_bank_load_ms", "cfg_preparse_ms", "cfg_vendor_activation_ms",
             "input_pack_ms", "output_unpack_ms", "h2c_ms", "npu_ms", "c2h_ms",
             "decoder_host_ops_ms", "host_graph_and_python_residual_ms")
HOST_BREAKDOWN_V2 = ("host_profile_ms_total", "unattributed_host_residual_ms")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_frame(name, report):
    require(name in STAGES, f"unknown stage: {name}")
    for field, expected in (("output_sha256", EXPECTED), ("resident_bank_sha256", BANK),
                            ("layout_codec", "native"), ("fallback_reasons", {}),
                            ("finite", True)):
        require(type(report.get(field)) is type(expected) and report[field] == expected,
                f"{name}: invalid {field}")
    metrics = report.get("metrics")
    require(isinstance(metrics, dict), f"{name}: missing metrics")
    for field in ("relative_l2", "rmse"):
        require(type(metrics.get(field)) in (int, float) and metrics[field] == 0,
                f"{name}: {field} must be zero")
    calls, groups = STAGES[name]
    for field, expected in (("vendor_pack_calls", 0), ("vendor_unpack_calls", 0),
                            ("npu_calls", calls), ("submission_groups", groups),
                            ("submission_group_dispatches", calls)):
        require(type(report.get(field)) is int and report[field] == expected,
                f"{name}: invalid {field}")
    runtime = report.get("cpp_runtime", {})
    require(runtime.get("safe_dma") is True, f"{name}: safe_dma is required")
    for field, expected in (("stale_events", 0), ("physical_npu_dispatches", calls),
                            ("python_submission_groups", groups)):
        require(type(runtime.get(field)) is int and runtime[field] == expected,
                f"{name}: invalid {field}")
    if name.startswith("demo05_full_"):
        for field, expected in (("native_pack_calls", 1103), ("native_unpack_calls", 683)):
            require(type(report.get(field)) is int and report[field] == expected,
                    f"{name}: invalid {field}")
    if name == "demo05_full_resident":
        for field in ("resident_bank_reused", "cpp_runtime_reused", "codec_yaml_reused"):
            require(report.get(field) is True, f"{name}: invalid {field}")
        require(type(report.get("load_ms")) in (int, float) and report["load_ms"] == 0,
                f"{name}: load_ms must be zero")
    for field in ("wall_ms", "process_wall_ms"):
        value = report.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"{name}: invalid {field}")
    for field in BREAKDOWN:
        value = report.get("latency_breakdown", {}).get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"{name}: invalid {field}")
    schema = report.get("summary_schema_version", 1)
    require(type(schema) is int and schema in (1, 2),
            f"{name}: invalid summary_schema_version")
    if schema == 2:
        profile = report.get("host_profile")
        require(isinstance(profile, dict) and profile,
                f"{name}: missing host profile")
        for operation, item in profile.items():
            require(isinstance(operation, str) and operation
                    and isinstance(item, dict),
                    f"{name}: malformed host profile")
            require(type(item.get("calls")) is int and item["calls"] > 0,
                    f"{name}: invalid host profile {operation}.calls")
            for field in ("ms", "elements", "bytes"):
                value = item.get(field)
                require(type(value) in (int, float) and math.isfinite(value)
                        and value >= 0,
                        f"{name}: invalid host profile {operation}.{field}")
        breakdown = report["latency_breakdown"]
        for field in HOST_BREAKDOWN_V2:
            value = breakdown.get(field)
            require(type(value) in (int, float) and math.isfinite(value)
                    and value >= 0, f"{name}: invalid {field}")
        measured_profile = sum(float(item["ms"]) for item in profile.values())
        require(abs(measured_profile - breakdown["host_profile_ms_total"]) <= 0.5,
                f"{name}: host profile total does not match operations")
        combined_host = (breakdown["host_profile_ms_total"]
                         + breakdown["unattributed_host_residual_ms"])
        require(abs(combined_host
                    - breakdown["host_graph_and_python_residual_ms"]) <= 0.5,
                f"{name}: host profile does not reconcile with compatibility residual")
        accounted = sum(float(breakdown[field]) for field in BREAKDOWN
                        if field != "host_graph_and_python_residual_ms")
        require(abs(accounted + combined_host - report["process_wall_ms"]) <= 0.5,
                f"{name}: latency breakdown does not reconcile with process wall")


def check_log(content):
    require(not re.search(r"NPU timeout|timed out|stale[ -]event|event poll failed|Traceback",
                          content, re.IGNORECASE), "device error in execution log")


def verify_package(package, expected_inventory_sha256):
    path = package / "deployment_sha256.json"
    require(digest(path) == expected_inventory_sha256, "deployment inventory SHA-256 mismatch")
    inventory = json.loads(path.read_text())
    files = inventory["files"]
    required = {"tools/run_u250_depthanything_hybrid.py", "tools/u250_cpp_mapped_runtime.py",
                "tools/u250_layout_descriptors.py", "tools/fpga_dma_batch.cpp",
                "tools/check_u250_native_board_gate.py", "tools/run_u250_mapped_r58_gate.sh",
                "tools/depthanything_u250_resident_server.py", "resident_kernel_bank_manifest.json",
                "artifacts/u250_native_codec/all_oracle.json", "depthanything_u250_resident_kernel_bank.bin"}
    require(required <= files.keys(), "deployment inventory missing required files")
    for name, expected in files.items():
        candidate = package / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                f"invalid deployment path: {name}")
        require(digest(candidate) == expected, f"deployed SHA-256 mismatch: {name}")
    for name, expected in inventory["runtime_files"].items():
        require(digest(Path(inventory["runtime_dir"]) / name) == expected,
                f"runtime SHA-256 mismatch: {name}")
    report = json.loads((package / "artifacts/u250_native_codec/all_oracle.json").read_text())
    require(report.get("qualified") is True and len(report.get("descriptors", [])) == 41,
            "complete ALL oracle is required")
    require(report["native_source_sha256"] == files["tools/fpga_dma_batch.cpp"],
            "qualified native source SHA-256 mismatch")
    require(report["manifest_sha256"] == files["resident_kernel_bank_manifest.json"],
            "qualified manifest SHA-256 mismatch")
    require(report["extension_sha256"] == files[inventory["extension"]],
            "qualified extension SHA-256 mismatch")
    require(files["depthanything_u250_resident_kernel_bank.bin"] == BANK,
            "resident bank SHA-256 mismatch")
    return {"inventory_sha256": digest(path), **inventory}


def check_run(root):
    reports = {}
    for name in STAGES:
        reports[name] = json.loads((root / f"{name}.summary.json").read_text())
        check_frame(name, reports[name])
    logs = ("demo05_decoder_only.log", "demo05_resume_l11.log",
            "resident_server.jsonl", "resident_server.stderr")
    for name in logs:
        check_log((root / name).read_text())
    server = [json.loads(line) for line in (root / "resident_server.jsonl").read_text().splitlines()
              if line.startswith("{")]
    require(len(server) == 4 and server[0].get("ready") is True
            and type(server[0].get("pid")) is int and server[-1].get("shutdown") is True,
            "resident server must contain one ready PID, two results, and shutdown")
    for event, name in zip(server[1:3], ("demo05_full_first", "demo05_full_resident")):
        require(event.get("ok") is True and event.get("code") == 0
                and event.get("output_sha256") == EXPECTED
                and Path(event.get("summary", "")).name == f"{name}.summary.json",
                f"{name}: resident server request failed")
    provenance = json.loads((root / "deployment_verified.json").read_text())
    require(isinstance(provenance, dict)
            and re.fullmatch(r"[0-9a-f]{64}", provenance.get("inventory_sha256", ""))
            and isinstance(provenance.get("files"), dict)
            and provenance["files"].get("depthanything_u250_resident_kernel_bank.bin") == BANK,
            "missing or invalid deployment provenance")
    steady = reports["demo05_full_resident"]["wall_ms"]
    summary = {
        "passed": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "board, safe DMA; two full frames in one resident process",
        "expected_sha256": EXPECTED, "resident_bank_sha256": BANK,
        "stale_events_semantics": "per-frame cumulative count after reset_frame_stats; all zero",
        "resident_pid": server[0]["pid"],
        "provenance": provenance,
        "reports": {name: {**report, "summary_sha256": digest(root / f"{name}.summary.json")}
                    for name, report in reports.items()},
        "log_sha256": {name: digest(root / name) for name in logs},
        "steady_comparisons": {str(baseline): {"baseline_wall_ms": baseline,
            "measured_steady_wall_ms": steady, "speedup": baseline / steady,
            "reduction_percent": 100 * (1 - steady / baseline)}
            for baseline in (60561.193, 14202.047)},
    }
    (root / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", type=Path)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--verify-package", type=Path)
    parser.add_argument("--deployment-sha256")
    args = parser.parse_args()
    if args.frame:
        check_frame(args.frame.name.removesuffix(".summary.json"), json.loads(args.frame.read_text()))
    elif args.run_dir:
        check_run(args.run_dir)
    else:
        require(args.deployment_sha256, "--deployment-sha256 is required")
        print(json.dumps(verify_package(args.verify_package, args.deployment_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
