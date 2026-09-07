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


def require(condition, message):
    # An evidence gate must still run under python -O.
    if not condition:
        raise AssertionError(message)


def assert_native_controlflow(summary):
    """Reject incomplete counters, device evidence, or an unmet timing gate."""
    expected = {
        "native_pack_calls": 1103, "native_unpack_calls": 683,
        "vendor_pack_calls": 0, "vendor_unpack_calls": 0,
        "npu_calls": 443, "submission_groups": 248,
        "submission_group_dispatches": 443,
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
                       ("transport_dispatches", 443), ("transport_groups", 248),
                       ("native_api_calls", {"pack": 1103, "unpack": 683})):
        require(evidence.get(key) == value, f"invalid CPU evidence: {key}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_path(path):
    resolved = os.path.realpath(os.fsdecode(path))
    if resolved == "/tmp/ds-u250-runtime.lock":
        return "runtime_lock"
    if resolved.startswith("/dev/") and resolved not in {
            "/dev/null", "/dev/urandom", "/dev/random"}:
        return "device"
    return None


def inspect_open_trace(path):
    """Count attempts, including failed opens, in strace's open-family log."""
    text = Path(path).read_text()
    require(bool(text.strip()), "empty open trace")
    attempts = {"device": 0, "runtime_lock": 0}
    opens = 0
    for line in text.splitlines():
        if not re.search(r"\b(?:open|openat|openat2|creat)\(", line):
            continue
        match = re.search(r'"((?:[^"\\]|\\.)*)"', line)
        require(match is not None, f"cannot parse open trace: {line}")
        name = json.loads('"' + match.group(1) + '"')
        opens += 1
        category = forbidden_path(name)
        if category:
            attempts[category] += 1
    require(opens > 0, "trace contains no open calls")
    return {"open_trace_checked": True,
            "traced_open_calls": opens,
            "device_open_attempts": attempts["device"],
            "runtime_lock_open_attempts": attempts["runtime_lock"],
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
    for key, value in inputs.items():
        argv.extend(["--" + key, str(value)])
    argv.extend(["--dma-runtime", "cpp_mapped", "--layout-codec", "native",
                 "--attention-launch-group", "3", "--decoder-launch-group", "32",
                 "--depth-only"])
    fake_extension = SimpleNamespace(DmaBatch=ZeroOutputDma)
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
        "invocation": argv,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-summary", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--runtime-dir", type=Path,
                        default=Path("/home/visitor/Documents/nn_inference"))
    parser.add_argument("--layout-codec-report", type=Path)
    parser.add_argument("--fpga-dma-batch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_summary:
        assert_native_controlflow(json.loads(args.check_summary.read_text()))
        print("native CPU control-flow gate passed")
        return 0
    for key in ("case_dir", "runtime_dir", "layout_codec_report", "fpga_dma_batch", "output"):
        if getattr(args, key) is None:
            parser.error("--" + key.replace("_", "-") + " is required")
        setattr(args, key, getattr(args, key).resolve())
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
    command = [strace, "-f", "-qq", "-s", "4096", "-e", "trace=open,openat,openat2,creat",
               "-o", str(trace), sys.executable, str(Path(__file__).resolve()),
               *sys.argv[1:], "--worker"]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if result.returncode:
        raise RuntimeError(f"CPU control-flow worker failed ({result.returncode}); see {log}")
    summary = json.loads(args.output.read_text())
    summary["cpu_controlflow"].update(inspect_open_trace(trace))
    summary["provenance"]["worker_log_sha256"] = sha256_file(log)
    summary["native_codec_total_ms"] = summary["native_pack_ms"] + summary["native_unpack_ms"]
    summary["vendor_codec_baseline_ms"] = VENDOR_CODEC_BASELINE_MS
    assert_native_controlflow(summary)
    summary["cpu_controlflow"]["passed"] = True
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("native CPU control-flow gate passed: " + json.dumps({
        key: summary[key] for key in ("npu_calls", "submission_groups", "native_pack_calls",
                                     "native_unpack_calls", "vendor_pack_calls",
                                     "vendor_unpack_calls", "native_codec_total_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
