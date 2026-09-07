#!/usr/bin/env python3
"""JSON-lines same-process server for the mapped DepthAnything U250 runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


def replace_option(arguments: list[str], option: str, value: str | None) -> list[str]:
    result = list(arguments)
    while option in result:
        index = result.index(option)
        del result[index:index + 2]
    if value is not None:
        result.extend([option, value])
    return result


def load_runner(path: Path):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("depthanything_mapped_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--base-args", type=Path, required=True,
        help="JSON array containing all invariant runner arguments",
    )
    args = parser.parse_args()
    base = json.loads(args.base_args.read_text())
    if not isinstance(base, list) or not all(isinstance(item, str) for item in base):
        raise ValueError("base args must be a JSON string array")
    runner = load_runner(args.runner)
    print(json.dumps({"ready": True, "pid": __import__("os").getpid()}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        request_started = time.perf_counter()
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"shutdown": True}), flush=True)
                return 0
            run_args = replace_option(base, "--input", str(request["input"]))
            run_args = replace_option(run_args, "--output", str(request["output"]))
            run_args = replace_option(
                run_args, "--golden",
                str(request["golden"]) if request.get("golden") else None,
            )
            old_argv = sys.argv
            try:
                sys.argv = [str(args.runner), *run_args]
                code = int(runner.main())
            finally:
                sys.argv = old_argv
            output = Path(request["output"])
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            print(json.dumps({
                "ok": code == 0, "code": code,
                "request_wall_ms": (time.perf_counter() - request_started) * 1000.0,
                "output_sha256": summary["output_sha256"],
                "runtime_wall_ms": summary["wall_ms"],
                "resident_bank_reused": summary["resident_bank_reused"],
                "cpp_runtime_reused": summary["cpp_runtime_reused"],
                "summary": str(output.with_suffix(".summary.json")),
            }, sort_keys=True), flush=True)
        except Exception as error:  # keep the process alive for diagnosable host errors
            print(json.dumps({
                "ok": False, "error": str(error),
                "traceback": traceback.format_exc(),
                "request_wall_ms": (time.perf_counter() - request_started) * 1000.0,
            }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
