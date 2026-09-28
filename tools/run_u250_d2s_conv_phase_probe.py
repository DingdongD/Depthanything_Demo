#!/usr/bin/env python3
"""Validate resident low-resolution phase Conv kernels against a traced Conv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, np.float32) / scale),
                   -128, 127).astype(np.int8)


def bf16_round(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def depth_to_space(value: np.ndarray, factor: int) -> np.ndarray:
    result = value
    stages = 2 if factor == 4 else 1
    for _ in range(stages):
        n, channels, height, width = result.shape
        result = result.reshape(n, channels // 4, 2, 2, height, width)
        result = result.transpose(0, 1, 4, 2, 5, 3).reshape(
            n, channels // 4, height * 2, width * 2
        )
    return result


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    return {
        "cosine": float(np.dot(actual, expected)
                        / (np.linalg.norm(actual) * np.linalg.norm(expected))),
        "relative_l2": float(np.linalg.norm(actual - expected)
                             / max(np.linalg.norm(expected), 1e-30)),
        "rmse": float(np.sqrt(np.mean((actual - expected) ** 2))),
        "max_abs": float(np.max(np.abs(actual - expected))),
        "exact_percent": float(np.mean(actual == expected) * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--resident-manifest", type=Path, required=True)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--index", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(case_dir))
    sys.path.insert(1, str(runtime_dir))
    import fpgaDma  # type: ignore
    import npz2bin  # type: ignore
    from npz_util import createBF16TensorFromDict  # type: ignore
    from run_u250_resident_compiled_case import (  # type: ignore
        C2H_DEVICES, DDR_BASES, EventWaiter, H2C_DEVICES,
        clear_interrupt, configure_npu, export_npz_allow_nonfinite,
        merge_2ddr, preferred_output_key, reg_write, split_2ddr,
    )

    npz2bin.read_yaml([str(runtime_dir / "arch_16_mono.yaml"),
                       str(runtime_dir / "arch_256_mono.yaml")])
    resident = json.loads(args.resident_manifest.read_text())
    phase_manifest = json.loads(args.phase_manifest.read_text())
    phase = next(record for record in phase_manifest["kernels"]
                 if int(record["index"]) == args.index)
    records = {record["name"]: record for record in resident["cases"]}
    is_gather = "slices" in phase["phases"][0]
    with np.load(args.trace, allow_pickle=False) as trace:
        source = trace[f"decoder_conv_{args.index - 6:02d}"]
        expected = (None if is_gather else
                    trace[f"decoder_conv_{args.index:02d}"])
    source_code = quantize(source, float(phase["input_scale"]))
    if is_gather:
        expected = depth_to_space(
            bf16_round(source_code.astype(np.float32)
                       * np.float32(phase["input_scale"])),
            int(phase["factor"]),
        )

    bank_path = case_dir / resident["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    waiter = EventWaiter()
    waiter.drain()
    timings = []

    def run_kernel(name: str, output_shape: list[int]) -> np.ndarray:
        record = records[name]
        cfg_path = args.cfg_dir / (name + "_cfg.txt")
        npz2bin.read_cfg(str(cfg_path).removesuffix("_cfg.txt"))
        packed_value = npz2bin.read_npz_dict(
            createBF16TensorFromDict, "input",
            [{"input": np.ascontiguousarray(source_code)}],
        )[0]
        packed = np.ascontiguousarray(packed_value).reshape(-1).view(np.uint8)
        configure_npu(fpgaDma, record)
        reg_write(fpgaDma, 0x00, 0x3F)
        waiter.drain()
        reg_write(fpgaDma, 0x34, 1)
        tensor = record["inputs"][0]
        for bank, half in enumerate(split_2ddr(packed)):
            address = ((record["base_addresses"][4] + tensor["address"])
                       * 0x80 + DDR_BASES[bank])
            fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
        reg_write(fpgaDma, 0x34, 2)
        started = time.perf_counter()
        event = waiter.wait(args.timeout_ms)
        elapsed = (time.perf_counter() - started) * 1000.0
        clear_interrupt(fpgaDma)
        physical = []
        for tensor in record["outputs"]:
            halves = []
            for bank in range(2):
                address = ((record["base_addresses"][4] + tensor["address"])
                           * 0x80 + DDR_BASES[bank])
                halves.append(np.ascontiguousarray(fpgaDma.card2np(
                    C2H_DEVICES[bank], address, tensor["size_per_bank"])))
            physical.append(merge_2ddr(halves[0], halves[1]))
        decoded = npz2bin.buffer_to_npz_dict(
            export_npz_allow_nonfinite, "output", physical)
        timings.append({"kernel": name, "event": int(event), "npu_ms": elapsed})
        return np.ascontiguousarray(
            decoded[0][preferred_output_key(decoded[0])]
        ).reshape(output_shape)

    output = np.empty(phase["interleaved_output_shape"], dtype=np.float32)
    try:
        for item in phase["phases"]:
            if is_gather:
                value = np.concatenate([
                    run_kernel(part["name"], part["output_shape"])
                    for part in item["slices"]
                ], axis=1)
            else:
                value = run_kernel(item["name"], phase["phase_output_shape"])
            output[:, :, int(item["phase_y"])::int(phase["factor"]),
                   int(item["phase_x"])::int(phase["factor"])] = value
    finally:
        waiter.close()
    report = {
        "schema_version": 1, "index": args.index,
        "factor": phase["factor"],
        "strategy": "gather" if is_gather else "fused_phase_conv",
        "kernels": (sum(len(item["slices"]) for item in phase["phases"])
                    if is_gather else len(phase["phases"])),
        ("metrics_vs_quantized_host_depth_to_space" if is_gather else
         "metrics_vs_traced_original_npu_conv"): metrics(output, expected),
        "npu_ms_total": sum(item["npu_ms"] for item in timings),
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("D2S_PHASE_PROBE_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
