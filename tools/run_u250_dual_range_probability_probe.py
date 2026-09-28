#!/usr/bin/env python3
"""Capture fine, BF16, and residual probability carriers from U250."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, np.float32) / scale), -128, 127).astype(
        np.int8
    )


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def tensor_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    difference = actual - expected
    norm_product = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    return {
        "cosine": float(np.dot(actual, expected) / max(norm_product, 1e-30)),
        "relative_l2": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-30)
        ),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--fine-step", type=float, default=1.0 / 16384.0)
    parser.add_argument("--residual-step", type=float, default=1.0 / 128.0)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(case_dir))
    sys.path.insert(1, str(runtime_dir))
    import fpgaDma  # type: ignore
    import npz2bin  # type: ignore
    from npz_util import createBF16TensorFromDict  # type: ignore
    from run_u250_resident_compiled_case import (  # type: ignore
        C2H_DEVICES,
        DDR_BASES,
        EventWaiter,
        H2C_DEVICES,
        clear_interrupt,
        configure_npu,
        export_npz_allow_nonfinite,
        merge_2ddr,
        preferred_output_key,
        reg_write,
        split_2ddr,
    )

    manifest = json.loads(args.manifest.read_text())
    record = manifest["cases"][0]
    contract = json.loads(args.contract.read_text())
    scales = contract["encoder"][0]["attention"]["heads"][args.head]["scales_bf16"]
    with np.load(args.trace, allow_pickle=False) as trace:
        begin, end = args.head * 64, (args.head + 1) * 64
        q = np.asarray(trace["q_l00"][0, 0, :, begin:end], np.float32)
        k = np.asarray(trace["k_l00"][0, 0, :, begin:end], np.float32)
    q_code = quantize(q, float(scales["q"]))
    k_code = quantize(k, float(scales["k"]))
    start = args.row_start
    logical_inputs = [q_code[None, None, start : start + 256], k_code.T[None, None]]

    npz2bin.read_yaml(
        [str(runtime_dir / "arch_16_mono.yaml"), str(runtime_dir / "arch_256_mono.yaml")]
    )
    npz2bin.read_cfg(str(args.cfg).removesuffix("_cfg.txt"))
    packed_values = npz2bin.read_npz_dict(
        createBF16TensorFromDict,
        "input",
        [{"input": np.ascontiguousarray(value)} for value in logical_inputs],
    )
    packed = [np.ascontiguousarray(value).reshape(-1).view(np.uint8) for value in packed_values]
    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    configure_npu(fpgaDma, record)
    reg_write(fpgaDma, 0x00, 0x3F)
    reg_write(fpgaDma, 0x34, 1)
    for tensor, combined in zip(record["inputs"], packed):
        for bank, half in enumerate(split_2ddr(combined)):
            address = (
                (record["base_addresses"][4] + tensor["address"]) * 0x80
                + DDR_BASES[bank]
            )
            fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
    waiter = EventWaiter()
    waiter.drain()
    reg_write(fpgaDma, 0x34, 2)
    started = time.perf_counter()
    try:
        event = waiter.wait(args.timeout_ms)
        npu_ms = (time.perf_counter() - started) * 1000.0
        clear_interrupt(fpgaDma)
        physical = []
        for tensor in record["outputs"]:
            halves = []
            for bank in range(2):
                address = (
                    (record["base_addresses"][4] + tensor["address"]) * 0x80
                    + DDR_BASES[bank]
                )
                halves.append(
                    np.ascontiguousarray(
                        fpgaDma.card2np(C2H_DEVICES[bank], address, tensor["size_per_bank"])
                    )
                )
            physical.append(merge_2ddr(halves[0], halves[1]))
    finally:
        waiter.close()
    decoded = npz2bin.buffer_to_npz_dict(
        export_npz_allow_nonfinite, "output", physical
    )
    values = [np.asarray(item[preferred_output_key(item)]) for item in decoded]
    fine_code = values[0].reshape(256, 1370).astype(np.int8)
    bf16_probability = values[1].reshape(256, 1370).astype(np.float32)
    residual_code = values[2].reshape(256, 1370).astype(np.int8)
    logits = (
        q_code[start : start + 256].astype(np.int32) @ k_code.astype(np.int32).T
    ).astype(np.float32) * np.float32(float(scales["q"]) * float(scales["k"]))
    fp32_probability = softmax(logits)
    represented = (
        fine_code.astype(np.float32) * args.fine_step
        + residual_code.astype(np.float32) * args.residual_step
    )
    report = {
        "schema_version": 1,
        "event": int(event),
        "npu_ms": npu_ms,
        "rows": [start, start + 256],
        "fine_vs_bf16": tensor_metrics(
            fine_code.astype(np.float32) * args.fine_step, bf16_probability
        ),
        "bf16_vs_fp32": tensor_metrics(bf16_probability, fp32_probability),
        "represented_vs_bf16": tensor_metrics(represented, bf16_probability),
        "row_sum": {
            "bf16_mean": float(np.mean(np.sum(bf16_probability, axis=-1))),
            "represented_mean": float(np.mean(np.sum(represented, axis=-1))),
            "fp32_mean": float(np.mean(np.sum(fp32_probability, axis=-1))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        fine_code=fine_code,
        bf16_probability=bf16_probability,
        residual_code=residual_code,
        represented=represented,
        fp32_probability=fp32_probability,
    )
    print("DUAL_RANGE_PROBABILITY_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
