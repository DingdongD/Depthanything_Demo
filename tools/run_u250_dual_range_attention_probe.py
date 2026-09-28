#!/usr/bin/env python3
"""Run one r89 dual-range attention head on U250 with captured Q/K/V."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def bf16(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, np.float32) / scale), -128, 127).astype(
        np.int8
    )


def tensor_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual64 = np.asarray(actual, np.float64).reshape(-1)
    expected64 = np.asarray(expected, np.float64).reshape(-1)
    denominator = max(float(np.linalg.norm(expected64)), 1e-30)
    norm_product = float(np.linalg.norm(actual64) * np.linalg.norm(expected64))
    difference = actual64 - expected64
    return {
        "cosine": float(np.dot(actual64, expected64) / max(norm_product, 1e-30)),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def dual_range_reference(
    q_code: np.ndarray,
    k_code: np.ndarray,
    v_code: np.ndarray,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    fine_step: float,
    residual_step: float,
) -> tuple[np.ndarray, dict]:
    logits = bf16(
        (q_code.astype(np.int32) @ k_code.astype(np.int32).T).astype(np.float32)
        * np.float32(q_scale * k_scale)
    )
    probability = bf16(softmax(logits))
    threshold = np.float32(127.0 * fine_step)
    # U250 SPU static-A8 Softmax truncates positive probability codes.
    fine_code = np.clip(np.floor(probability / fine_step), 0, 127).astype(np.int8)
    residual_code = np.maximum(
        np.clip(
            np.rint((probability - threshold) / residual_step), -128, 127
        ).astype(np.int16),
        0,
    ).astype(np.int8)
    fine_context = bf16(
        (fine_code.astype(np.int32) @ v_code.astype(np.int32)).astype(np.float32)
        * np.float32(fine_step * v_scale)
    )
    residual_context = bf16(
        (residual_code.astype(np.int32) @ v_code.astype(np.int32)).astype(np.float32)
        * np.float32(residual_step * v_scale)
    )
    context = bf16(fine_context + residual_context)
    stats = {
        "fine_nonzero_fraction": float(np.count_nonzero(fine_code) / fine_code.size),
        "fine_saturated_fraction": float(np.count_nonzero(fine_code == 127) / fine_code.size),
        "residual_nonzero_fraction": float(
            np.count_nonzero(residual_code) / residual_code.size
        ),
        "residual_saturated_fraction": float(
            np.count_nonzero(residual_code == 127) / residual_code.size
        ),
        "represented_probability_row_sum_mean": float(
            np.mean(
                np.sum(
                    fine_code.astype(np.float32) * fine_step
                    + residual_code.astype(np.float32) * residual_step,
                    axis=-1,
                )
            )
        ),
    }
    return context, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--fine-step", type=float)
    parser.add_argument("--residual-step", type=float)
    parser.add_argument(
        "--reference-trace", type=Path,
        help="optional FP32 trace used for raw-attention model comparison",
    )
    parser.add_argument("--repeats", type=int, default=3)
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
    name = f"attention2_l{args.layer:02d}_h{args.head:02d}"
    record = next(item for item in manifest["cases"] if item["name"] == name)
    contract = json.loads(args.contract.read_text())
    attention = contract["encoder"][args.layer]["attention"]
    head = attention["heads"][args.head]
    scales = head["scales_bf16"]
    q_scale, k_scale, v_scale = (float(scales[key]) for key in ("q", "k", "v"))
    probability = attention.get("dual_range_probability", {})
    probability = probability.get("heads", {}).get(str(args.head), probability)
    fine_step = (
        float(args.fine_step) if args.fine_step is not None
        else float(probability["fine_step"])
    )
    residual_step = (
        float(args.residual_step) if args.residual_step is not None
        else float(probability["residual_step"])
    )

    with np.load(args.trace, allow_pickle=False) as trace:
        begin, end = args.head * 64, (args.head + 1) * 64
        q_float = np.asarray(
            trace[f"q_l{args.layer:02d}"][0, 0, :, begin:end], np.float32
        )
        k_float = np.asarray(
            trace[f"k_l{args.layer:02d}"][0, 0, :, begin:end], np.float32
        )
        v_float = np.asarray(
            trace[f"v_l{args.layer:02d}"][0, 0, :, begin:end], np.float32
        )
    q_code = quantize(q_float, q_scale)
    k_code = quantize(k_float, k_scale)
    v_code = quantize(v_float, v_scale)
    row0 = args.row_start
    row1 = row0 + 256
    row2 = row1 + 256
    if row2 > q_code.shape[0]:
        raise ValueError("probe requires two complete 256-row query chunks")

    # npz2bin follows cfg order, which is the established runtime ABI:
    # q0, K^T, V, q1. The compiler may reorder ONNX graph inputs internally.
    logical_inputs = [
        q_code[None, None, row0:row1],
        k_code.T[None, None],
        v_code[None, None],
        q_code[None, None, row1:row2],
    ]
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
    waiter = EventWaiter()
    waiter.drain()
    decoded_runs = []
    timings = []
    try:
        for _ in range(args.repeats):
            reg_write(fpgaDma, 0x00, 0x3F)
            waiter.drain()
            reg_write(fpgaDma, 0x34, 1)
            h2c_started = time.perf_counter()
            for tensor, combined in zip(record["inputs"], packed):
                for bank, half in enumerate(split_2ddr(combined)):
                    address = (
                        (record["base_addresses"][4] + tensor["address"]) * 0x80
                        + DDR_BASES[bank]
                    )
                    fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
            h2c_ms = (time.perf_counter() - h2c_started) * 1000.0
            reg_write(fpgaDma, 0x34, 2)
            npu_started = time.perf_counter()
            event = waiter.wait(args.timeout_ms)
            npu_ms = (time.perf_counter() - npu_started) * 1000.0
            clear_interrupt(fpgaDma)
            c2h_started = time.perf_counter()
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
                            fpgaDma.card2np(
                                C2H_DEVICES[bank], address, tensor["size_per_bank"]
                            )
                        )
                    )
                physical.append(merge_2ddr(halves[0], halves[1]))
            c2h_ms = (time.perf_counter() - c2h_started) * 1000.0
            decoded = npz2bin.buffer_to_npz_dict(
                export_npz_allow_nonfinite, "output", physical
            )
            decoded_runs.append(
                [np.asarray(item[preferred_output_key(item)], np.float32) for item in decoded]
            )
            timings.append(
                {"event": int(event), "h2c_ms": h2c_ms, "npu_ms": npu_ms, "c2h_ms": c2h_ms}
            )
    finally:
        waiter.close()

    hardware = np.concatenate(
        [value.reshape(256, 64) for value in decoded_runs[0]], axis=0
    )
    software_parts = []
    probability_stats = []
    fp32_parts = []
    for start in (row0, row1):
        reference, stats = dual_range_reference(
            q_code[start : start + 256],
            k_code,
            v_code,
            q_scale,
            k_scale,
            v_scale,
            fine_step,
            residual_step,
        )
        software_parts.append(reference)
        probability_stats.append(stats)
        logits = q_float[start : start + 256] @ k_float.T
        fp32_parts.append(softmax(logits) @ v_float)
    software = np.concatenate(software_parts, axis=0)
    fp32 = np.concatenate(fp32_parts, axis=0)
    report = {
        "schema_version": 1,
        "case": name,
        "head": args.head,
        "rows": [row0, row2],
        "layer": args.layer,
        "fine_step": fine_step,
        "threshold": 127.0 * fine_step,
        "residual_step": residual_step,
        "v_unchanged": True,
        "av_output_gain": 1.0,
        "timings": timings,
        "all_repeats_exact": all(
            all(np.array_equal(a, b) for a, b in zip(decoded_runs[0], run))
            for run in decoded_runs[1:]
        ),
        "probability_stats": probability_stats,
        "hardware_vs_instruction_model": tensor_metrics(hardware, software),
        "hardware_vs_fp32_attention": tensor_metrics(hardware, fp32),
        "instruction_model_vs_fp32_attention": tensor_metrics(software, fp32),
    }
    saved = {"hardware": hardware, "software": software, "fp32": fp32}
    if args.reference_trace is not None:
        with np.load(args.reference_trace, allow_pickle=False) as reference_trace:
            reference = np.asarray(
                reference_trace[f"encoder_l{args.layer:02d}_attention"]
                [0, 0, row0:row2, begin:end],
                np.float32,
            )
        report["hardware_vs_model_fp32_attention"] = tensor_metrics(
            hardware, reference
        )
        saved["model_fp32"] = reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **saved)
    print("DUAL_RANGE_PROBE_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
