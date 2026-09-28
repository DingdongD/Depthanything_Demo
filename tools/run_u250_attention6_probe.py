#!/usr/bin/env python3
"""Run one six-head attention BIN and compare it with a board trace."""

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


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    difference = actual - expected
    product = float(np.linalg.norm(actual) * np.linalg.norm(expected))
    return {
        "cosine": float(np.dot(actual, expected) / max(product, 1e-30)),
        "relative_l2": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-30)
        ),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
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
        C2H_DEVICES, DDR_BASES, EventWaiter, H2C_DEVICES, clear_interrupt,
        configure_npu, export_npz_allow_nonfinite, merge_2ddr,
        preferred_output_key, reg_write, split_2ddr,
    )

    manifest = json.loads(
        (case_dir / "resident_kernel_bank_manifest.json").read_text()
    )
    expected_name = f"attention6_l{args.layer:02d}"
    matches = [item for item in manifest["cases"]
               if item["name"] == expected_name]
    record = matches[0] if len(matches) == 1 else manifest["cases"][0]
    if record["name"] not in {expected_name, "attention6_l00"}:
        raise ValueError(f"resident package has no {expected_name} kernel")
    cfg = case_dir / "cfg" / f"{record['name']}_cfg.txt"
    contract = json.loads(args.contract.read_text())
    block = contract["encoder"][args.layer]
    with np.load(args.trace, allow_pickle=False) as trace:
        q = np.asarray(trace[f"q_l{args.layer:02d}"], np.float32)[0, 0]
        k = np.asarray(trace[f"k_l{args.layer:02d}"], np.float32)[0, 0]
        v = np.asarray(trace[f"v_l{args.layer:02d}"], np.float32)[0, 0]
        reference = np.asarray(
            trace[f"attention_l{args.layer:02d}"], np.float32
        )

    logical_inputs = []
    for head in block["attention"]["heads"]:
        index = int(head["head"])
        begin, end = index * 64, (index + 1) * 64
        scales = head["scales_bf16"]
        qh = quantize(q[:, begin:end], float(scales["q"]))
        kh = quantize(k[:, begin:end], float(scales["k"]))
        vh = quantize(v[:, begin:end], float(scales["v"]))
        q1 = np.zeros((256, 64), np.int8)
        q1[:145] = qh[256:]
        logical_inputs.extend([
            qh[:256][None, None], kh.T[None, None],
            vh[None, None], q1[None, None],
        ])

    npz2bin.read_yaml([
        str(runtime_dir / "arch_16_mono.yaml"),
        str(runtime_dir / "arch_256_mono.yaml"),
    ])
    npz2bin.read_cfg(str(cfg).removesuffix("_cfg.txt"))
    packed = [
        np.ascontiguousarray(value).reshape(-1).view(np.uint8)
        for value in npz2bin.read_npz_dict(
            createBF16TensorFromDict, "input",
            [{"input": value} for value in logical_inputs],
        )
    ]

    linked = fpgaDma.file2np(
        str(case_dir / manifest["bank_file"]),
        os.path.getsize(case_dir / manifest["bank_file"]),
    )
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    configure_npu(fpgaDma, record)
    waiter = EventWaiter()
    decoded_runs = []
    timings = []
    try:
        for _ in range(args.repeats):
            reg_write(fpgaDma, 0x00, 0x3F)
            waiter.drain()
            reg_write(fpgaDma, 0x34, 1)
            started = time.perf_counter()
            for tensor, combined in zip(record["inputs"], packed):
                for bank, half in enumerate(split_2ddr(combined)):
                    address = ((record["base_addresses"][4] + tensor["address"])
                               * 0x80 + DDR_BASES[bank])
                    fpgaDma.np2card(
                        H2C_DEVICES[bank], address, half.size, half
                    )
            h2c_ms = (time.perf_counter() - started) * 1000.0
            reg_write(fpgaDma, 0x34, 2)
            started = time.perf_counter()
            event = waiter.wait(args.timeout_ms)
            npu_ms = (time.perf_counter() - started) * 1000.0
            clear_interrupt(fpgaDma)
            started = time.perf_counter()
            physical = []
            for tensor in record["outputs"]:
                halves = []
                for bank in range(2):
                    address = ((record["base_addresses"][4] + tensor["address"])
                               * 0x80 + DDR_BASES[bank])
                    halves.append(np.ascontiguousarray(fpgaDma.card2np(
                        C2H_DEVICES[bank], address,
                        int(tensor["size_per_bank"]) // 2,
                    )))
                physical.append(merge_2ddr(*halves))
            c2h_ms = (time.perf_counter() - started) * 1000.0
            decoded = npz2bin.buffer_to_npz_dict(
                export_npz_allow_nonfinite, "output", physical
            )
            decoded_runs.append([
                np.asarray(item[preferred_output_key(item)], np.float32)
                for item in decoded
            ])
            timings.append({
                "event": int(event), "h2c_ms": h2c_ms,
                "npu_ms": npu_ms, "c2h_ms": c2h_ms,
            })
    finally:
        waiter.close()

    attention_runs = []
    for values in decoded_runs:
        if len(values) != 12:
            raise RuntimeError(f"expected 12 attention outputs, got {len(values)}")
        heads = [
            np.concatenate((
                np.asarray(values[head * 2]).reshape(256, 64),
                np.asarray(values[head * 2 + 1]).reshape(256, 64)[:145],
            ), axis=0)
            for head in range(6)
        ]
        attention_runs.append(np.concatenate(heads, axis=-1)[None, None])
    hardware = attention_runs[-1]
    report = {
        "schema_version": 1,
        "kernel": record["name"],
        "layer": args.layer,
        "timings": timings,
        "hardware_vs_reference_attention": metrics(hardware, reference),
        "earlier_runs_vs_steady_state": [
            metrics(value, hardware) for value in attention_runs[:-1]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"), hardware=hardware,
        reference=reference, hardware_runs=np.concatenate(attention_runs),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
