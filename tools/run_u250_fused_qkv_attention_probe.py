#!/usr/bin/env python3
"""Run one fused QKV+six-head-attention kernel on U250 and compare a trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, np.float32) / scale), -128, 127).astype(
        np.int8
    )


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
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

    manifest = json.loads(
        (case_dir / "resident_kernel_bank_manifest.json").read_text()
    )
    record = manifest["cases"][0]
    cfg = case_dir / "cfg" / f"{record['name']}_cfg.txt"
    contract = json.loads(args.contract.read_text())
    if not 0 <= args.layer < len(contract["encoder"]):
        raise ValueError(f"layer {args.layer} is outside the encoder contract")
    input_scale = float(
        contract["encoder"][args.layer]["qkv"]["input_quantization"]["scale"]
    )
    with np.load(args.trace, allow_pickle=False) as trace:
        normalized = np.asarray(
            trace[f"encoder_l{args.layer:02d}_norm1"], np.float32
        )
        reference = np.asarray(trace[f"attention_l{args.layer:02d}"], np.float32)
    code = quantize(normalized, input_scale)[:, None]

    npz2bin.read_yaml([
        str(runtime_dir / "arch_16_mono.yaml"),
        str(runtime_dir / "arch_256_mono.yaml"),
    ])
    npz2bin.read_cfg(str(cfg).removesuffix("_cfg.txt"))
    packed_values = npz2bin.read_npz_dict(
        createBF16TensorFromDict, "input", [{"input": code}]
    )
    packed = [
        np.ascontiguousarray(value).reshape(-1).view(np.uint8)
        for value in packed_values
    ]

    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
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
                    address = (
                        (record["base_addresses"][4] + tensor["address"]) * 0x80
                        + DDR_BASES[bank]
                    )
                    fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
            h2c_ms = (time.perf_counter() - started) * 1000.0
            reg_write(fpgaDma, 0x34, 2)
            started = time.perf_counter()
            event = waiter.wait(args.timeout_ms)
            npu_ms = (time.perf_counter() - started) * 1000.0
            clear_interrupt(fpgaDma)
            started = time.perf_counter()
            physical = []
            for tensor in record["outputs"]:
                combined_size = int(tensor["size_per_bank"])
                halves = []
                for bank in range(2):
                    address = (
                        (record["base_addresses"][4] + tensor["address"]) * 0x80
                        + DDR_BASES[bank]
                    )
                    halves.append(np.ascontiguousarray(fpgaDma.card2np(
                        C2H_DEVICES[bank], address, combined_size // 2
                    )))
                physical.append(merge_2ddr(halves[0], halves[1]))
            c2h_ms = (time.perf_counter() - started) * 1000.0
            decoded = npz2bin.buffer_to_npz_dict(
                export_npz_allow_nonfinite, "output", physical
            )
            # Preserve the codec dtype here.  In particular, bitdepth-8
            # network outputs are already the physical INT8 codes.  Casting
            # them to float32 would make the comparison below quantize those
            # codes a second time and report a false, heavily saturated QKV
            # mismatch.
            decoded_runs.append([
                np.asarray(item[preferred_output_key(item)]) for item in decoded
            ])
            timings.append({
                "event": int(event), "h2c_ms": h2c_ms,
                "npu_ms": npu_ms, "c2h_ms": c2h_ms,
            })
    finally:
        waiter.close()

    io_order_path = case_dir / "cfg" / f"{record['name']}.IO_order.yaml"
    if not io_order_path.exists():
        io_order_path = cfg.with_name(cfg.name.replace("_cfg.txt", ".IO_order.yaml"))
    output_order = None
    if io_order_path.exists():
        output_order = yaml.safe_load(io_order_path.read_text()).get("Outputs")
        if output_order is not None:
            output_order = [int(index) for index in output_order]
            if sorted(output_order) != list(range(len(decoded_runs[-1]))):
                raise ValueError("IO_order Outputs is not a permutation")
            decoded_runs = [
                [run[index] for index in output_order] for run in decoded_runs
            ]

    # The first launch after loading a fresh resident bank may observe the
    # bitstream's cold-start state.  Judge numerical correctness from the last
    # launch and report every earlier launch against that steady-state result.
    values = decoded_runs[-1]
    if len(values) == 36:
        qkv_runs = [run[:24] for run in decoded_runs]
        context_runs = [run[24:] for run in decoded_runs]
        expected_codes = []
        output_scales = []
        with np.load(args.trace, allow_pickle=False) as trace:
            q = np.asarray(trace[f"q_l{args.layer:02d}"], np.float32)[0, 0]
            k = np.asarray(trace[f"k_l{args.layer:02d}"], np.float32)[0, 0]
            v = np.asarray(trace[f"v_l{args.layer:02d}"], np.float32)[0, 0]
        for head in contract["encoder"][args.layer]["attention"]["heads"]:
            index = int(head["head"])
            begin, end = index * 64, (index + 1) * 64
            scales = head["scales_bf16"]
            qh = quantize(q[:, begin:end], float(scales["q"]))
            kh = quantize(k[:, begin:end], float(scales["k"]))
            vh = quantize(v[:, begin:end], float(scales["v"]))
            expected_codes.extend([qh[:256], kh.T, vh, qh[256:]])
            output_scales.extend([
                float(scales["q"]), float(scales["k"]),
                float(scales["v"]), float(scales["q"]),
            ])

        actual_codes = []
        qkv_code_metrics = []
        for actual, expected, scale in zip(
                qkv_runs[-1], expected_codes, output_scales):
            actual_array = np.asarray(actual)
            code = (actual_array.astype(np.int8, copy=False)
                    if np.issubdtype(actual_array.dtype, np.integer)
                    else quantize(actual_array, scale))
            code = code.reshape(expected.shape)
            actual_codes.append(code)
            delta = code.astype(np.int16) - expected.astype(np.int16)
            qkv_code_metrics.append({
                "exact": bool(np.array_equal(code, expected)),
                "mismatch_fraction": float(np.mean(delta != 0)),
                "max_code_delta": int(np.max(np.abs(delta))),
                "mean_abs_code_delta": float(np.mean(np.abs(delta))),
            })

        attention_runs = []
        for run in context_runs:
            heads = [
                np.concatenate((
                    np.asarray(run[head * 2]).reshape(256, 64),
                    np.asarray(run[head * 2 + 1]).reshape(145, 64),
                ), axis=0)
                for head in range(6)
            ]
            attention_runs.append(np.concatenate(heads, axis=-1)[None, None])
        hardware = attention_runs[-1]
        report = {
            "schema_version": 1,
            "kernel": record["name"],
            "layer": args.layer,
            "input_scale": input_scale,
            "timings": timings,
            "io_output_order": output_order,
            "qkv_code_metrics": qkv_code_metrics,
            "qkv_all_exact": all(item["exact"] for item in qkv_code_metrics),
            "raw_output_summaries": [
                {
                    "index": index,
                    "dtype": str(np.asarray(value).dtype),
                    "shape": list(np.asarray(value).shape),
                    "minimum": float(np.min(value)),
                    "maximum": float(np.max(value)),
                }
                for index, value in enumerate(values)
            ],
            "hardware_vs_reference_attention": metrics(hardware, reference),
            "earlier_runs_vs_steady_state": [
                metrics(value, hardware) for value in attention_runs[:-1]
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        saved = {
            "hardware": hardware, "reference": reference,
            "hardware_runs": np.concatenate(attention_runs),
        }
        saved.update({f"qkv_code_{index:02d}": value
                      for index, value in enumerate(actual_codes)})
        saved.update({f"raw_output_{index:02d}": np.asarray(value)
                      for index, value in enumerate(values)})
        np.savez_compressed(args.output.with_suffix(".npz"), **saved)
        print(json.dumps(report, sort_keys=True))
        return 0
    if len(values) == 24:
        flat_runs = [np.concatenate([
            np.asarray(value, np.float32).reshape(-1) for value in run
        ]) for run in decoded_runs]
        report = {
            "schema_version": 1,
            "kernel": record["name"],
            "input_scale": input_scale,
            "timings": timings,
            "earlier_runs_vs_steady_state": [
                metrics(run, flat_runs[-1]) for run in flat_runs[:-1]
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        saved = {f"qkv_output_{index:02d}": value
                 for index, value in enumerate(values)}
        saved["qkv_flat_runs"] = np.stack(flat_runs)
        np.savez_compressed(args.output.with_suffix(".npz"), **saved)
        print(json.dumps(report, sort_keys=True))
        return 0
    diagnostic = None
    output_bitdepths = [int(item["bitdepth"]) for item in record["outputs"]]
    if len(values) == 3 and output_bitdepths == [8, 16, 16]:
        diagnostic = {"k_matrix": np.asarray(values[0])}
        context_runs = [run[1:] for run in decoded_runs]
        values = context_runs[-1]
    elif len(values) == 6 and output_bitdepths == [8, 8, 16, 8, 8, 16]:
        # Diagnostic one-head ABI order emitted by
        # fuse_u250_qkv_attention_layer.py --emit-qkv.  Compiler scheduling
        # places context0 between K and V, and context1 last.
        diagnostic = {
            "q0": np.asarray(values[0]), "k": np.asarray(values[1]),
            "v": np.asarray(values[3]), "q1": np.asarray(values[4]),
        }
        context_runs = [[run[2], run[5]] for run in decoded_runs]
        values = context_runs[-1]
    elif len(values) == 12 and output_order == [
            index for head in range(6) for index in (head, 6 + head)]:
        # IO_order maps the production compiler's physical chunk-major ABI
        # [q0_h0..q0_h5, q1_h0..q1_h5] into logical head-major pairs.  Keep
        # that ordering explicit; treating the reordered values as two
        # contiguous chunk groups silently crosses heads and invalidates the
        # production-boundary comparison.
        head_major_runs = decoded_runs
        run_tensors = [np.concatenate([
            np.concatenate((
                np.asarray(run[head * 2]).reshape(256, 64),
                np.asarray(run[head * 2 + 1]).reshape(145, 64),
            ), axis=0)
            for head in range(6)
        ], axis=-1)[None, None] for run in head_major_runs]
        hardware = run_tensors[-1]
        report = {
            "schema_version": 1,
            "kernel": record["name"],
            "layer": args.layer,
            "input_scale": input_scale,
            "timings": timings,
            "io_output_order": output_order,
            "output_order_after_io_map": "head-major",
            "hardware_vs_r136_attention": metrics(hardware, reference),
            "earlier_runs_vs_steady_state": [
                metrics(run_tensor, hardware)
                for run_tensor in run_tensors[:-1]
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            hardware=hardware,
            reference=reference,
            hardware_runs=np.concatenate(run_tensors, axis=0),
            **{
                f"raw_output_head_major_{index:02d}": np.asarray(value)
                for index, value in enumerate(values)
            },
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    else:
        context_runs = decoded_runs
    if len(values) == 0 or len(values) % 2:
        raise RuntimeError(f"expected paired context outputs, got {len(values)}")
    head_count = len(values) // 2
    chunk0 = [np.asarray(value).reshape(256, 64)
              for value in values[:head_count]]
    chunk1 = [np.asarray(value).reshape(145, 64)
              for value in values[head_count:]]
    heads = [np.concatenate((chunk0[i], chunk1[i]), axis=0)
             for i in range(head_count)]
    hardware = np.concatenate(heads, axis=-1)[None, None]
    run_tensors = [np.concatenate([
        np.concatenate((
            np.asarray(run[i]).reshape(256, 64),
            np.asarray(run[head_count + i]).reshape(145, 64),
        ), axis=0)
        for i in range(head_count)
    ], axis=-1)[None, None] for run in context_runs]
    reference = reference[..., :head_count * 64]
    repeat_metrics = [metrics(
        run_tensor,
        hardware,
    ) for run_tensor in run_tensors[:-1]]
    report = {
        "schema_version": 1,
        "kernel": record["name"],
        "layer": args.layer,
        "input_scale": input_scale,
        "timings": timings,
        "hardware_vs_r136_attention": metrics(hardware, reference),
        "earlier_runs_vs_steady_state": repeat_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    saved = {
        "hardware": hardware, "reference": reference,
        "hardware_runs": np.concatenate(run_tensors, axis=0),
    }
    if diagnostic is not None:
        saved.update({f"diagnostic_{key}": value
                      for key, value in diagnostic.items()})
    np.savez_compressed(args.output.with_suffix(".npz"), **saved)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
