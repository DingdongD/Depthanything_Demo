#!/usr/bin/env python3
"""Run a resident QK->Softmax->AV probe chain and report stage errors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def bf16_round(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)
    norm_product = np.linalg.norm(actual) * np.linalg.norm(expected)
    return {
        "cosine": float(np.dot(actual, expected) / norm_product),
        "relative_l2": float(np.linalg.norm(actual - expected)
                             / max(np.linalg.norm(expected), 1e-30)),
        "rmse": float(np.sqrt(np.mean((actual - expected) ** 2))),
        "max_abs": float(np.max(np.abs(actual - expected))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--resident-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--bf16-softmax-kernel",
                        help="combined probe with BF16 Softmax and an AV A8 boundary")
    parser.add_argument("--bf16-softmax-probe-name",
                        help="probe record whose Q/K/V inputs feed the combined kernel")
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
    probes = json.loads(args.probe_manifest.read_text())
    records = {record["name"]: record for record in resident["cases"]}
    bank_path = case_dir / resident["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    waiter = EventWaiter()
    waiter.drain()
    timings = []

    def run_kernel(name: str, logical_inputs: list[np.ndarray]) -> list[np.ndarray]:
        record = records[name]
        cfg_path = args.cfg_dir / (name + "_cfg.txt")
        npz2bin.read_cfg(str(cfg_path).removesuffix("_cfg.txt"))
        packed_values = npz2bin.read_npz_dict(
            createBF16TensorFromDict, "input",
            [{"input": np.ascontiguousarray(value)} for value in logical_inputs],
        )
        packed = [np.ascontiguousarray(value).reshape(-1).view(np.uint8)
                  for value in packed_values]
        configure_npu(fpgaDma, record)
        reg_write(fpgaDma, 0x00, 0x3F)
        waiter.drain()
        reg_write(fpgaDma, 0x34, 1)
        for tensor, combined in zip(record["inputs"], packed):
            for bank, half in enumerate(split_2ddr(combined)):
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
        return [np.ascontiguousarray(item[preferred_output_key(item)])
                for item in decoded]

    report = {"schema_version": 1, "probes": [], "timings": timings}
    saved = {}
    try:
        for probe in probes["probes"]:
            with np.load(args.probe_manifest.parent / probe["data"],
                         allow_pickle=False) as data:
                q = data["q"][:, None]
                kt = data["kt"][:, None]
                value = data["v"][:, None]
                logits_reference = data["logits_bf16"][:, None]
                probability_reference = data["probability_i8"][:, None]
                context_reference = data["context_bf16"][:, None]
            logits_hardware = run_kernel(
                probe["kernels"]["qk"], [q, kt]
            )[0].reshape(logits_reference.shape)
            probability_from_hardware_logits = bf16_round(
                softmax(logits_hardware.astype(np.float32))
            )
            probability_code_from_hardware_logits = np.clip(
                np.rint(probability_from_hardware_logits
                        / float(probe["probability_scale"])), 0, 127
            ).astype(np.int8)
            probability_hardware = run_kernel(
                probe["kernels"]["softmax"], [logits_hardware]
            )[0].reshape(probability_reference.shape).astype(np.int8)
            context_from_hardware_codes = bf16_round(
                (probability_hardware.astype(np.int32)
                 @ value.astype(np.int32)).astype(np.float32)
                * np.float32(float(probe["probability_scale"])
                             * float(probe["v_scale"]))
            )
            context_hardware = run_kernel(
                probe["kernels"]["av"], [probability_hardware, value]
            )[0].reshape(context_reference.shape)
            valid = int(probe["valid_rows"])
            row = {
                "name": probe["name"], "layer": probe["layer"],
                "head": probe["head"], "chunk": probe["chunk"],
                "qk_hardware_vs_bf16_model": metrics(
                    logits_hardware[:, :, :valid], logits_reference[:, :, :valid]),
                "softmax_hardware_vs_model_same_logits": metrics(
                    probability_hardware[:, :, :valid],
                    probability_code_from_hardware_logits[:, :, :valid]),
                "softmax_chain_vs_reference_codes": metrics(
                    probability_hardware[:, :, :valid],
                    probability_reference[:, :, :valid]),
                "av_hardware_vs_integer_model_same_codes": metrics(
                    context_hardware[:, :, :valid],
                    context_from_hardware_codes[:, :, :valid]),
                "full_chain_vs_software_model": metrics(
                    context_hardware[:, :, :valid], context_reference[:, :, :valid]),
            }
            if (args.bf16_softmax_kernel
                    and probe["name"] == args.bf16_softmax_probe_name):
                bf16_boundary_context = run_kernel(
                    args.bf16_softmax_kernel, [q, kt, value]
                )[0].reshape(context_reference.shape)
                row["bf16_softmax_boundary_vs_round_model"] = metrics(
                    bf16_boundary_context[:, :, :valid],
                    context_reference[:, :, :valid],
                )
                saved[probe["name"] + "_bf16_softmax_context"] = (
                    bf16_boundary_context
                )
            report["probes"].append(row)
            for key, value_array in (
                ("logits", logits_hardware), ("probability", probability_hardware),
                ("context", context_hardware),
            ):
                saved[probe["name"] + "_" + key] = value_array
    finally:
        waiter.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **saved)
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ATTENTION_STAGE_PROBE_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
