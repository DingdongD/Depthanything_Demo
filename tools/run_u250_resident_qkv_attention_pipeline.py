#!/usr/bin/env python3
"""Board proof for one once-loaded resident QKV + attention kernel bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qkv-name", default="qkv_projection_l00")
    parser.add_argument("--attention-name", default="attention2_l00_h00")
    parser.add_argument("--qkv-cfg", type=Path, required=True)
    parser.add_argument("--attention-cfg", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=10000)
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
        merge_2ddr, preferred_output_key, reg_write, sha256_array, split_2ddr,
        tensor_metrics,
    )

    npz2bin.read_yaml([
        str(runtime_dir / "arch_16_mono.yaml"),
        str(runtime_dir / "arch_256_mono.yaml"),
    ])
    manifest = json.loads(args.manifest.read_text())
    records = {record["name"]: record for record in manifest["cases"]}
    qkv = records[args.qkv_name]
    attention = records[args.attention_name]

    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    load_start = time.perf_counter()
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    waiter = EventWaiter()
    waiter.drain()

    def run(record: dict, packed_inputs: list[np.ndarray], cfg_path: Path) -> tuple[list[dict], dict]:
        npz2bin.read_cfg(str(cfg_path).removesuffix("_cfg.txt"))
        cfg = dict(npz2bin.get_cfg_dict())
        runtime_cfg = {
            "base_addresses": record["base_addresses"],
            "isa_ranges": record["isa_ranges"],
        }
        configure_npu(fpgaDma, runtime_cfg)
        reg_write(fpgaDma, 0x00, 0x3F)
        waiter.drain()
        reg_write(fpgaDma, 0x34, 1)
        h2c_start = time.perf_counter()
        for tensor, combined in zip(record["inputs"], packed_inputs):
            halves = split_2ddr(combined)
            for bank, half in enumerate(halves):
                address = ((record["base_addresses"][4] + tensor["address"])
                           * 0x80 + DDR_BASES[bank])
                fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
        h2c_ms = (time.perf_counter() - h2c_start) * 1000.0
        reg_write(fpgaDma, 0x34, 2)
        npu_start = time.perf_counter()
        event = waiter.wait(args.timeout_ms)
        npu_ms = (time.perf_counter() - npu_start) * 1000.0
        clear_interrupt(fpgaDma)
        c2h_start = time.perf_counter()
        physical = []
        for tensor in record["outputs"]:
            halves = []
            for bank in range(2):
                address = ((record["base_addresses"][4] + tensor["address"])
                           * 0x80 + DDR_BASES[bank])
                halves.append(np.ascontiguousarray(fpgaDma.card2np(
                    C2H_DEVICES[bank], address, tensor["size_per_bank"])))
            physical.append(merge_2ddr(halves[0], halves[1]))
        c2h_ms = (time.perf_counter() - c2h_start) * 1000.0
        decoded = npz2bin.buffer_to_npz_dict(
            export_npz_allow_nonfinite, "output", physical)
        return decoded, {"event": int(event), "h2c_ms": h2c_ms,
                         "npu_ms": npu_ms, "c2h_ms": c2h_ms}

    qkv_input = fpgaDma.file2np(str(args.input), os.path.getsize(args.input))
    qkv_runs = []
    attention_runs = []
    outputs = []
    try:
        for repeat in range(args.repeats):
            decoded_qkv, qkv_timing = run(qkv, [qkv_input], args.qkv_cfg)
            qkv_runs.append(qkv_timing)
            q, k, v = [np.asarray(item[preferred_output_key(item)])
                       for item in decoded_qkv]
            q = q[0, 0, :, 0:64]
            k = k[0, 0, :, 0:64]
            v = v[0, 0, :, 0:64]
            head_outputs = []
            call_timings = []
            for group, (start0, start1) in enumerate(((0, 256), (512, 768),
                                                       (1024, 1280))):
                q0 = q[start0:start0 + 256]
                if group < 2:
                    q1 = q[start1:start1 + 256]
                    valid1 = 256
                else:
                    q1 = np.zeros((256, 64), dtype=np.int8)
                    q1[:90] = q[1280:1370]
                    valid1 = 90
                logical = [
                    {"input": q0[None, None]},
                    {"input": k.T[None, None]},
                    {"input": v[None, None]},
                    {"input": q1[None, None]},
                ]
                npz2bin.read_cfg(str(args.attention_cfg).removesuffix("_cfg.txt"))
                packed = [np.ascontiguousarray(x).reshape(-1).view(np.uint8)
                          for x in npz2bin.read_npz_dict(
                              createBF16TensorFromDict, "input", logical)]
                decoded_attention, timing = run(
                    attention, packed, args.attention_cfg)
                call_timings.append(timing)
                head_outputs.append(np.asarray(decoded_attention[0][
                    preferred_output_key(decoded_attention[0])])[:, :, :256])
                head_outputs.append(np.asarray(decoded_attention[1][
                    preferred_output_key(decoded_attention[1])])[:, :, :valid1])
            combined = np.concatenate(head_outputs, axis=2)
            outputs.append(combined)
            attention_runs.append(call_timings)
    finally:
        waiter.close()

    first = outputs[0]
    result = {
        "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": sha256_array(linked),
        "static_h2c_write_count": 2,
        "static_reloads_during_pipeline": 0,
        "load_ms": load_ms,
        "qkv": qkv_runs,
        "attention": attention_runs,
        "output_shape": list(first.shape),
        "output_sha256": sha256_array(first),
        "all_repeats_exact": all(np.array_equal(first, item)
                                 for item in outputs[1:]),
    }
    golden_path = case_dir / "golden_fused_fp.npz"
    if golden_path.is_file():
        golden = np.load(golden_path)
        reference = np.concatenate(
            [golden[f"output{i}_bf16"] for i in range(6)], axis=1)
        result["fused_fp_metrics"] = tensor_metrics(first[:, 0], reference)
    np.savez(case_dir / "resident_pipeline_output.npz", output_bf16=first)
    (case_dir / "resident_pipeline_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PIPELINE_SUMMARY=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
