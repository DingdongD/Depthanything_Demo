#!/usr/bin/env python3
"""Switch among decoder Conv kernels while keeping one resident bank loaded."""

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
    parser.add_argument("--probe", action="append", required=True)
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
        merge_2ddr, preferred_output_key, reg_write, sha256_array,
        split_2ddr, tensor_metrics,
    )

    npz2bin.read_yaml([
        str(runtime_dir / "arch_16_mono.yaml"),
        str(runtime_dir / "arch_256_mono.yaml"),
    ])
    manifest = json.loads(args.manifest.read_text())
    records = {item["name"]: item for item in manifest["cases"]}
    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    started = time.perf_counter()
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    load_ms = (time.perf_counter() - started) * 1000.0

    waiter = EventWaiter()
    waiter.drain()
    results = []
    try:
        for name in args.probe:
            record = records[name]
            probe_dir = case_dir / ("board_" + name)
            cfg_path = probe_dir / (name + "_cfg.txt")
            npz2bin.read_cfg(str(cfg_path).removesuffix("_cfg.txt"))
            with np.load(probe_dir / "logical_input0.npz", allow_pickle=False) as archive:
                logical = {"input": np.ascontiguousarray(archive["input"])}
            values = npz2bin.read_npz_dict(
                createBF16TensorFromDict, "input", [logical]
            )
            packed = np.ascontiguousarray(values[0]).reshape(-1).view(np.uint8)
            configure_npu(fpgaDma, record)
            reg_write(fpgaDma, 0x00, 0x3F)
            waiter.drain()
            reg_write(fpgaDma, 0x34, 1)
            h2c_start = time.perf_counter()
            for bank, half in enumerate(split_2ddr(packed)):
                tensor = record["inputs"][0]
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
                export_npz_allow_nonfinite, "output", physical
            )
            value = np.asarray(decoded[0][preferred_output_key(decoded[0])])
            case = json.loads((probe_dir / "case.json").read_text())
            if case.get("output_scale") is not None:
                value = value.astype(np.float32) * float(case["output_scale"])
            golden = np.load(probe_dir / (name + "_golden.npy"))
            np.save(probe_dir / (name + "_board.npy"), value)
            results.append({
                "name": name, "event": int(event), "h2c_ms": h2c_ms,
                "npu_ms": npu_ms, "c2h_ms": c2h_ms,
                "shape": list(value.shape), "finite": bool(np.isfinite(value).all()),
                "metrics": tensor_metrics(value, golden),
            })
    finally:
        waiter.close()

    summary = {
        "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": sha256_array(linked),
        "static_h2c_write_count": 2,
        "static_reloads_during_suite": 0,
        "load_ms": load_ms,
        "cases": results,
    }
    (case_dir / "decoder_suite_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("DECODER_SUITE_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
