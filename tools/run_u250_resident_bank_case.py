#!/usr/bin/env python3
"""Run one relocated case from a once-loaded multi-kernel resident bank."""

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
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(case_dir))
    sys.path.insert(1, str(runtime_dir))
    import fpgaDma  # type: ignore
    import npz2bin  # type: ignore
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
    npz2bin.read_cfg(str(args.cfg).removesuffix("_cfg.txt"))
    codec_cfg = dict(npz2bin.get_cfg_dict())
    manifest = json.loads(args.manifest.read_text())
    record = next(item for item in manifest["cases"]
                  if item["name"] == args.case_name)
    if len(args.input) != len(record["inputs"]):
        raise ValueError("input count does not match resident record")

    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    started = time.perf_counter()
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    configure_npu(fpgaDma, record)
    load_ms = (time.perf_counter() - started) * 1000.0

    packed = [fpgaDma.file2np(str(path), os.path.getsize(path))
              for path in args.input]
    waiter = EventWaiter()
    waiter.drain()
    outputs = []
    timings = []
    try:
        for _ in range(args.repeats):
            reg_write(fpgaDma, 0x00, 0x3F)
            waiter.drain()
            reg_write(fpgaDma, 0x34, 1)
            h2c_start = time.perf_counter()
            for tensor, combined in zip(record["inputs"], packed):
                for bank, half in enumerate(split_2ddr(combined)):
                    address = ((record["base_addresses"][4] + tensor["address"])
                               * 0x80 + DDR_BASES[bank])
                    fpgaDma.np2card(H2C_DEVICES[bank], address, half.size, half)
            h2c_ms = (time.perf_counter() - h2c_start) * 1000.0
            reg_write(fpgaDma, 0x34, 2)
            npu_start = time.perf_counter()
            event = waiter.wait(args.timeout_ms)
            npu_ms = (time.perf_counter() - npu_start) * 1000.0
            clear_interrupt(fpgaDma)
            physical = []
            c2h_start = time.perf_counter()
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
            outputs.append(decoded)
            timings.append({"event": int(event), "h2c_ms": h2c_ms,
                            "npu_ms": npu_ms,
                            "c2h_ms": c2h_ms})
    finally:
        waiter.close()

    result = {
        "case": args.case_name,
        "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": sha256_array(linked),
        "static_h2c_write_count": 2,
        "static_reloads": 0,
        "load_ms": load_ms,
        "timings": timings,
        "outputs": [],
    }
    for index, item in enumerate(outputs[0]):
        key = preferred_output_key(item)
        value = np.asarray(item[key])
        entry = {
            "index": index,
            "key": key,
            "shape": list(value.shape),
            "finite": bool(np.isfinite(value).all()),
            "sha256": sha256_array(value),
            "all_repeats_exact": all(np.array_equal(
                value, run[index][preferred_output_key(run[index])]
            ) for run in outputs[1:]),
        }
        if args.golden is not None:
            entry["metrics"] = tensor_metrics(value, np.load(args.golden))
        result["outputs"].append(entry)
    (case_dir / (args.case_name + "_resident_summary.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESIDENT_CASE_SUMMARY=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
