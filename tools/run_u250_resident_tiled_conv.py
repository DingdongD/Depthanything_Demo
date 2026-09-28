#!/usr/bin/env python3
"""Run and stitch one row-tiled Conv from a once-loaded resident U250 bank."""

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
    parser.add_argument("--case-name", required=True,
                        help="full convolution name, for example decoder_conv_24")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--tile-rows", type=int, required=True)
    parser.add_argument("--kernel-height", type=int, choices=(1, 3), required=True)
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
    with np.load(args.input, allow_pickle=False) as archive:
        full_input = np.ascontiguousarray(archive["input"], dtype=np.int8)
    height = full_input.shape[2]
    if height % args.tile_rows:
        raise ValueError("input height is not divisible by tile rows")

    bank_path = case_dir / manifest["bank_file"]
    linked = fpgaDma.file2np(str(bank_path), os.path.getsize(bank_path))
    started = time.perf_counter()
    for bank, half in enumerate(split_2ddr(linked)):
        fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
    reg_write(fpgaDma, 0x2C, 1)
    load_ms = (time.perf_counter() - started) * 1000.0

    waiter = EventWaiter()
    waiter.drain()
    outputs = []
    timings = []
    try:
        tile_count = height // args.tile_rows
        for tile in range(tile_count):
            start = tile * args.tile_rows
            end = start + args.tile_rows
            if args.kernel_height == 1:
                suffix = f"tile{args.tile_rows}"
                logical = full_input[:, :, start:end, :]
            elif tile == 0:
                suffix = "tile_first"
                logical = full_input[:, :, :end + 1, :]
            elif tile == tile_count - 1:
                suffix = "tile_last"
                logical = full_input[:, :, start - 1:end, :]
            else:
                suffix = "tile_middle"
                logical = full_input[:, :, start - 1:end + 1, :]
            name = args.case_name + "_" + suffix
            record = records[name]
            cfg_path = args.input.resolve().parent / (name + "_cfg.txt")
            if not cfg_path.is_file():
                cfg_path = case_dir / (name + "_cfg.txt")
            npz2bin.read_cfg(str(cfg_path).removesuffix("_cfg.txt"))
            packed_values = npz2bin.read_npz_dict(
                createBF16TensorFromDict, "input", [{"input": logical}]
            )
            packed = np.ascontiguousarray(packed_values[0]).reshape(-1).view(np.uint8)

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
            halves = []
            tensor = record["outputs"][0]
            for bank in range(2):
                address = ((record["base_addresses"][4] + tensor["address"])
                           * 0x80 + DDR_BASES[bank])
                halves.append(np.ascontiguousarray(fpgaDma.card2np(
                    C2H_DEVICES[bank], address, tensor["size_per_bank"])))
            c2h_ms = (time.perf_counter() - c2h_start) * 1000.0
            decoded = npz2bin.buffer_to_npz_dict(
                export_npz_allow_nonfinite, "output",
                [merge_2ddr(halves[0], halves[1])],
            )
            outputs.append(np.asarray(decoded[0][preferred_output_key(decoded[0])]))
            timings.append({"tile": tile, "kernel": name, "event": int(event),
                            "h2c_ms": h2c_ms, "npu_ms": npu_ms,
                            "c2h_ms": c2h_ms})
    finally:
        waiter.close()

    stitched = np.concatenate(outputs, axis=2)
    golden = np.load(args.golden)
    result = {
        "case": args.case_name,
        "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": sha256_array(linked),
        "static_h2c_write_count": 2,
        "static_reloads": 0,
        "load_ms": load_ms,
        "tile_count": len(outputs),
        "output_shape": list(stitched.shape),
        "finite": bool(np.isfinite(stitched).all()),
        "metrics": tensor_metrics(stitched, golden),
        "timings": timings,
    }
    np.save(case_dir / (args.case_name + "_stitched.npy"), stitched)
    (case_dir / (args.case_name + "_tiled_summary.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("TILED_CONV_SUMMARY=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
