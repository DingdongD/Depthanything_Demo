#!/usr/bin/env python3
"""Compare one multi-output U250 program with independent reference programs.

The comparison deliberately stays in the compiler's physical two-bank tensor
layout.  This makes it sensitive to arithmetic, output order, padding, and DDR
placement without introducing vendor or native host codec differences.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


DDR_BASES = (0, 0x400000000)
ADDRESS_UNIT_BYTES_PER_BANK = 128


def split_combined(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    if value.size % 256:
        raise ValueError("combined buffer must be 256-byte aligned")
    rows = value.reshape(-1, 128)
    return (np.ascontiguousarray(rows[0::2].reshape(-1)),
            np.ascontiguousarray(rows[1::2].reshape(-1)))


def merge_banks(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    left = np.ascontiguousarray(even, dtype=np.uint8).reshape(-1)
    right = np.ascontiguousarray(odd, dtype=np.uint8).reshape(-1)
    if left.size != right.size or left.size % 128:
        raise ValueError("bank buffers must be equal and 128-byte aligned")
    result = np.empty((left.size // 128 * 2, 128), dtype=np.uint8)
    result[0::2] = left.reshape(-1, 128)
    result[1::2] = right.reshape(-1, 128)
    return result.reshape(-1)


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("fpgaDmaBatch", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_addresses(record: dict, tensor: dict) -> tuple[int, int]:
    relative = (int(record["base_addresses"][4]) + int(tensor["address"]))
    relative *= ADDRESS_UNIT_BYTES_PER_BANK
    return relative, relative + DDR_BASES[1]


def program(record: dict) -> dict:
    return {
        "stage_id": record["name"],
        "base_addresses": [int(value) for value in record["base_addresses"]],
        "isa_ranges": [int(value) for value in record["isa_ranges"]],
    }


def unpack_logical(codec, combined: np.ndarray, tensor: dict, index: int
                   ) -> np.ndarray:
    """Decode valid lanes while leaving physical padding out of precision gates."""
    dims = [int(value) for value in tensor["dims"]]
    bitdepth = int(tensor["bitdepth"])
    if tensor["layout"] != "NDWC":
        raise ValueError("logical validator currently supports NDWC outputs")
    c_align = dims[1] * math.ceil(dims[3] / 16) * (bitdepth // 8)
    w_align = math.ceil(dims[2] / 16) * c_align
    descriptor = {
        "layout": "NDWC",
        "dims": dims,
        "bitdepth": bitdepth,
        "c_align": c_align,
        "w_align": w_align,
        "combined_bytes": int(tensor["size_per_bank"]),
        "direction": "output",
        "index": index,
        "matrix_role": "output",
    }
    return np.ascontiguousarray(codec.unpack_tensor(
        *split_combined(combined), descriptor
    ))


def run_case(transport, record: dict, inputs: list[np.ndarray], timeout_ms: int
             ) -> tuple[list[np.ndarray], dict]:
    h2c = []
    for tensor, combined in zip(record["inputs"], inputs):
        halves = split_combined(combined)
        addresses = tensor_addresses(record, tensor)
        h2c.extend((bank, addresses[bank], halves[bank]) for bank in range(2))
    c2h = []
    for tensor in record["outputs"]:
        combined_bytes = int(tensor["size_per_bank"])
        if combined_bytes % 2:
            raise ValueError(f"{record['name']}: odd output size")
        addresses = tensor_addresses(record, tensor)
        c2h.extend((bank, addresses[bank], combined_bytes // 2)
                   for bank in range(2))
    result = dict(transport.run_resident_transaction(
        h2c, [program(record)], c2h, timeout_ms, True))
    raw = [np.ascontiguousarray(value, dtype=np.uint8)
           for value in result["outputs"]]
    outputs = [merge_banks(raw[index], raw[index + 1])
               for index in range(0, len(raw), 2)]
    return outputs, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--seed", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    manifest = json.loads(args.manifest.read_text())
    records = {record["name"]: record for record in manifest["cases"]}
    candidate = records[args.candidate]
    references = [records[name] for name in args.reference]
    if len(candidate["outputs"]) != len(references):
        raise ValueError("candidate output count must match reference count")
    if any(len(record["inputs"]) != 1 or len(record["outputs"]) != 1
           for record in references) or len(candidate["inputs"]) != 1:
        raise ValueError("validator requires one-input, one-output references")
    input_bytes = int(candidate["inputs"][0]["size_per_bank"])
    if any(int(record["inputs"][0]["size_per_bank"]) != input_bytes
           for record in references):
        raise ValueError("candidate and references have different input extents")

    extension = load_extension(args.extension)
    transport = extension.DmaBatch()
    bank = np.fromfile(args.manifest.parent / manifest["bank_file"], dtype=np.uint8)
    bank_halves = split_combined(bank)
    transport.h2c_batch_safe([
        (index, DDR_BASES[index], half)
        for index, half in enumerate(bank_halves)
    ])

    rng = np.random.default_rng(args.seed)
    logical_input = rng.integers(0, 256, size=input_bytes, dtype=np.uint8)
    comparisons = []
    all_exact = True
    all_physical_exact = True
    candidate_hashes = []
    for repeat in range(args.repeats):
        candidate_outputs, candidate_timing = run_case(
            transport, candidate, [logical_input], args.timeout_ms)
        candidate_hashes.append([
            hashlib.sha256(value).hexdigest() for value in candidate_outputs
        ])
        for index, reference in enumerate(references):
            reference_outputs, reference_timing = run_case(
                transport, reference, [logical_input], args.timeout_ms)
            left = candidate_outputs[index]
            right = reference_outputs[0]
            physical_exact = bool(np.array_equal(left, right))
            physical_mismatch = int(np.count_nonzero(left != right))
            candidate_logical = unpack_logical(
                extension.DmaBatch, left, candidate["outputs"][index], index
            )
            reference_logical = unpack_logical(
                extension.DmaBatch, right, reference["outputs"][0], 0
            )
            exact = bool(np.array_equal(candidate_logical, reference_logical))
            mismatch = int(np.count_nonzero(
                candidate_logical.view(np.uint32) != reference_logical.view(np.uint32)
            ))
            absolute = np.abs(candidate_logical.astype(np.float64)
                              - reference_logical.astype(np.float64))
            all_exact &= exact
            all_physical_exact &= physical_exact
            comparisons.append({
                "repeat": repeat,
                "candidate_output": index,
                "reference": reference["name"],
                "exact": exact,
                "mismatched_elements": mismatch,
                "logical_elements": int(candidate_logical.size),
                "max_abs": float(absolute.max(initial=0.0)),
                "physical_exact": physical_exact,
                "physical_mismatched_bytes": physical_mismatch,
                "physical_bytes": int(left.size),
                "candidate_sha256": hashlib.sha256(left).hexdigest(),
                "reference_sha256": hashlib.sha256(right).hexdigest(),
                "candidate_npu_ms": float(candidate_timing["npu_seconds"][0]) * 1000,
                "reference_npu_ms": float(reference_timing["npu_seconds"][0]) * 1000,
            })

    report = {
        "schema_version": 1,
        "manifest_sha256": manifest["bank_sha256"],
        "candidate": candidate["name"],
        "references": [record["name"] for record in references],
        "seed": args.seed,
        "repeats": args.repeats,
        "input_sha256": hashlib.sha256(logical_input).hexdigest(),
        "all_exact": all_exact,
        "all_physical_exact": all_physical_exact,
        "candidate_repeat_stable": all(
            hashes == candidate_hashes[0] for hashes in candidate_hashes[1:]
        ),
        "comparisons": comparisons,
        "transport_stats": dict(transport.stats()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("U250_MULTIOUTPUT_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
