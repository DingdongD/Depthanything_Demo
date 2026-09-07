#!/usr/bin/env python3
"""C++ mapped-BAR transport for the DepthAnything U250 resident bank.

The NPU programs remain independent compiler-qualified BINs.  This module
places independent invocations in disjoint feature-memory slots so their
inputs can be uploaded in one DMA batch, their programs can be submitted by
one C++ ``run_npu_chain`` call, and their outputs can be downloaded in one
DMA batch.  Static instructions and weights are loaded only once.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


DDR_BASES = (0x0, 0x400000000)
ADDRESS_UNIT_BYTES_PER_BANK = 128
_RUNTIME_CACHE: dict[tuple[str, bool], "CppMappedRuntime"] = {}


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def load_fpga_dma_batch(path: Path | None = None) -> Any:
    """Load the ABI-suffixed fpgaDmaBatch extension, optionally by path."""
    if path is None:
        import fpgaDmaBatch  # type: ignore
        return fpgaDmaBatch
    resolved = path.resolve()
    if resolved.is_dir():
        matches = sorted(resolved.glob("fpgaDmaBatch*.so"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one fpgaDmaBatch extension in {resolved}, got {matches}"
            )
        resolved = matches[0]
    spec = importlib.util.spec_from_file_location("fpgaDmaBatch", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fpgaDmaBatch from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fpgaDmaBatch"] = module
    spec.loader.exec_module(module)
    return module


def split_combined_ddr(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    if value.size % 256:
        raise ValueError(f"combined DS buffer size {value.size} is not 256 aligned")
    rows = value.reshape(-1, 128)
    return (np.ascontiguousarray(rows[0::2].reshape(-1)),
            np.ascontiguousarray(rows[1::2].reshape(-1)))


def merge_combined_ddr(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    left = np.ascontiguousarray(even).reshape(-1).view(np.uint8)
    right = np.ascontiguousarray(odd).reshape(-1).view(np.uint8)
    if left.size != right.size or left.size % 128:
        raise ValueError("DDR bank halves must be equal and 128-byte aligned")
    result = np.empty((left.size // 128 * 2, 128), dtype=np.uint8)
    result[0::2] = left.reshape(-1, 128)
    result[1::2] = right.reshape(-1, 128)
    return result.reshape(-1)


def record_span_per_bank(record: dict) -> int:
    """Return the exact per-bank FM address span used by one compiled case."""
    end = 0
    for tensor in record["inputs"] + record["outputs"]:
        combined_bytes = int(tensor["size_per_bank"])
        if combined_bytes % 2:
            raise ValueError(f"{record['name']}: odd combined tensor size")
        end = max(
            end,
            int(tensor["address"]) * ADDRESS_UNIT_BYTES_PER_BANK
            + combined_bytes // 2,
        )
    return _align(end, 4096)


class CppMappedRuntime:
    """Resident mapped-BAR runtime backed by CompletionFormer's C++ bridge."""

    def __init__(self, manifest: dict, extension: Any, *, safe_dma: bool = True):
        self.manifest = manifest
        self.transport = extension.DmaBatch()
        self.safe_dma = bool(safe_dma)
        combined_workspace = int(manifest.get("shared_fm_workspace_bytes") or 0)
        if combined_workspace <= 0 or combined_workspace % 2:
            raise ValueError("resident manifest has no valid shared FM workspace")
        self.workspace_bytes_per_bank = combined_workspace // 2
        self.groups = 0
        self.dispatches = 0
        self.h2c_seconds = 0.0
        self.c2h_seconds = 0.0
        self.loaded_bank_sha256: str | None = None

    def _h2c(self, requests: list[tuple[int, int, np.ndarray]]) -> None:
        method = (self.transport.h2c_batch_safe if self.safe_dma
                  else self.transport.h2c_batch)
        method(requests)

    def _c2h(self, requests: list[tuple[int, int, int]]) -> list[np.ndarray]:
        method = (self.transport.c2h_batch_safe if self.safe_dma
                  else self.transport.c2h_batch)
        return [np.ascontiguousarray(value) for value in method(requests)]

    def load_bank(self, combined: np.ndarray) -> float:
        halves = split_combined_ddr(combined)
        started = time.perf_counter()
        self._h2c([(bank, DDR_BASES[bank], halves[bank]) for bank in range(2)])
        return (time.perf_counter() - started) * 1000.0

    def ensure_bank(self, combined: np.ndarray, sha256: str) -> tuple[float, bool]:
        """Load one bank image once and reject accidental in-process replacement."""
        if self.loaded_bank_sha256 is not None:
            if self.loaded_bank_sha256 != sha256:
                raise RuntimeError(
                    "resident runtime cannot replace its live bank image: "
                    f"{self.loaded_bank_sha256} != {sha256}"
                )
            return 0.0, True
        elapsed = self.load_bank(combined)
        self.loaded_bank_sha256 = sha256
        return elapsed, False

    def max_group_size(self, records: list[dict]) -> int:
        if not records:
            return 0
        stride = max(record_span_per_bank(record) for record in records)
        return self.workspace_bytes_per_bank // stride

    def run_group(
        self,
        records: list[dict],
        packed_calls: list[list[np.ndarray]],
        upload_masks: list[list[bool]],
        timeout_ms: int,
    ) -> tuple[list[list[np.ndarray]], dict]:
        """Execute independent calls in disjoint FM slots as one submission."""
        if not records or len(records) != len(packed_calls):
            raise ValueError("records and packed calls must be non-empty and aligned")
        if len(upload_masks) != len(records):
            raise ValueError("upload masks do not match grouped calls")
        stride = max(record_span_per_bank(record) for record in records)
        if stride * len(records) > self.workspace_bytes_per_bank:
            raise ValueError(
                f"group needs {stride * len(records)} bytes per bank, only "
                f"{self.workspace_bytes_per_bank} are reserved"
            )

        h2c_requests: list[tuple[int, int, np.ndarray]] = []
        programs = []
        output_requests: list[tuple[int, int, int]] = []
        output_plan: list[int] = []
        uploaded_bytes = 0
        downloaded_bytes = 0
        skipped_bytes = 0
        for slot, (record, packed, mask) in enumerate(
                zip(records, packed_calls, upload_masks)):
            if len(packed) != len(record["inputs"]) or len(mask) != len(packed):
                raise ValueError(f"{record['name']}: grouped input count mismatch")
            slot_bytes = slot * stride
            slot_units = slot_bytes // ADDRESS_UNIT_BYTES_PER_BANK
            for tensor, combined, upload in zip(record["inputs"], packed, mask):
                halves = split_combined_ddr(combined)
                if not upload:
                    skipped_bytes += int(combined.size)
                    continue
                for bank, half in enumerate(halves):
                    address = (
                        (int(record["base_addresses"][4])
                         + int(tensor["address"]) + slot_units)
                        * ADDRESS_UNIT_BYTES_PER_BANK + DDR_BASES[bank]
                    )
                    h2c_requests.append((bank, address, half))
                    uploaded_bytes += int(half.size)
            bases = [int(value) for value in record["base_addresses"]]
            bases[4] += slot_units
            programs.append({
                "stage_id": record["name"],
                "base_addresses": bases,
                "isa_ranges": [int(value) for value in record["isa_ranges"]],
            })
            output_plan.append(len(record["outputs"]))
            for tensor in record["outputs"]:
                combined_size = int(tensor["size_per_bank"])
                if combined_size % 2:
                    raise ValueError(f"{record['name']}: odd output byte count")
                half_size = combined_size // 2
                for bank in range(2):
                    address = (
                        (int(record["base_addresses"][4])
                         + int(tensor["address"]) + slot_units)
                        * ADDRESS_UNIT_BYTES_PER_BANK + DDR_BASES[bank]
                    )
                    output_requests.append((bank, address, half_size))
                    downloaded_bytes += half_size

        h2c_started = time.perf_counter()
        if h2c_requests:
            self._h2c(h2c_requests)
        h2c_ms = (time.perf_counter() - h2c_started) * 1000.0
        npu_seconds = [float(value) for value in
                       self.transport.run_npu_chain(programs, int(timeout_ms))]
        c2h_started = time.perf_counter()
        raw = self._c2h(output_requests)
        c2h_ms = (time.perf_counter() - c2h_started) * 1000.0

        results: list[list[np.ndarray]] = []
        cursor = 0
        for output_count in output_plan:
            values = []
            for _ in range(output_count):
                values.append(merge_combined_ddr(raw[cursor], raw[cursor + 1]))
                cursor += 2
            results.append(values)
        if cursor != len(raw):
            raise RuntimeError("C2H result count does not match grouped output plan")
        self.groups += 1
        self.dispatches += len(records)
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        return results, {
            "submission_group_size": len(records),
            "workspace_stride_per_bank": stride,
            "h2c_ms": h2c_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "c2h_ms": c2h_ms,
            "h2c_bytes": uploaded_bytes,
            "c2h_bytes": downloaded_bytes,
            "h2c_skipped_bytes": skipped_bytes,
        }

    def stats(self) -> dict:
        result = dict(self.transport.stats())
        result.update({
            "python_submission_groups": self.groups,
            "physical_npu_dispatches": self.dispatches,
            "shared_fm_bytes_per_bank": self.workspace_bytes_per_bank,
            "safe_dma": self.safe_dma,
            "mapped_bar": True,
            "locked_host_buffers": True,
        })
        return result

    def reset_frame_stats(self) -> None:
        self.groups = 0
        self.dispatches = 0
        self.h2c_seconds = 0.0
        self.c2h_seconds = 0.0
        self.transport.reset_stats()


def get_cached_cpp_runtime(
    cache_key: str, manifest: dict, extension_path: Path | None,
    *, safe_dma: bool,
) -> tuple[CppMappedRuntime, bool]:
    """Return a process-resident transport, preserving mmap/fds/pinned buffers."""
    key = (str(cache_key), bool(safe_dma))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key], True
    extension = load_fpga_dma_batch(extension_path)
    runtime = CppMappedRuntime(manifest, extension, safe_dma=safe_dma)
    _RUNTIME_CACHE[key] = runtime
    return runtime, False
