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
import hashlib
from dataclasses import asdict
import json
import math
import re
from pathlib import Path
import sys
import time
from typing import Any, Union

import numpy as np

if __package__:
    from .u250_layout_descriptors import TensorLayoutDescriptor
else:
    from u250_layout_descriptors import TensorLayoutDescriptor


DDR_BASES = (0x0, 0x400000000)
ADDRESS_UNIT_BYTES_PER_BANK = 128
_RUNTIME_CACHE: dict[tuple[str, bool], "CppMappedRuntime"] = {}
_EXTENSION_CACHE: dict[Path, Any] = {}
BankPair = tuple[np.ndarray, np.ndarray]
PhysicalTensor = Union[np.ndarray, BankPair]


class LayoutCodecSelection:
    """CPU-only qualification, case-direction routing, and tensor accounting."""

    def __init__(self, mode: str, manifest_sha256: str, report_path: Path | None,
                 descriptors: dict[str, dict[str, list[TensorLayoutDescriptor]]]):
        if mode not in {"vendor", "auto", "native"}:
            raise ValueError(f"invalid layout codec mode: {mode}")
        self.mode = mode
        self.descriptors = descriptors
        for name, directions in descriptors.items():
            for direction, values in directions.items():
                for desc in values:
                    if desc.direction != direction:
                        raise RuntimeError(f"{name}: {direction} {desc.index} descriptor "
                                           f"{desc.identity()}: active direction mismatch")
        self._unique = {d.identity(): d for directions in descriptors.values()
                        for values in directions.values() for d in values}
        self._reasons: dict[str, str] = {}
        self._routes: dict[tuple[str, str], bool] = {}
        self._prepared_type = None
        self.report_extension_sha256: str | None = None
        self.extension_path: str | None = None
        self.extension_sha256: str | None = None
        self.fallback_reasons: dict[str, str] = {}
        self.reset_stats()
        if mode != "vendor":
            entries, self.report_extension_sha256, error = self._read_report(
                report_path, manifest_sha256)
            if error and mode == "native" and not self._unique:
                raise RuntimeError(error)
            for identity, desc in self._unique.items():
                reason = error or self._qualification_error(entries.get(identity), desc)
                if reason:
                    self._reasons[identity] = reason
        self._select_routes()

    @staticmethod
    def _read_report(path: Path | None, manifest_sha256: str
                     ) -> tuple[dict, str | None, str | None]:
        if path is None:
            return {}, None, "qualification report is required"
        try:
            report = json.loads(path.read_bytes())
            if not isinstance(report, dict):
                return {}, None, "invalid qualification report object"
            if report.get("manifest_sha256") != manifest_sha256:
                return {}, None, "qualification report manifest_sha256 mismatch"
            digest = report.get("extension_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                return {}, None, "extension provenance: qualification report extension_sha256 is required and must be a SHA-256"
            values = report.get("descriptors")
            if not isinstance(values, list):
                return {}, digest, "invalid qualification report descriptors"
            entries = {}
            for entry in values:
                if not isinstance(entry, dict) or not isinstance(entry.get("identity"), str):
                    return {}, digest, "invalid qualification report descriptor identity"
                identity = entry["identity"]
                if identity in entries:
                    return {}, digest, f"duplicate report descriptor {identity}"
                entries[identity] = entry
            return entries, digest, None
        except (OSError, ValueError) as error:
            return {}, None, f"cannot read qualification report: {error}"

    def bind_extension(self, path: Path | None, digest: str | None,
                       error: str | None = None) -> None:
        """Bind eligibility to file bytes before loading code or opening DMA."""
        self.extension_path = str(path) if path is not None else None
        self.extension_sha256 = digest
        if self.mode == "vendor":
            return
        reason = error
        if reason is None and digest != self.report_extension_sha256:
            reason = "qualification report extension_sha256 mismatch"
        if reason:
            reason = "extension provenance: " + reason
            self._reasons.update({identity: reason for identity in self._unique})
            if self.mode == "native" and not self._unique:
                raise RuntimeError(reason)
        self._select_routes()

    @staticmethod
    def _qualification_error(entry: dict | None, desc: TensorLayoutDescriptor) -> str | None:
        if entry is None:
            return "not native_exact"
        expected = asdict(desc)
        expected.pop("index")  # Identities are shared across cfg tensor indices.
        expected["dims"] = list(desc.dims)
        actual = {key: entry.get(key) for key in expected}
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            return "report descriptor fields/direction do not match identity"
        for flag in ("native_exact", "pack_exact", "unpack_exact", "production_enabled"):
            if entry.get(flag) is not True:
                return f"not {flag}"
        benchmark = entry.get("benchmark")
        operation = "pack" if desc.direction == "input" else "unpack"
        if (not isinstance(benchmark, dict)
                or benchmark.get("production_enabled") is not True
                or benchmark.get("operation") != operation):
            return "benchmark is not production_enabled for required direction"
        native, vendor = (benchmark.get(f"{backend}_median_ms")
                          for backend in ("native", "vendor"))
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (native, vendor)) or not 0 <= native < vendor):
            return "benchmark native median is not strictly faster"
        return None

    def _select_routes(self) -> None:
        self.fallback_reasons = dict(self._reasons) if self.mode == "auto" else {}
        for name, directions in self.descriptors.items():
            for direction, values in directions.items():
                rejected = [d for d in values if d.identity() in self._reasons]
                if rejected and self.mode == "native":
                    desc = rejected[0]
                    raise RuntimeError(
                        f"{name}: {direction} {desc.index} descriptor {desc.identity()}: "
                        f"{self._reasons[desc.identity()]}"
                    )
                self._routes[name, direction] = self.mode != "vendor" and not rejected
                if rejected and self.mode == "auto":
                    cause = rejected[0].identity()
                    for desc in values:
                        self.fallback_reasons.setdefault(
                            desc.identity(), f"{name} {direction} batch fallback: "
                            f"{cause}: {self._reasons[cause]}"
                        )

    def prepare(self, codec_type: Any) -> None:
        """Validate static APIs before constructing or reusing a DMA transport."""
        if (self.mode == "vendor"
                or self._prepared_type is codec_type and codec_type is not None):
            return
        if self.extension_sha256 is None:
            self.bind_extension(None, None, "loaded extension SHA-256 is unavailable")
        available = all(callable(getattr(codec_type, name, None))
                        for name in ("validate_descriptor", "pack_tensor", "unpack_tensor"))
        for identity, desc in self._unique.items():
            if identity in self._reasons:
                continue
            if not available:
                self._reasons[identity] = "native codec API requires a compatible cpp_mapped extension"
                continue
            try:
                codec_type.validate_descriptor(asdict(desc))
            except Exception as error:
                self._reasons[identity] = f"native descriptor validation failed: {error}"
        self._select_routes()
        self._prepared_type = codec_type

    def native_for(self, name: str, direction: str) -> bool:
        if self.mode == "vendor":
            return False
        if (name, direction) not in self._routes:
            raise RuntimeError(f"{name}: {direction} was not included in codec preflight")
        return self._routes[name, direction]

    def require_native(self, desc: TensorLayoutDescriptor, operation: str) -> None:
        identity = desc.identity()
        expected_direction = "input" if operation == "pack" else "output"
        if (self.mode == "vendor" or self._prepared_type is None
                or identity not in self._unique or identity in self._reasons
                or desc.direction != expected_direction):
            raise RuntimeError(f"{desc.direction} {desc.index} descriptor {identity}: "
                               f"not qualified for native {operation}")

    def reset_stats(self) -> None:
        self._totals = {f"{backend}_{operation}_{field}": 0.0 if field == "ms" else 0
                        for backend in ("native", "vendor")
                        for operation in ("pack", "unpack")
                        for field in ("calls", "logical_bytes", "physical_bytes", "ms")}
        self._by_layout_dtype: dict[str, dict] = {}

    def record(self, backend: str, operation: str, descriptors: list[TensorLayoutDescriptor],
               logical_bytes: list[int], elapsed_ms: float) -> None:
        if self.mode == "native" and backend == "vendor":
            raise RuntimeError("native mode requires zero vendor codec calls")
        physical_bytes = sum(d.combined_bytes for d in descriptors)
        for field, value in (("calls", len(descriptors)), ("logical_bytes", sum(logical_bytes)),
                             ("physical_bytes", physical_bytes), ("ms", elapsed_ms)):
            self._totals[f"{backend}_{operation}_{field}"] += value
        for desc, size in zip(descriptors, logical_bytes):
            dtype = "INT8" if desc.bitdepth == 8 else "BF16"
            group = self._by_layout_dtype.setdefault(f"{desc.layout}_{dtype}", {})
            for field, value in (("calls", 1), ("logical_bytes", size),
                                 ("physical_bytes", desc.combined_bytes),
                                 ("ms", elapsed_ms * desc.combined_bytes / physical_bytes)):
                key = f"{backend}_{operation}_{field}"
                group[key] = group.get(key, 0) + value

    def stats(self) -> dict:
        if self.mode == "native" and any(self._totals[f"vendor_{op}_calls"]
                                          for op in ("pack", "unpack")):
            raise RuntimeError("native mode requires zero vendor codec calls")
        return {"layout_codec": self.mode, **self._totals,
                "qualification_extension_sha256": self.report_extension_sha256,
                **{f"codec_{op}_ms_total": sum(self._totals[f"{backend}_{op}_ms"]
                                              for backend in ("native", "vendor"))
                   for op in ("pack", "unpack")},
                "codec_by_layout_dtype": {
                    key: dict(value) for key, value in self._by_layout_dtype.items()},
                "codec_timing_note": (
                    "pack timing includes contiguous input preparation for both backends; "
                    "native time is measured per tensor; vendor time is measured per cfg "
                    "direction and apportioned by physical bytes for layout/dtype subtotals"
                ),
                "fallback_reasons": dict(self.fallback_reasons)}


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def resolve_fpga_dma_batch(path: Path | None = None) -> Path:
    """Resolve the exact extension file without importing/constructing it."""
    if path is None:
        loaded = sys.modules.get("fpgaDmaBatch")
        if loaded is not None:
            path = Path(loaded.__file__)
        else:
            spec = importlib.util.find_spec("fpgaDmaBatch")
            if spec is None or spec.origin is None:
                raise ImportError("cannot find fpgaDmaBatch extension")
            path = Path(spec.origin)
    resolved = path.resolve()
    if resolved.is_dir():
        matches = sorted(resolved.glob("fpgaDmaBatch*.so"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one fpgaDmaBatch extension in {resolved}, got {matches}"
            )
        resolved = matches[0].resolve()
    return resolved


def _extension_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loaded_extension_identity(extension: Any) -> tuple[Path | None, str | None]:
    path = getattr(extension, "__file__", None)
    return (Path(path).resolve() if path else None,
            getattr(extension, "_u250_extension_sha256", None))


def _check_extension(selection: LayoutCodecSelection, requested: Path | None,
                     loaded: tuple[Path | None, str | None] | None = None) -> None:
    if selection.mode == "vendor":
        return
    path = digest = None
    error = None
    try:
        path = resolve_fpga_dma_batch(requested)
        digest = _extension_sha256(path)
        if loaded is None:
            module = _EXTENSION_CACHE.get(path)
            current = sys.modules.get("fpgaDmaBatch")
            if module is None and current is not None and _loaded_extension_identity(current)[0] == path:
                module = current
            if module is not None:
                loaded = _loaded_extension_identity(module)
        if loaded is not None:
            loaded_path, loaded_digest = loaded
            if loaded_path != path:
                error = "requested extension path changed from loaded extension"
            elif loaded_digest is None:
                error = "loaded extension SHA-256 is unavailable"
            elif loaded_digest != digest:
                error = "loaded extension file SHA-256 changed"
    except (OSError, ImportError, RuntimeError, AttributeError, ValueError):
        error = "cannot resolve or hash requested extension"
    selection.bind_extension(path, digest, error)


def load_fpga_dma_batch(path: Path | None = None) -> Any:
    """Record file identity before loading; retain it across Python/DSO caches."""
    resolved = resolve_fpga_dma_batch(path)
    if resolved in _EXTENSION_CACHE:
        return _EXTENSION_CACHE[resolved]
    current = sys.modules.get("fpgaDmaBatch")
    if current is not None and _loaded_extension_identity(current)[0] == resolved:
        # An externally imported module has no provable load-time digest. Never
        # assign one from today's file bytes: the OS may still hold older code.
        _EXTENSION_CACHE[resolved] = current
        return current
    digest = _extension_sha256(resolved)
    spec = importlib.util.spec_from_file_location("fpgaDmaBatch", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load fpgaDmaBatch from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fpgaDmaBatch"] = module
    spec.loader.exec_module(module)
    module._u250_extension_sha256 = digest
    _EXTENSION_CACHE[resolved] = module
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

    def __init__(self, manifest: dict, extension: Any, *, safe_dma: bool = True,
                 codec_selection: LayoutCodecSelection | None = None):
        self.manifest = manifest
        self.codec_type = extension.DmaBatch
        self.codec_selection = codec_selection or LayoutCodecSelection("vendor", "", None, {})
        self.extension_path, self.extension_sha256 = _loaded_extension_identity(extension)
        requested = (Path(self.codec_selection.extension_path)
                     if self.codec_selection.extension_path else self.extension_path)
        _check_extension(self.codec_selection, requested,
                         (self.extension_path, self.extension_sha256))
        self.codec_selection.prepare(self.codec_type)
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
        self.transport = extension.DmaBatch()

    def pack_tensor(self, array: np.ndarray, descriptor: TensorLayoutDescriptor
                    ) -> tuple[np.ndarray, np.ndarray]:
        self.codec_selection.require_native(descriptor, "pack")
        started = time.perf_counter()
        logical = np.ascontiguousarray(array)
        banks = self.codec_type.pack_tensor(logical, asdict(descriptor))
        elapsed = (time.perf_counter() - started) * 1000.0
        self.codec_selection.record("native", "pack", [descriptor], [logical.nbytes], elapsed)
        return banks

    def unpack_tensor(self, even: np.ndarray, odd: np.ndarray,
                      descriptor: TensorLayoutDescriptor) -> np.ndarray:
        self.codec_selection.require_native(descriptor, "unpack")
        started = time.perf_counter()
        value = self.codec_type.unpack_tensor(even, odd, asdict(descriptor))
        elapsed = (time.perf_counter() - started) * 1000.0
        self.codec_selection.record("native", "unpack", [descriptor], [value.nbytes], elapsed)
        return value

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
        packed_calls: list[list[PhysicalTensor]],
        upload_masks: list[list[bool]],
        timeout_ms: int,
    ) -> tuple[list[list[PhysicalTensor]], dict]:
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
        output_plan: list[tuple[int, bool]] = []
        uploaded_bytes = 0
        downloaded_bytes = 0
        skipped_bytes = 0
        for slot, (record, packed, mask) in enumerate(
                zip(records, packed_calls, upload_masks)):
            native_input = self.codec_selection.native_for(record["name"], "input")
            native_output = self.codec_selection.native_for(record["name"], "output")
            if len(packed) != len(record["inputs"]) or len(mask) != len(packed):
                raise ValueError(f"{record['name']}: grouped input count mismatch")
            slot_bytes = slot * stride
            slot_units = slot_bytes // ADDRESS_UNIT_BYTES_PER_BANK
            for tensor, combined, upload in zip(record["inputs"], packed, mask):
                combined_size = int(tensor["size_per_bank"])
                if native_input:
                    if (not isinstance(combined, tuple) or len(combined) != 2
                            or any(not isinstance(half, np.ndarray)
                                   or half.dtype != np.uint8 or half.ndim != 1
                                   or not half.flags.c_contiguous
                                   or half.nbytes != combined_size // 2 for half in combined)):
                        raise ValueError(f"{record['name']}: native input requires exact uint8 bank pair")
                    halves = combined
                else:
                    halves = split_combined_ddr(combined)
                if not upload:
                    skipped_bytes += combined_size
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
            output_plan.append((len(record["outputs"]), native_output))
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

        results: list[list[PhysicalTensor]] = []
        cursor = 0
        if len(raw) != len(output_requests):
            raise RuntimeError("C2H result count does not match grouped output plan")
        for output_count, native_output in output_plan:
            values = []
            for _ in range(output_count):
                banks = raw[cursor], raw[cursor + 1]
                values.append(banks if native_output else merge_combined_ddr(*banks))
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
            **self.codec_selection.stats(),
            "extension_path": str(self.extension_path) if self.extension_path else None,
            "extension_sha256": self.extension_sha256,
            "python_submission_groups": self.groups,
            "physical_npu_dispatches": self.dispatches,
            "shared_fm_bytes_per_bank": self.workspace_bytes_per_bank,
            "safe_dma": self.safe_dma,
            "mapped_bar": True,
            "locked_host_buffers": True,
        })
        return result

    def reset_frame_stats(self) -> None:
        self.codec_selection.reset_stats()
        self.groups = 0
        self.dispatches = 0
        self.h2c_seconds = 0.0
        self.c2h_seconds = 0.0
        self.transport.reset_stats()


def get_cached_cpp_runtime(
    cache_key: str, manifest: dict, extension_path: Path | None,
    *, safe_dma: bool, codec_selection: LayoutCodecSelection | None = None,
) -> tuple[CppMappedRuntime, bool]:
    """Return a process-resident transport, preserving mmap/fds/pinned buffers."""
    key = (str(cache_key), bool(safe_dma))
    selection = codec_selection or LayoutCodecSelection("vendor", "", None, {})
    if key in _RUNTIME_CACHE:
        runtime = _RUNTIME_CACHE[key]
        _check_extension(selection, extension_path,
                         (runtime.extension_path, runtime.extension_sha256))
        selection.prepare(runtime.codec_type)
        runtime.codec_selection = selection
        return runtime, True
    _check_extension(selection, extension_path)
    extension = load_fpga_dma_batch(Path(selection.extension_path)
                                    if selection.extension_path else extension_path)
    runtime = CppMappedRuntime(manifest, extension, safe_dma=safe_dma,
                               codec_selection=selection)
    _RUNTIME_CACHE[key] = runtime
    return runtime, False
