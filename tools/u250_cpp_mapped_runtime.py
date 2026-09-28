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
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class DeviceTensorHandle:
    """A versioned reference to qualified tensor lanes in shared U250 FM."""

    runtime_id: int
    frame_epoch: int
    handle_id: int
    bank_addresses: tuple[int, int]
    bytes_per_bank: int
    storage_identity: str
    scale: float | None
    owner_group: int
    lifetime: str
    label: str


ResidentInput = Union[PhysicalTensor, DeviceTensorHandle, None]


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
        self._qualification_entries: dict[str, dict] = {}
        self.report_extension_sha256: str | None = None
        self.extension_path: str | None = None
        self.extension_sha256: str | None = None
        self.fallback_reasons: dict[str, str] = {}
        self.reset_stats()
        if mode != "vendor":
            entries, self.report_extension_sha256, error = self._read_report(
                report_path, manifest_sha256)
            self._qualification_entries = entries
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

    def require_native_symmetric_pack(self, desc: TensorLayoutDescriptor) -> None:
        """Authorize a mirrored-cfg pack proven by the codec oracle report."""
        identity = desc.identity()
        entry = self._qualification_entries.get(identity)
        benchmark = entry.get("benchmark") if isinstance(entry, dict) else None
        native = (benchmark.get("native_pack_median_ms")
                  if isinstance(benchmark, dict) else None)
        vendor = (benchmark.get("vendor_pack_median_ms")
                  if isinstance(benchmark, dict) else None)
        if (self.mode == "vendor" or self._prepared_type is None
                or identity not in self._unique or identity in self._reasons
                or desc.direction != "output"
                or not isinstance(entry, dict)
                or entry.get("pack_exact") is not True
                or not isinstance(benchmark, dict)
                or benchmark.get("production_enabled") is not True
                or type(native) not in (int, float)
                or type(vendor) not in (int, float)
                or not math.isfinite(native) or not math.isfinite(vendor)
                or not 0 <= native < vendor):
            raise RuntimeError(
                f"{desc.direction} {desc.index} descriptor {identity}: "
                "not qualified for native symmetric pack"
            )

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
        self._runtime_id = id(self)
        self._frame_epoch = 0
        self._next_handle_id = 1
        self._live_handles: dict[int, DeviceTensorHandle] = {}
        self._resident_handle_creations = 0
        self._resident_handle_invalidations = 0
        self._resident_forwarded_inputs = 0
        self._resident_connections = 0
        self._python_transport_api_calls = 0
        self._cpp_resident_transaction_calls = 0
        self._frame_graph_nodes = 0
        self._frame_graph_peak_tensors = 0
        self._frame_graph_wall_seconds = 0.0
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

    def pack_tensor_symmetric(
        self, array: np.ndarray, descriptor: TensorLayoutDescriptor
    ) -> tuple[np.ndarray, np.ndarray]:
        self.codec_selection.require_native_symmetric_pack(descriptor)
        started = time.perf_counter()
        logical = np.ascontiguousarray(array)
        banks = self.codec_type.pack_tensor(logical, asdict(descriptor))
        elapsed = (time.perf_counter() - started) * 1000.0
        self.codec_selection.record(
            "native", "pack", [descriptor], [logical.nbytes], elapsed
        )
        return banks

    def quantize_pack_tensor(
        self, array: np.ndarray, descriptor: TensorLayoutDescriptor, scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse FP32-to-INT8 quantization with the qualified physical pack."""
        self.codec_selection.require_native(descriptor, "pack")
        if descriptor.bitdepth != 8:
            raise ValueError("quantize-pack requires an INT8 descriptor")
        started = time.perf_counter()
        logical = np.ascontiguousarray(array, dtype=np.float32)
        banks = self.codec_type.quantize_pack_tensor(
            logical, asdict(descriptor), float(scale)
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.codec_selection.record(
            "native", "pack", [descriptor], [math.prod(descriptor.dims)], elapsed
        )
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
        self._python_transport_api_calls += 1
        method(requests)

    def _c2h(self, requests: list[tuple[int, int, int]]) -> list[np.ndarray]:
        method = (self.transport.c2h_batch_safe if self.safe_dma
                  else self.transport.c2h_batch)
        self._python_transport_api_calls += 1
        return [np.ascontiguousarray(value) for value in method(requests)]

    def _execute_transaction(
        self,
        h2c_requests: list[tuple[int, int, np.ndarray]],
        programs: list[dict],
        c2h_requests: list[tuple[int, int, int]],
        timeout_ms: int,
    ) -> tuple[list[np.ndarray], list[float], float, float, bool]:
        """Compile a physical transaction to the C++ frame interpreter."""
        frame_method = getattr(self.transport, "run_frame_graph", None)
        if callable(frame_method):
            if len(h2c_requests) % 2 or len(c2h_requests) % 2:
                raise RuntimeError("frame transaction requires complete bank pairs")
            initial_tensors = {}
            input_names = []
            input_addresses = []
            for index in range(0, len(h2c_requests), 2):
                left, right = h2c_requests[index:index + 2]
                if (left[0], right[0]) != (0, 1):
                    raise RuntimeError("frame H2C bank pair order is invalid")
                if left[2].nbytes != right[2].nbytes:
                    raise RuntimeError("frame H2C bank pair extents differ")
                name = f"transaction.input.{index // 2}"
                initial_tensors[name] = (left[2], right[2])
                input_names.append(name)
                input_addresses.append((left[1], right[1]))

            output_names = []
            output_requests = []
            for index in range(0, len(c2h_requests), 2):
                left, right = c2h_requests[index:index + 2]
                if (left[0], right[0]) != (0, 1):
                    raise RuntimeError("frame C2H bank pair order is invalid")
                if left[2] != right[2]:
                    raise RuntimeError("frame C2H bank pair extents differ")
                output_names.append(f"transaction.output.{index // 2}")
                output_requests.append((left, right))

            nodes = []
            if input_names:
                nodes.append({
                    "op": "device_write", "inputs": input_names,
                    "addresses": input_addresses,
                })
            nodes.append({"op": "npu_chain", "programs": programs})
            if output_names:
                nodes.append({
                    "op": "device_read", "outputs": output_names,
                    "requests": output_requests,
                })
            self._python_transport_api_calls += 1
            result = dict(frame_method(
                initial_tensors, nodes, output_names, int(timeout_ms),
                self.safe_dma,
            ))
            required = {
                "outputs", "npu_seconds", "node_seconds", "opcode_seconds",
                "nodes", "programs", "peak_tensors", "wall_seconds",
            }
            if not required <= result.keys():
                raise RuntimeError("C++ frame transaction returned incomplete metadata")
            if (int(result["nodes"]) != len(nodes)
                    or int(result["programs"]) != len(programs)):
                raise RuntimeError("C++ frame transaction execution count mismatch")
            npu_seconds = [float(value) for value in result["npu_seconds"]]
            if len(npu_seconds) != len(programs):
                raise RuntimeError("C++ frame transaction timing count mismatch")
            returned = dict(result["outputs"])
            raw = [np.ascontiguousarray(bank)
                   for name in output_names for bank in returned[name]]
            if len(raw) != len(c2h_requests):
                raise RuntimeError("C++ frame transaction output count mismatch")
            opcode_seconds = dict(result["opcode_seconds"])
            self._cpp_resident_transaction_calls += 1
            self._frame_graph_nodes += len(nodes)
            self._frame_graph_peak_tensors = max(
                self._frame_graph_peak_tensors, int(result["peak_tensors"])
            )
            self._frame_graph_wall_seconds += float(result["wall_seconds"])
            return (
                raw, npu_seconds,
                float(opcode_seconds.get("device_write", 0.0)) * 1000.0,
                float(opcode_seconds.get("device_read", 0.0)) * 1000.0,
                True,
            )

        method = getattr(self.transport, "run_resident_transaction", None)
        if callable(method):
            self._python_transport_api_calls += 1
            result = dict(method(
                h2c_requests, programs, c2h_requests, int(timeout_ms),
                self.safe_dma,
            ))
            required = {"outputs", "npu_seconds", "h2c_seconds", "c2h_seconds"}
            if not required <= result.keys():
                raise RuntimeError("C++ resident transaction returned incomplete metadata")
            raw = [np.ascontiguousarray(value) for value in result["outputs"]]
            npu_seconds = [float(value) for value in result["npu_seconds"]]
            if len(npu_seconds) != len(programs):
                raise RuntimeError("C++ resident transaction timing count mismatch")
            self._cpp_resident_transaction_calls += 1
            return (
                raw, npu_seconds,
                float(result["h2c_seconds"]) * 1000.0,
                float(result["c2h_seconds"]) * 1000.0,
                True,
            )

        h2c_started = time.perf_counter()
        if h2c_requests:
            self._h2c(h2c_requests)
        h2c_ms = (time.perf_counter() - h2c_started) * 1000.0
        self._python_transport_api_calls += 1
        npu_seconds = [float(value) for value in
                       self.transport.run_npu_chain(programs, int(timeout_ms))]
        c2h_started = time.perf_counter()
        raw = self._c2h(c2h_requests) if c2h_requests else []
        c2h_ms = (time.perf_counter() - c2h_started) * 1000.0
        return raw, npu_seconds, h2c_ms, c2h_ms, False

    def load_bank(self, combined: np.ndarray) -> float:
        self._invalidate_all_handles()
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

    @staticmethod
    def _overlaps(left_address: int, left_size: int,
                  right_address: int, right_size: int) -> bool:
        return (left_address < right_address + right_size
                and right_address < left_address + left_size)

    def _tensor_addresses(self, record: dict, tensor: dict,
                          base_offset_units: int) -> tuple[int, int]:
        relative = (
            int(record["base_addresses"][4]) + int(tensor["address"])
            + int(base_offset_units)
        ) * ADDRESS_UNIT_BYTES_PER_BANK
        return tuple(relative + base for base in DDR_BASES)  # type: ignore[return-value]

    def _validate_offset(self, record: dict, base_offset_units: int) -> None:
        if type(base_offset_units) is not int or base_offset_units < 0:
            raise ValueError(f"{record['name']}: invalid FM base offset")
        end = base_offset_units * ADDRESS_UNIT_BYTES_PER_BANK + record_span_per_bank(record)
        if end > self.workspace_bytes_per_bank:
            raise ValueError(
                f"{record['name']}: relocated FM span {end} exceeds "
                f"{self.workspace_bytes_per_bank} bytes per bank"
            )

    def _invalidate_all_handles(self) -> None:
        self._resident_handle_invalidations += len(self._live_handles)
        self._live_handles.clear()

    def _invalidate_ranges(self, ranges: list[tuple[tuple[int, int], int]]) -> None:
        stale = []
        for handle_id, handle in self._live_handles.items():
            if any(any(self._overlaps(handle.bank_addresses[bank],
                                      handle.bytes_per_bank,
                                      addresses[bank], size)
                       for bank in range(2))
                   for addresses, size in ranges):
                stale.append(handle_id)
        for handle_id in stale:
            del self._live_handles[handle_id]
        self._resident_handle_invalidations += len(stale)

    def _new_handle(self, record: dict, tensor: dict,
                    descriptor: TensorLayoutDescriptor, base_offset_units: int,
                    scale: float | None, owner_group: int, label: str
                    ) -> DeviceTensorHandle:
        combined = int(tensor["size_per_bank"])
        if combined != descriptor.combined_bytes or combined % 2:
            raise ValueError(f"{label}: descriptor extent does not match manifest")
        handle = DeviceTensorHandle(
            runtime_id=self._runtime_id,
            frame_epoch=self._frame_epoch,
            handle_id=self._next_handle_id,
            bank_addresses=self._tensor_addresses(record, tensor, base_offset_units),
            bytes_per_bank=combined // 2,
            storage_identity=descriptor.storage_identity(),
            scale=None if scale is None else float(scale),
            owner_group=owner_group,
            lifetime="frame",
            label=label,
        )
        self._next_handle_id += 1
        self._live_handles[handle.handle_id] = handle
        self._resident_handle_creations += 1
        return handle

    def _validate_handle(self, handle: DeviceTensorHandle, record: dict,
                         tensor: dict, descriptor: TensorLayoutDescriptor,
                         base_offset_units: int, scale: float | None) -> None:
        if handle.runtime_id != self._runtime_id:
            raise RuntimeError(f"{record['name']}: device tensor belongs to another runtime")
        if handle.frame_epoch != self._frame_epoch:
            raise RuntimeError(f"{record['name']}: device tensor has expired frame lifetime")
        if self._live_handles.get(handle.handle_id) != handle:
            raise RuntimeError(f"{record['name']}: device tensor handle is stale")
        expected_addresses = self._tensor_addresses(record, tensor, base_offset_units)
        expected_bytes = int(tensor["size_per_bank"]) // 2
        expected_scale = None if scale is None else float(scale)
        if handle.bank_addresses != expected_addresses:
            raise RuntimeError(f"{record['name']}: device tensor address mismatch")
        if handle.bytes_per_bank != expected_bytes:
            raise RuntimeError(f"{record['name']}: device tensor extent mismatch")
        if handle.storage_identity != descriptor.storage_identity():
            raise RuntimeError(f"{record['name']}: device tensor storage ABI mismatch")
        if handle.scale != expected_scale:
            raise RuntimeError(f"{record['name']}: device tensor scale mismatch")

    def run_resident_chain(
        self,
        records: list[dict],
        packed_calls: list[list[ResidentInput]],
        base_offsets_units: list[int],
        connections: dict[tuple[int, int], tuple[int, int]],
        timeout_ms: int,
        *,
        download_masks: list[list[bool]] | None = None,
        input_scales: list[list[float | None]] | None = None,
        output_scales: list[list[float | None]] | None = None,
    ) -> tuple[
        list[list[PhysicalTensor | None]],
        list[list[DeviceTensorHandle | None]],
        list[list[DeviceTensorHandle]],
        dict,
    ]:
        """Run an address-qualified chain with existing or connected FM inputs."""
        count = len(records)
        if not count or len(packed_calls) != count or len(base_offsets_units) != count:
            raise ValueError("resident chain records, inputs, and offsets must align")
        if download_masks is None:
            download_masks = [
                [True] * len(record["outputs"]) for record in records
            ]
        if input_scales is None:
            input_scales = [
                [None] * len(record["inputs"]) for record in records
            ]
        if output_scales is None:
            output_scales = [
                [None] * len(record["outputs"]) for record in records
            ]
        if not all(len(values) == count for values in (
                download_masks, input_scales, output_scales)):
            raise ValueError("resident chain metadata does not align with records")

        descriptors = []
        for call, (record, packed, downloads, in_scale, out_scale, offset) in enumerate(
                zip(records, packed_calls, download_masks, input_scales,
                    output_scales, base_offsets_units)):
            self._validate_offset(record, offset)
            name = record["name"]
            if (not self.codec_selection.native_for(name, "input")
                    or not self.codec_selection.native_for(name, "output")):
                raise RuntimeError(f"{name}: resident chaining requires qualified native IO")
            case = self.codec_selection.descriptors[name]
            if (len(packed) != len(record["inputs"])
                    or len(case["input"]) != len(record["inputs"])
                    or len(case["output"]) != len(record["outputs"])
                    or len(in_scale) != len(record["inputs"])
                    or len(downloads) != len(record["outputs"])
                    or len(out_scale) != len(record["outputs"])):
                raise ValueError(f"{name}: resident chain tensor metadata mismatch")
            descriptors.append(case)

        # Validate graph connections before constructing any DMA request.
        for (consumer_call, input_index), (producer_call, output_index) in connections.items():
            if not (0 <= producer_call < consumer_call < count):
                raise ValueError("resident connection must point from an earlier producer")
            consumer = records[consumer_call]
            producer = records[producer_call]
            if not (0 <= input_index < len(consumer["inputs"])
                    and 0 <= output_index < len(producer["outputs"])):
                raise ValueError("resident connection tensor index is out of range")
            if packed_calls[consumer_call][input_index] is not None:
                raise ValueError("connected resident input must use a None placeholder")
            source_tensor = producer["outputs"][output_index]
            target_tensor = consumer["inputs"][input_index]
            source_desc = descriptors[producer_call]["output"][output_index]
            target_desc = descriptors[consumer_call]["input"][input_index]
            if source_desc.storage_identity() != target_desc.storage_identity():
                raise RuntimeError("resident connection storage ABI mismatch")
            if self._tensor_addresses(producer, source_tensor,
                                      base_offsets_units[producer_call]) != self._tensor_addresses(
                                          consumer, target_tensor,
                                          base_offsets_units[consumer_call]):
                raise RuntimeError("resident connection address mismatch")
            if output_scales[producer_call][output_index] != input_scales[consumer_call][input_index]:
                raise RuntimeError("resident connection scale mismatch")

        h2c_requests: list[tuple[int, int, np.ndarray]] = []
        upload_ranges: list[tuple[int, tuple[int, int], int]] = []
        existing_handles: list[tuple[int, int, DeviceTensorHandle]] = []
        uploaded_bytes = skipped_bytes = 0
        for call, (record, packed, offset, scales) in enumerate(
                zip(records, packed_calls, base_offsets_units, input_scales)):
            for index, (tensor, value, scale) in enumerate(
                    zip(record["inputs"], packed, scales)):
                connection = connections.get((call, index))
                if connection is not None:
                    skipped_bytes += int(tensor["size_per_bank"])
                    continue
                if value is None:
                    raise ValueError(f"{record['name']}: unconnected resident input is missing")
                descriptor = descriptors[call]["input"][index]
                if isinstance(value, DeviceTensorHandle):
                    self._validate_handle(value, record, tensor, descriptor, offset, scale)
                    existing_handles.append((call, index, value))
                    skipped_bytes += int(tensor["size_per_bank"])
                    continue
                combined = int(tensor["size_per_bank"])
                if (not isinstance(value, tuple) or len(value) != 2
                        or any(not isinstance(half, np.ndarray)
                               or half.dtype != np.uint8 or half.ndim != 1
                               or not half.flags.c_contiguous
                               or half.nbytes != combined // 2 for half in value)):
                    raise ValueError(
                        f"{record['name']}: resident native input requires exact uint8 bank pair"
                    )
                addresses = self._tensor_addresses(record, tensor, offset)
                upload_ranges.append((call, addresses, combined // 2))
                for bank, half in enumerate(value):
                    h2c_requests.append((bank, addresses[bank], half))
                    uploaded_bytes += int(half.size)

        # All uploads happen before the first program. Ambiguous overlapping
        # uploads or destruction of a forwarded input therefore fail closed.
        for index, (upload_call, addresses, size) in enumerate(upload_ranges):
            for _, other_addresses, other_size in upload_ranges[index + 1:]:
                if any(self._overlaps(addresses[bank], size,
                                      other_addresses[bank], other_size)
                       for bank in range(2)):
                    raise RuntimeError("resident chain contains overlapping H2C writes")
            for _, _, handle in existing_handles:
                if any(self._overlaps(addresses[bank], size,
                                      handle.bank_addresses[bank], handle.bytes_per_bank)
                       for bank in range(2)):
                    raise RuntimeError("resident H2C write overlaps a forwarded device tensor")

        output_ranges = []
        for call, (record, offset) in enumerate(zip(records, base_offsets_units)):
            for index, tensor in enumerate(record["outputs"]):
                output_ranges.append((call, index,
                                      self._tensor_addresses(record, tensor, offset),
                                      int(tensor["size_per_bank"]) // 2))
        for index, (call, output, addresses, size) in enumerate(output_ranges):
            for next_call, next_output, other_addresses, other_size in output_ranges[index + 1:]:
                if any(self._overlaps(addresses[bank], size,
                                      other_addresses[bank], other_size)
                       for bank in range(2)):
                    raise RuntimeError(
                        f"resident outputs overlap: {call}:{output} and "
                        f"{next_call}:{next_output}"
                    )

        # Inputs are staged before any program runs. An earlier producer may
        # therefore overwrite a later call's preloaded/forwarded input. Such a
        # dependency must be represented explicitly in `connections`; silently
        # accepting it would make a handle look valid while its bytes changed.
        for output_call, output_index, addresses, size in output_ranges:
            for upload_call, input_addresses, input_size in upload_ranges:
                if output_call >= upload_call:
                    continue
                if any(self._overlaps(addresses[bank], size,
                                      input_addresses[bank], input_size)
                       for bank in range(2)):
                    raise RuntimeError(
                        f"resident output {output_call}:{output_index} overwrites "
                        f"preloaded input for call {upload_call}"
                    )
            for handle_call, input_index, handle in existing_handles:
                if output_call >= handle_call:
                    continue
                if any(self._overlaps(addresses[bank], size,
                                      handle.bank_addresses[bank],
                                      handle.bytes_per_bank)
                       for bank in range(2)):
                    raise RuntimeError(
                        f"resident output {output_call}:{output_index} overwrites "
                        f"forwarded input {handle_call}:{input_index}"
                    )

        programs = []
        for record, offset in zip(records, base_offsets_units):
            bases = [int(value) for value in record["base_addresses"]]
            bases[4] += offset
            programs.append({
                "stage_id": record["name"], "base_addresses": bases,
                "isa_ranges": [int(value) for value in record["isa_ranges"]],
            })

        output_requests = []
        output_plan = []
        downloaded_bytes = 0
        output_locations = {
            (call, index): (addresses, size)
            for call, index, addresses, size in output_ranges
        }
        for call, (record, downloads) in enumerate(zip(records, download_masks)):
            output_plan.append(list(downloads))
            for index, download in enumerate(downloads):
                if not download:
                    continue
                addresses, size = output_locations[call, index]
                for bank in range(2):
                    output_requests.append((bank, addresses[bank], size))
                    downloaded_bytes += size

        # Invalidate ranges before entering the atomic C++ transaction.  On
        # failure all handles are invalidated below, so no speculative handle
        # can escape after a partial DMA or timed-out program chain.
        self._invalidate_ranges([(addresses, size)
                                 for _, addresses, size in upload_ranges])
        owner_group = self.groups + 1
        input_handles: list[list[DeviceTensorHandle | None]] = [
            [None] * len(record["inputs"]) for record in records
        ]
        for call, index, handle in existing_handles:
            input_handles[call][index] = handle
        for call, (record, packed, offset, scales) in enumerate(
                zip(records, packed_calls, base_offsets_units, input_scales)):
            for index, (tensor, value, scale) in enumerate(
                    zip(record["inputs"], packed, scales)):
                if value is not None and not isinstance(value, DeviceTensorHandle):
                    input_handles[call][index] = self._new_handle(
                        record, tensor, descriptors[call]["input"][index], offset,
                        scale, owner_group, f"{record['name']}:input:{index}",
                    )

        try:
            raw, npu_seconds, h2c_ms, c2h_ms, cpp_transaction = (
                self._execute_transaction(
                    h2c_requests, programs, output_requests, timeout_ms
                )
            )
        except Exception:
            self._invalidate_all_handles()
            raise
        self._invalidate_ranges([(addresses, size)
                                 for _, _, addresses, size in output_ranges])
        output_handles: list[list[DeviceTensorHandle]] = []
        for call, (record, offset, scales) in enumerate(
                zip(records, base_offsets_units, output_scales)):
            output_handles.append([
                self._new_handle(
                    record, tensor, descriptors[call]["output"][index], offset,
                    scales[index], owner_group, f"{record['name']}:output:{index}",
                )
                for index, tensor in enumerate(record["outputs"])
            ])

        results: list[list[PhysicalTensor | None]] = []
        cursor = 0
        for downloads in output_plan:
            values: list[PhysicalTensor | None] = []
            for download in downloads:
                if download:
                    values.append((raw[cursor], raw[cursor + 1]))
                    cursor += 2
                else:
                    values.append(None)
            results.append(values)
        if cursor != len(raw):
            raise RuntimeError("resident C2H result count does not match output plan")

        self.groups += 1
        self.dispatches += count
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        self._resident_forwarded_inputs += len(existing_handles)
        self._resident_connections += len(connections)
        return results, input_handles, output_handles, {
            "submission_group_size": count,
            "h2c_ms": h2c_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "c2h_ms": c2h_ms,
            "h2c_bytes": uploaded_bytes,
            "c2h_bytes": downloaded_bytes,
            "h2c_skipped_bytes": skipped_bytes,
            "resident_forwarded_inputs": len(existing_handles),
            "resident_connections": len(connections),
            "base_offsets_units": list(base_offsets_units),
            "cpp_resident_transaction": cpp_transaction,
        }

    def run_decoder_capture_stems(
        self,
        stems: list[dict],
        timeout_ms: int,
    ) -> tuple[list[list[PhysicalTensor]], dict]:
        """Bridge resident BF16 captures to qualified project Conv groups.

        The complete capture snapshot, LayerNorm/layout/quantization bridge,
        and all project launches execute under one C++ schedule lock.  This is
        the fail-closed replacement for Mat2Img programs that compile but do
        not complete on the deployed U250 bitstream.
        """
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError(
                "loaded cpp_mapped extension has no C++ frame-graph interpreter"
            )
        if not stems:
            raise ValueError("decoder capture stem list is empty")

        native_stems = []
        output_plans = []
        dispatches = 0
        h2c_bytes = c2h_bytes = source_c2h_bytes = 0
        for stem_index, stem in enumerate(stems):
            source = stem["source"]
            source_descriptor = stem["source_descriptor"]
            target_descriptor = stem["target_descriptor"]
            if (source_descriptor.layout != "NDWC"
                    or source_descriptor.bitdepth != 16
                    or target_descriptor.layout != "NCHW"
                    or target_descriptor.bitdepth != 8):
                raise RuntimeError("decoder capture bridge storage ABI mismatch")
            native = {
                "source_descriptor": asdict(source_descriptor),
                "target_descriptor": asdict(target_descriptor),
                "gamma": np.ascontiguousarray(stem["gamma"], dtype=np.float32),
                "beta": np.ascontiguousarray(stem["beta"], dtype=np.float32),
                "scale": float(stem["scale"]),
                "epsilon": float(stem.get("epsilon", 1.0e-6)),
                "targets": [],
            }
            if isinstance(source, DeviceTensorHandle):
                if source.runtime_id != self._runtime_id:
                    raise RuntimeError("decoder capture belongs to another runtime")
                if source.frame_epoch != self._frame_epoch:
                    raise RuntimeError("decoder capture has expired frame lifetime")
                if self._live_handles.get(source.handle_id) != source:
                    raise RuntimeError("decoder capture handle is stale")
                if source.storage_identity != source_descriptor.storage_identity():
                    raise RuntimeError("decoder capture storage ABI mismatch")
                if source.bytes_per_bank * 2 != source_descriptor.combined_bytes:
                    raise RuntimeError("decoder capture extent mismatch")
                native["source_requests"] = [
                    (bank, source.bank_addresses[bank], source.bytes_per_bank)
                    for bank in range(2)
                ]
                source_c2h_bytes += source.bytes_per_bank * 2
            else:
                if (not isinstance(source, tuple) or len(source) != 2
                        or any(not isinstance(bank, np.ndarray)
                               or bank.dtype != np.uint8 or bank.ndim != 1
                               or not bank.flags.c_contiguous
                               or bank.nbytes != source_descriptor.combined_bytes // 2
                               for bank in source)):
                    raise ValueError(
                        "host decoder capture requires an exact physical bank pair"
                    )
                native["source_pair"] = source

            records = list(stem["records"])
            if not records:
                raise ValueError("decoder project has no kernel records")
            if any(len(record["inputs"]) != 1 for record in records):
                raise RuntimeError("decoder project bridge requires one-input kernels")
            stride = max(record_span_per_bank(record) for record in records)
            if stride * len(records) > self.workspace_bytes_per_bank:
                raise ValueError(
                    f"decoder stem {stem_index} exceeds shared FM workspace"
                )
            stem_plan = []
            for slot, record in enumerate(records):
                name = record["name"]
                if (not self.codec_selection.native_for(name, "input")
                        or not self.codec_selection.native_for(name, "output")):
                    raise RuntimeError(
                        f"{name}: decoder frame graph requires qualified native IO"
                    )
                input_descriptor = self.codec_selection.descriptors[name]["input"][0]
                if input_descriptor.storage_identity() != target_descriptor.storage_identity():
                    raise RuntimeError(f"{name}: decoder target storage ABI mismatch")
                slot_units = slot * stride // ADDRESS_UNIT_BYTES_PER_BANK
                bases = [int(value) for value in record["base_addresses"]]
                bases[4] += slot_units
                input_addresses = self._tensor_addresses(
                    record, record["inputs"][0], slot_units
                )
                requests = []
                output_count = 0
                for tensor in record["outputs"]:
                    combined = int(tensor["size_per_bank"])
                    if combined % 2:
                        raise ValueError(f"{name}: odd output byte count")
                    addresses = self._tensor_addresses(record, tensor, slot_units)
                    for bank in range(2):
                        requests.append((bank, addresses[bank], combined // 2))
                        c2h_bytes += combined // 2
                    output_count += 1
                native["targets"].append({
                    "program": {
                        "stage_id": name,
                        "base_addresses": bases,
                        "isa_ranges": [int(value) for value in record["isa_ranges"]],
                    },
                    "input_addresses": input_addresses,
                    "output_requests": requests,
                })
                h2c_bytes += target_descriptor.combined_bytes
                dispatches += 1
                stem_plan.append(output_count)
            output_plans.append(stem_plan)
            native_stems.append(native)

        # Compile this boundary to generic C++ frame-graph bytecode.  The first
        # read snapshots every resident capture before any project program can
        # reuse its FM address; all later dependencies remain in the C++ tensor
        # table and never become Python scheduling decisions.
        initial_tensors = {}
        nodes = []
        capture_names = []
        resident_names = []
        resident_requests = []
        for stem_index, native in enumerate(native_stems):
            name = f"decoder.capture.{stem_index}"
            capture_names.append(name)
            if "source_requests" in native:
                resident_names.append(name)
                resident_requests.append(native["source_requests"])
            else:
                initial_tensors[name] = native["source_pair"]
        source_read_node = None
        if resident_names:
            source_read_node = len(nodes)
            nodes.append({
                "op": "device_read",
                "outputs": resident_names,
                "requests": resident_requests,
            })

        project_inputs = []
        for stem_index, (native, capture_name) in enumerate(
                zip(native_stems, capture_names)):
            project_input = f"decoder.project_input.{stem_index}"
            project_inputs.append(project_input)
            nodes.append({
                "op": "decoder_capture_pack_bf16",
                "input": capture_name,
                "output": project_input,
                "source_descriptor": native["source_descriptor"],
                "target_descriptor": native["target_descriptor"],
                "gamma": native["gamma"],
                "beta": native["beta"],
                "scale": native["scale"],
                "epsilon": native["epsilon"],
            })

        fetches = []
        fetched_plans = []
        for stem_index, (native, project_input) in enumerate(
                zip(native_stems, project_inputs)):
            targets = native["targets"]
            nodes.append({
                "op": "device_write",
                "inputs": [project_input] * len(targets),
                "addresses": [target["input_addresses"] for target in targets],
            })
            nodes.append({
                "op": "npu_chain",
                "programs": [target["program"] for target in targets],
            })
            output_names = []
            output_requests = []
            stem_plan = []
            for target_index, target in enumerate(targets):
                target_names = []
                requests = target["output_requests"]
                for output_index in range(0, len(requests), 2):
                    name = (f"decoder.project_output.{stem_index}."
                            f"{target_index}.{output_index // 2}")
                    target_names.append(name)
                    output_names.append(name)
                    output_requests.append(requests[output_index:output_index + 2])
                stem_plan.append(target_names)
            nodes.append({
                "op": "device_read",
                "outputs": output_names,
                "requests": output_requests,
            })
            fetches.extend(output_names)
            fetched_plans.append(stem_plan)

        self._python_transport_api_calls += 1
        try:
            result = dict(method(
                initial_tensors, nodes, fetches, int(timeout_ms), self.safe_dma
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        required = {
            "outputs", "npu_seconds", "node_seconds", "opcode_seconds",
            "nodes", "programs", "peak_tensors", "wall_seconds",
        }
        if not required <= result.keys():
            raise RuntimeError("C++ decoder frame graph returned incomplete metadata")
        if int(result["programs"]) != dispatches:
            raise RuntimeError("C++ decoder frame graph dispatch count mismatch")
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        if len(npu_seconds) != dispatches:
            raise RuntimeError("C++ decoder frame graph timing count mismatch")

        if int(result["nodes"]) != len(nodes):
            raise RuntimeError("C++ decoder frame graph node count mismatch")
        raw_outputs = dict(result["outputs"])
        physical_results = []
        for stem_plan, expected_counts in zip(fetched_plans, output_plans):
            values = []
            for target_names, expected_count in zip(stem_plan, expected_counts):
                if len(target_names) != expected_count:
                    raise RuntimeError("decoder frame graph output plan mismatch")
                call_outputs = []
                for name in target_names:
                    pair = tuple(np.ascontiguousarray(value)
                                 for value in raw_outputs[name])
                    if len(pair) != 2:
                        raise RuntimeError(
                            "decoder frame graph returned malformed bank pair"
                        )
                    call_outputs.append(pair)
                values.append(call_outputs)
            physical_results.append(values)

        # The decoder bridge snapshots every retained encoder capture before
        # launching any project Conv.  Those launches reuse the shared FM
        # workspace, so no pre-decoder handle remains valid afterwards even
        # though its nominal byte range may not overlap the first target.
        self._invalidate_all_handles()

        groups = len(stems)
        opcode_seconds = dict(result["opcode_seconds"])
        node_seconds = [float(value) for value in result["node_seconds"]]
        source_c2h_seconds = (
            node_seconds[source_read_node] if source_read_node is not None else 0.0
        )
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        self.groups += groups
        self.dispatches += dispatches
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        return physical_results, {
            "submission_groups": groups,
            "submission_group_size": dispatches,
            "h2c_ms": h2c_ms,
            "c2h_ms": c2h_ms,
            "source_c2h_ms": source_c2h_seconds * 1000.0,
            "bridge_ms": float(opcode_seconds.get(
                "decoder_capture_pack_bf16", 0.0
            )) * 1000.0,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "h2c_bytes": h2c_bytes,
            "c2h_bytes": c2h_bytes + source_c2h_bytes,
            "source_c2h_bytes": source_c2h_bytes,
            "cpp_frame_graph": True,
            "frame_graph_nodes": len(nodes),
            "frame_graph_peak_tensors": int(result["peak_tensors"]),
            "frame_graph_wall_ms": float(result["wall_seconds"]) * 1000.0,
        }

    def run_fc1_gelu_fc2(
        self,
        fc1_records: list[dict],
        fc1_input: PhysicalTensor,
        fc2_record: dict,
        scale: float,
        timeout_ms: int,
    ) -> tuple[list[PhysicalTensor], dict]:
        """Execute sharded FC1, physical GELU bridge and FC2 in one graph."""
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError("loaded extension has no C++ frame-graph interpreter")
        if not fc1_records:
            raise ValueError("FC1 frame graph requires at least one shard")
        if not isinstance(fc1_input, tuple) or len(fc1_input) != 2:
            raise ValueError("FC1 frame graph input must be a physical bank pair")
        input_half_bytes = int(fc1_records[0]["inputs"][0]["size_per_bank"]) // 2
        if any(not isinstance(bank, np.ndarray)
               or bank.dtype != np.uint8 or bank.ndim != 1
               or not bank.flags.c_contiguous or bank.nbytes != input_half_bytes
               for bank in fc1_input):
            raise ValueError("FC1 frame graph input has the wrong physical extent")

        stride = max(record_span_per_bank(record) for record in fc1_records)
        parallel_shards = self.workspace_bytes_per_bank // stride
        if parallel_shards < 1:
            raise ValueError("one FC1 shard exceeds shared FM workspace")
        source_descriptors = []
        input_addresses = []
        fc1_programs = []
        fc1_output_names = []
        fc1_output_requests = []
        fc1_output_names_by_record = []
        fc1_output_requests_by_record = []
        for slot, record in enumerate(fc1_records):
            name = record["name"]
            descriptors = self.codec_selection.descriptors[name]
            if (len(record["inputs"]) != 1
                    or len(descriptors["input"]) != 1
                    or not record["outputs"]
                    or len(descriptors["output"]) != len(record["outputs"])):
                raise RuntimeError(f"{name}: FC1 shard ABI is not supported")
            if (not self.codec_selection.native_for(name, "input")
                    or not self.codec_selection.native_for(name, "output")):
                raise RuntimeError(f"{name}: FC1 frame graph requires native IO")
            if int(record["inputs"][0]["size_per_bank"]) // 2 != input_half_bytes:
                raise RuntimeError("FC1 shard input extents differ")
            # Outputs are materialized after each chunk, so a later chunk may
            # reuse the same physical FM slots inside this one transaction.
            offset = (slot % parallel_shards) * stride // ADDRESS_UNIT_BYTES_PER_BANK
            bases = [int(value) for value in record["base_addresses"]]
            bases[4] += offset
            input_addresses.append(self._tensor_addresses(
                record, record["inputs"][0], offset
            ))
            record_names = []
            record_requests = []
            for output_index, (output_tensor, output_descriptor) in enumerate(
                    zip(record["outputs"], descriptors["output"])):
                combined = int(output_tensor["size_per_bank"])
                if combined % 2:
                    raise ValueError(f"{name}: odd FC1 output extent")
                addresses = self._tensor_addresses(record, output_tensor, offset)
                output_name = f"fc1.output.{slot}.{output_index}"
                requests = [
                    (bank, addresses[bank], combined // 2) for bank in range(2)
                ]
                record_names.append(output_name)
                record_requests.append(requests)
                fc1_output_names.append(output_name)
                fc1_output_requests.append(requests)
                source_descriptors.append(output_descriptor)
            fc1_output_names_by_record.append(record_names)
            fc1_output_requests_by_record.append(record_requests)
            fc1_programs.append({
                "stage_id": name,
                "base_addresses": bases,
                "isa_ranges": [int(value) for value in record["isa_ranges"]],
            })

        fc2_name = fc2_record["name"]
        fc2_descriptors = self.codec_selection.descriptors[fc2_name]
        if (len(fc2_record["inputs"]) != 1
                or len(fc2_descriptors["input"]) != 1
                or not self.codec_selection.native_for(fc2_name, "input")
                or not self.codec_selection.native_for(fc2_name, "output")):
            raise RuntimeError(f"{fc2_name}: FC2 frame graph requires native IO")
        target_descriptor = fc2_descriptors["input"][0]
        if sum(item.dims[3] for item in source_descriptors) != target_descriptor.dims[3]:
            raise RuntimeError("FC1 shard channels do not match FC2 input")
        fc2_input_addresses = self._tensor_addresses(
            fc2_record, fc2_record["inputs"][0], 0
        )
        fc2_program = {
            "stage_id": fc2_name,
            "base_addresses": [int(value) for value in fc2_record["base_addresses"]],
            "isa_ranges": [int(value) for value in fc2_record["isa_ranges"]],
        }
        fc2_output_names = []
        fc2_output_requests = []
        for index, tensor in enumerate(fc2_record["outputs"]):
            combined = int(tensor["size_per_bank"])
            if combined % 2:
                raise ValueError(f"{fc2_name}: odd FC2 output extent")
            addresses = self._tensor_addresses(fc2_record, tensor, 0)
            fc2_output_names.append(f"fc2.output.{index}")
            fc2_output_requests.append([
                (bank, addresses[bank], combined // 2) for bank in range(2)
            ])

        touched_ranges = []
        for record, addresses in zip(fc1_records, input_addresses):
            touched_ranges.append((addresses, int(record["inputs"][0]["size_per_bank"]) // 2))
        for requests in fc1_output_requests:
            touched_ranges.append((tuple(item[1] for item in requests), requests[0][2]))
        touched_ranges.append((fc2_input_addresses,
                               int(fc2_record["inputs"][0]["size_per_bank"]) // 2))
        for requests in fc2_output_requests:
            touched_ranges.append((tuple(item[1] for item in requests), requests[0][2]))

        nodes = []
        for start in range(0, len(fc1_records), parallel_shards):
            stop = min(len(fc1_records), start + parallel_shards)
            chunk_output_names = [
                name for values in fc1_output_names_by_record[start:stop]
                for name in values
            ]
            chunk_output_requests = [
                requests
                for values in fc1_output_requests_by_record[start:stop]
                for requests in values
            ]
            nodes.extend([
                {"op": "device_write",
                 "inputs": ["fc1.input"] * (stop - start),
                 "addresses": input_addresses[start:stop]},
                {"op": "npu_chain", "programs": fc1_programs[start:stop]},
                {"op": "device_read", "outputs": chunk_output_names,
                 "requests": chunk_output_requests},
            ])
        nodes.extend([
            {"op": "gelu_pack_bf16_concatenate", "inputs": fc1_output_names,
             "output": "fc2.input",
             "source_descriptors": [asdict(item) for item in source_descriptors],
             "target_descriptor": asdict(target_descriptor), "scale": float(scale)},
            {"op": "device_write", "inputs": ["fc2.input"],
             "addresses": [fc2_input_addresses]},
            {"op": "npu_chain", "programs": [fc2_program]},
            {"op": "device_read", "outputs": fc2_output_names,
             "requests": fc2_output_requests},
        ])
        self._invalidate_ranges(touched_ranges)
        self._python_transport_api_calls += 1
        try:
            result = dict(method(
                {"fc1.input": fc1_input}, nodes, fc2_output_names,
                int(timeout_ms), self.safe_dma,
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        if (int(result["nodes"]) != len(nodes)
                or int(result["programs"]) != len(fc1_records) + 1):
            raise RuntimeError("FC frame graph execution count mismatch")
        returned = dict(result["outputs"])
        outputs = [tuple(np.ascontiguousarray(bank) for bank in returned[name])
                   for name in fc2_output_names]
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        if len(npu_seconds) != len(fc1_records) + 1:
            raise RuntimeError("FC frame graph timing count mismatch")
        opcode_seconds = dict(result["opcode_seconds"])
        self.groups += 1
        self.dispatches += len(fc1_records) + 1
        self._cpp_resident_transaction_calls += 1
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        return outputs, {
            "submission_group_size": len(fc1_records) + 1,
            "h2c_ms": h2c_ms,
            "c2h_ms": c2h_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "bridge_ms": float(opcode_seconds.get(
                "gelu_pack_bf16_concatenate", 0.0
            )) * 1000.0,
            "cpp_frame_graph": True,
            "frame_graph_nodes": len(nodes),
            "h2c_bytes": (sum(int(record["inputs"][0]["size_per_bank"])
                              for record in fc1_records)
                           + int(fc2_record["inputs"][0]["size_per_bank"])),
            "c2h_bytes": (sum(int(tensor["size_per_bank"])
                              for record in fc1_records
                              for tensor in record["outputs"])
                           + sum(int(tensor["size_per_bank"])
                                 for tensor in fc2_record["outputs"])),
        }

    def run_attention_post_frame_graph(
        self,
        attention_records: list[dict],
        attention_inputs: list[list[PhysicalTensor]],
        post_record: dict,
        post_residual: PhysicalTensor,
        valid_widths: list[int],
        scale: float,
        max_parallel: int,
        timeout_ms: int,
    ) -> tuple[list[PhysicalTensor], dict]:
        """Execute all attention heads, their physical bridge and post projection."""
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError("loaded extension has no C++ frame-graph interpreter")
        if (not attention_records
                or len(attention_records) != len(attention_inputs)
                or max_parallel < 1):
            raise ValueError("attention frame graph calls are not aligned")
        stride = max(record_span_per_bank(record) for record in attention_records)
        parallel = min(max_parallel, self.workspace_bytes_per_bank // stride)
        if parallel < 1:
            raise ValueError("one attention call exceeds shared FM workspace")

        initial = {"post.residual": post_residual}
        input_names_by_call = []
        input_addresses_by_call = []
        programs = []
        output_names_by_call = []
        output_requests_by_call = []
        source_descriptors = []
        touched_ranges = []
        for call, (record, packed) in enumerate(
                zip(attention_records, attention_inputs)):
            name = record["name"]
            descriptors = self.codec_selection.descriptors[name]
            if (len(packed) != len(record["inputs"])
                    or len(record["outputs"]) != len(descriptors["output"])
                    or not self.codec_selection.native_for(name, "input")
                    or not self.codec_selection.native_for(name, "output")):
                raise RuntimeError(f"{name}: attention frame-graph ABI is not supported")
            slot_units = (call % parallel) * stride // ADDRESS_UNIT_BYTES_PER_BANK
            names = []
            addresses = []
            for index, (tensor, physical) in enumerate(zip(record["inputs"], packed)):
                tensor_name = f"attention.input.{call}.{index}"
                initial[tensor_name] = physical
                names.append(tensor_name)
                location = self._tensor_addresses(record, tensor, slot_units)
                addresses.append(location)
                touched_ranges.append((location, int(tensor["size_per_bank"]) // 2))
            input_names_by_call.append(names)
            input_addresses_by_call.append(addresses)
            bases = [int(value) for value in record["base_addresses"]]
            bases[4] += slot_units
            programs.append({
                "stage_id": name,
                "base_addresses": bases,
                "isa_ranges": [int(value) for value in record["isa_ranges"]],
            })
            names = []
            requests = []
            for index, (tensor, descriptor) in enumerate(
                    zip(record["outputs"], descriptors["output"])):
                tensor_name = f"attention.output.{call}.{index}"
                location = self._tensor_addresses(record, tensor, slot_units)
                size = int(tensor["size_per_bank"]) // 2
                names.append(tensor_name)
                requests.append([(bank, location[bank], size) for bank in range(2)])
                source_descriptors.append(descriptor)
                touched_ranges.append((location, size))
            output_names_by_call.append(names)
            output_requests_by_call.append(requests)

        post_name = post_record["name"]
        post_descriptors = self.codec_selection.descriptors[post_name]
        if (len(post_record["inputs"]) != 2
                or len(post_descriptors["input"]) != 2
                or not post_record["outputs"]
                or not self.codec_selection.native_for(post_name, "input")
                or not self.codec_selection.native_for(post_name, "output")):
            raise RuntimeError(f"{post_name}: post frame-graph ABI is not supported")
        target_descriptor = post_descriptors["input"][0]
        if len(valid_widths) != len(source_descriptors):
            raise ValueError("attention valid widths do not match physical outputs")
        post_input_addresses = [
            self._tensor_addresses(post_record, tensor, 0)
            for tensor in post_record["inputs"]
        ]
        post_output_names = []
        post_output_requests = []
        for index, tensor in enumerate(post_record["outputs"]):
            location = self._tensor_addresses(post_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            post_output_names.append(f"post.output.{index}")
            post_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))
        for tensor, location in zip(post_record["inputs"], post_input_addresses):
            touched_ranges.append((location, int(tensor["size_per_bank"]) // 2))

        nodes = []
        for start in range(0, len(attention_records), parallel):
            stop = min(len(attention_records), start + parallel)
            nodes.extend([
                {
                    "op": "device_write",
                    "inputs": [name for values in input_names_by_call[start:stop]
                               for name in values],
                    "addresses": [address
                                  for values in input_addresses_by_call[start:stop]
                                  for address in values],
                },
                {"op": "npu_chain", "programs": programs[start:stop]},
                {
                    "op": "device_read",
                    "outputs": [name for values in output_names_by_call[start:stop]
                                for name in values],
                    "requests": [request
                                 for values in output_requests_by_call[start:stop]
                                 for request in values],
                },
            ])
        attention_output_names = [
            name for values in output_names_by_call for name in values
        ]
        nodes.extend([
            {
                "op": "attention_pack_bf16_heads",
                "inputs": attention_output_names,
                "output": "post.code",
                "source_descriptors": [asdict(item) for item in source_descriptors],
                "valid_widths": list(valid_widths),
                "target_descriptor": asdict(target_descriptor),
                "scale": float(scale),
                "heads": 6,
            },
            {
                "op": "device_write",
                "inputs": ["post.code", "post.residual"],
                "addresses": post_input_addresses,
            },
            {
                "op": "npu_chain",
                "programs": [{
                    "stage_id": post_name,
                    "base_addresses": [int(value) for value in post_record["base_addresses"]],
                    "isa_ranges": [int(value) for value in post_record["isa_ranges"]],
                }],
            },
            {
                "op": "device_read",
                "outputs": post_output_names,
                "requests": post_output_requests,
            },
        ])
        self._invalidate_ranges(touched_ranges)
        self._python_transport_api_calls += 1
        try:
            result = dict(method(
                initial, nodes, post_output_names, int(timeout_ms), self.safe_dma
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        returned = dict(result["outputs"])
        outputs = [tuple(np.ascontiguousarray(bank) for bank in returned[name])
                   for name in post_output_names]
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        expected_programs = len(attention_records) + 1
        if (int(result["nodes"]) != len(nodes)
                or int(result["programs"]) != expected_programs
                or len(npu_seconds) != expected_programs):
            raise RuntimeError("attention/post frame graph execution count mismatch")
        opcode_seconds = dict(result["opcode_seconds"])
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        self.groups += 1
        self.dispatches += expected_programs
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        self._cpp_resident_transaction_calls += 1
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        return outputs, {
            "submission_group_size": expected_programs,
            "h2c_ms": h2c_ms,
            "c2h_ms": c2h_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "bridge_ms": float(opcode_seconds.get(
                "attention_pack_bf16_heads", 0.0
            )) * 1000.0,
            "cpp_frame_graph": True,
            "frame_graph_nodes": len(nodes),
        }

    def run_fused_qkv_attention_post_frame_graph(
        self,
        fused_record: dict,
        fused_input: Union[PhysicalTensor, list[PhysicalTensor]],
        post_record: dict,
        post_residual: PhysicalTensor,
        valid_widths: list[int],
        scale: float,
        heads: int,
        timeout_ms: int,
        output_order: str = "chunk-major",
        capture_post_code: bool = False,
    ) -> tuple[list[PhysicalTensor], dict]:
        """Execute fused QKV/attention, its physical bridge, and post once."""
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError("loaded extension has no C++ frame-graph interpreter")
        fused_name = fused_record["name"]
        fused_descriptors = self.codec_selection.descriptors[fused_name]
        fused_inputs = (fused_input if isinstance(fused_input, list)
                        else [fused_input])
        if (len(fused_record["inputs"]) != len(fused_inputs)
                or len(fused_descriptors["input"]) != len(fused_inputs)
                or len(fused_record["outputs"]) != heads * 2
                or len(fused_descriptors["output"]) != heads * 2
                or not self.codec_selection.native_for(fused_name, "input")
                or not self.codec_selection.native_for(fused_name, "output")):
            raise RuntimeError(f"{fused_name}: fused attention ABI is not supported")
        if len(valid_widths) != heads * 2:
            raise ValueError("fused attention valid widths must be head-major pairs")

        post_name = post_record["name"]
        post_descriptors = self.codec_selection.descriptors[post_name]
        if (len(post_record["inputs"]) != 2
                or len(post_descriptors["input"]) != 2
                or not post_record["outputs"]
                or not self.codec_selection.native_for(post_name, "input")
                or not self.codec_selection.native_for(post_name, "output")):
            raise RuntimeError(f"{post_name}: post frame-graph ABI is not supported")

        fused_input_names = [
            f"fused_attention.input.{index}"
            for index in range(len(fused_inputs))
        ]
        fused_input_addresses = [
            self._tensor_addresses(fused_record, tensor, 0)
            for tensor in fused_record["inputs"]
        ]
        fused_output_names = []
        fused_output_requests = []
        touched_ranges = [
            (address, int(tensor["size_per_bank"]) // 2)
            for tensor, address in zip(
                fused_record["inputs"], fused_input_addresses
            )
        ]
        for index, tensor in enumerate(fused_record["outputs"]):
            location = self._tensor_addresses(fused_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            fused_output_names.append(f"fused_attention.output.{index}")
            fused_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))

        # The QKV+attention compiler schedules outputs chunk-major, while the
        # attention-only compiler preserves head-major output pairs.  Reorder
        # tensor references only; physical payloads are never copied here.
        if output_order == "chunk-major":
            head_major_indices = [
                index for head in range(heads)
                for index in (head, heads + head)
            ]
        elif output_order == "head-major":
            head_major_indices = list(range(heads * 2))
        else:
            raise ValueError(f"unsupported fused attention output order: {output_order}")
        pack_names = [fused_output_names[index]
                      for index in head_major_indices]
        pack_descriptors = [fused_descriptors["output"][index]
                            for index in head_major_indices]
        target_descriptor = post_descriptors["input"][0]

        post_input_addresses = [
            self._tensor_addresses(post_record, tensor, 0)
            for tensor in post_record["inputs"]
        ]
        post_output_names = []
        post_output_requests = []
        for tensor, location in zip(post_record["inputs"], post_input_addresses):
            touched_ranges.append((location, int(tensor["size_per_bank"]) // 2))
        for index, tensor in enumerate(post_record["outputs"]):
            location = self._tensor_addresses(post_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            post_output_names.append(f"post.output.{index}")
            post_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))

        initial = {
            **dict(zip(fused_input_names, fused_inputs)),
            "post.residual": post_residual,
        }
        nodes = [
            {
                "op": "device_write",
                "inputs": fused_input_names,
                "addresses": fused_input_addresses,
            },
            {
                "op": "npu_chain",
                "programs": [{
                    "stage_id": fused_name,
                    "base_addresses": [int(value) for value in
                                       fused_record["base_addresses"]],
                    "isa_ranges": [int(value) for value in
                                   fused_record["isa_ranges"]],
                }],
            },
            {
                "op": "device_read",
                "outputs": fused_output_names,
                "requests": fused_output_requests,
            },
            {
                "op": "attention_pack_bf16_heads",
                "inputs": pack_names,
                "output": "post.code",
                "source_descriptors": [asdict(item)
                                       for item in pack_descriptors],
                "valid_widths": list(valid_widths),
                "target_descriptor": asdict(target_descriptor),
                "scale": float(scale),
                "heads": heads,
            },
            {
                "op": "device_write",
                "inputs": ["post.code", "post.residual"],
                "addresses": post_input_addresses,
            },
            {
                "op": "npu_chain",
                "programs": [{
                    "stage_id": post_name,
                    "base_addresses": [int(value) for value in
                                       post_record["base_addresses"]],
                    "isa_ranges": [int(value) for value in
                                   post_record["isa_ranges"]],
                }],
            },
            {
                "op": "device_read",
                "outputs": post_output_names,
                "requests": post_output_requests,
            },
        ]
        self._invalidate_ranges(touched_ranges)
        self._python_transport_api_calls += 1
        try:
            fetch_names = list(post_output_names)
            if capture_post_code:
                fetch_names.append("post.code")
            result = dict(method(
                initial, nodes, fetch_names, int(timeout_ms), self.safe_dma
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        if (int(result["nodes"]) != len(nodes)
                or int(result["programs"]) != 2
                or len(result["npu_seconds"]) != 2):
            raise RuntimeError("fused attention/post frame graph count mismatch")
        returned = dict(result["outputs"])
        outputs = [tuple(np.ascontiguousarray(bank)
                         for bank in returned[name])
                   for name in post_output_names]
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        opcode_seconds = dict(result["opcode_seconds"])
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        self.groups += 1
        self.dispatches += 2
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        self._cpp_resident_transaction_calls += 1
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        metadata = {
            "submission_group_size": 2,
            "h2c_ms": h2c_ms,
            "c2h_ms": c2h_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "bridge_ms": float(opcode_seconds.get(
                "attention_pack_bf16_heads", 0.0
            )) * 1000.0,
            "cpp_frame_graph": True,
            "frame_graph_nodes": len(nodes),
        }
        if capture_post_code:
            metadata["captured_post_code"] = tuple(
                np.ascontiguousarray(bank)
                for bank in returned["post.code"]
            )
        return outputs, metadata

    def run_multi_attention6_post_frame_graph(
        self,
        attention_record: dict,
        attention_calls: list[list[PhysicalTensor]],
        post_record: dict,
        post_residual: PhysicalTensor | DeviceTensorHandle,
        valid_widths: list[int],
        scale: float,
        heads: int,
        timeout_ms: int,
        capture_post_code: bool = False,
        *,
        post_offset_units: int = 0,
        norm_record: dict | None = None,
        norm_offset_units: int = 0,
    ) -> tuple[list[PhysicalTensor], dict]:
        """Run one six-head program repeatedly, then assemble and project.

        The 518 graph has three two-query-chunk calls per head.  Compiling the
        six independent heads together changes 18 attention dispatches into
        three without changing the calibrated A8 ABI or attention arithmetic.
        Calls reuse the same FM addresses sequentially; every result is copied
        into a frame-graph tensor before the following call overwrites FM.
        """
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError("loaded extension has no C++ frame-graph interpreter")
        if heads < 1 or not attention_calls:
            raise ValueError("multi-attention6 requires heads and calls")

        attention_name = attention_record["name"]
        attention_descriptors = self.codec_selection.descriptors[attention_name]
        inputs_per_call = heads * 4
        outputs_per_call = heads * 2
        if (len(attention_record["inputs"]) != inputs_per_call
                or len(attention_record["outputs"]) != outputs_per_call
                or len(attention_descriptors["input"]) != inputs_per_call
                or len(attention_descriptors["output"]) != outputs_per_call
                or not self.codec_selection.native_for(attention_name, "input")
                or not self.codec_selection.native_for(attention_name, "output")):
            raise RuntimeError(
                f"{attention_name}: multi-attention6 ABI is not supported"
            )
        if any(len(call) != inputs_per_call for call in attention_calls):
            raise ValueError("multi-attention6 physical inputs are not call-aligned")
        if len(valid_widths) != len(attention_calls) * outputs_per_call:
            raise ValueError("multi-attention6 valid widths are not head/call-aligned")

        resident_post = isinstance(post_residual, DeviceTensorHandle)
        if resident_post != (norm_record is not None):
            raise ValueError(
                "resident attention/post requires a chained norm record"
            )
        self._validate_offset(post_record, post_offset_units)
        if norm_record is not None:
            self._validate_offset(norm_record, norm_offset_units)
        initial = {} if resident_post else {"post.residual": post_residual}
        nodes = []
        touched_ranges = []
        output_names_by_call = []
        output_requests = []
        input_addresses = [
            self._tensor_addresses(attention_record, tensor, 0)
            for tensor in attention_record["inputs"]
        ]
        output_addresses = [
            self._tensor_addresses(attention_record, tensor, 0)
            for tensor in attention_record["outputs"]
        ]
        for tensor, address in zip(attention_record["inputs"], input_addresses):
            touched_ranges.append((address, int(tensor["size_per_bank"]) // 2))
        for tensor, address in zip(attention_record["outputs"], output_addresses):
            touched_ranges.append((address, int(tensor["size_per_bank"]) // 2))
        program = {
            "stage_id": attention_name,
            "base_addresses": [int(value) for value in attention_record["base_addresses"]],
            "isa_ranges": [int(value) for value in attention_record["isa_ranges"]],
        }
        for call_index, packed in enumerate(attention_calls):
            input_names = [
                f"attention6.input.{call_index}.{index}"
                for index in range(inputs_per_call)
            ]
            for name, physical in zip(input_names, packed):
                initial[name] = physical
            output_names = [
                f"attention6.output.{call_index}.{index}"
                for index in range(outputs_per_call)
            ]
            output_names_by_call.append(output_names)
            requests = []
            for tensor, address in zip(
                    attention_record["outputs"], output_addresses):
                size = int(tensor["size_per_bank"]) // 2
                requests.append([
                    (bank, address[bank], size) for bank in range(2)
                ])
            output_requests.append(requests)
            nodes.extend([
                {"op": "device_write", "inputs": input_names,
                 "addresses": input_addresses},
                {"op": "npu_chain", "programs": [program]},
                {"op": "device_read", "outputs": output_names,
                 "requests": requests},
            ])

        # Compiler outputs are call-major/head-major/chunk-major.  The bridge
        # consumes head-major/call-major/chunk-major so each head's 1370 rows
        # remain contiguous when the three calls are concatenated.
        pack_names = []
        pack_descriptors = []
        pack_widths = []
        calls = len(attention_calls)
        for head in range(heads):
            for call_index in range(calls):
                for chunk in range(2):
                    output_index = head * 2 + chunk
                    pack_names.append(output_names_by_call[call_index][output_index])
                    pack_descriptors.append(
                        attention_descriptors["output"][output_index]
                    )
                    width_index = (head * calls + call_index) * 2 + chunk
                    pack_widths.append(int(valid_widths[width_index]))

        post_name = post_record["name"]
        post_descriptors = self.codec_selection.descriptors[post_name]
        if (len(post_record["inputs"]) != 2
                or len(post_descriptors["input"]) != 2
                or not post_record["outputs"]
                or not self.codec_selection.native_for(post_name, "input")
                or not self.codec_selection.native_for(post_name, "output")):
            raise RuntimeError(f"{post_name}: post frame-graph ABI is not supported")
        post_input_addresses = [
            self._tensor_addresses(post_record, tensor, post_offset_units)
            for tensor in post_record["inputs"]
        ]
        # input 0 is written by the physical attention bridge.  A resident
        # input 1 already lives in FM and must not be invalidated or uploaded.
        touched_ranges.append((
            post_input_addresses[0],
            int(post_record["inputs"][0]["size_per_bank"]) // 2,
        ))
        if resident_post:
            self._validate_handle(
                post_residual, post_record, post_record["inputs"][1],
                post_descriptors["input"][1], post_offset_units, None,
            )
        else:
            touched_ranges.append((
                post_input_addresses[1],
                int(post_record["inputs"][1]["size_per_bank"]) // 2,
            ))
        post_output_names = []
        post_output_requests = []
        for index, tensor in enumerate(post_record["outputs"]):
            address = self._tensor_addresses(
                post_record, tensor, post_offset_units
            )
            size = int(tensor["size_per_bank"]) // 2
            post_output_names.append(f"post.output.{index}")
            post_output_requests.append([
                (bank, address[bank], size) for bank in range(2)
            ])
            touched_ranges.append((address, size))
        post_program = {
            "stage_id": post_name,
            "base_addresses": [
                int(value) + (post_offset_units if index == 4 else 0)
                for index, value in enumerate(post_record["base_addresses"])
            ],
            "isa_ranges": [int(value) for value in post_record["isa_ranges"]],
        }
        post_write_inputs = ["post.code"]
        post_write_addresses = [post_input_addresses[0]]
        if not resident_post:
            post_write_inputs.append("post.residual")
            post_write_addresses.append(post_input_addresses[1])
        programs = [post_program]
        norm_output_names = []
        norm_output_requests = []
        if norm_record is not None:
            norm_name = norm_record["name"]
            norm_descriptors = self.codec_selection.descriptors[norm_name]
            if (len(norm_record["inputs"]) != 1
                    or not norm_record["outputs"]
                    or len(norm_descriptors["input"]) != 1
                    or not self.codec_selection.native_for(norm_name, "input")
                    or not self.codec_selection.native_for(norm_name, "output")):
                raise RuntimeError(
                    f"{norm_name}: resident norm frame-graph ABI is not supported"
                )
            post_output_address = self._tensor_addresses(
                post_record, post_record["outputs"][0], post_offset_units
            )
            norm_input_address = self._tensor_addresses(
                norm_record, norm_record["inputs"][0], norm_offset_units
            )
            if post_output_address != norm_input_address:
                raise RuntimeError("post/norm resident connection address mismatch")
            if (post_descriptors["output"][0].storage_identity()
                    != norm_descriptors["input"][0].storage_identity()):
                raise RuntimeError("post/norm resident connection storage ABI mismatch")
            norm_bases = [int(value) for value in norm_record["base_addresses"]]
            norm_bases[4] += norm_offset_units
            programs.append({
                "stage_id": norm_name, "base_addresses": norm_bases,
                "isa_ranges": [int(value) for value in norm_record["isa_ranges"]],
            })
            for index, tensor in enumerate(norm_record["outputs"]):
                address = self._tensor_addresses(
                    norm_record, tensor, norm_offset_units
                )
                size = int(tensor["size_per_bank"]) // 2
                norm_output_names.append(f"norm.output.{index}")
                norm_output_requests.append([
                    (bank, address[bank], size) for bank in range(2)
                ])
                touched_ranges.append((address, size))
        nodes.extend([
            {
                "op": "attention_pack_bf16_heads", "inputs": pack_names,
                "output": "post.code",
                "source_descriptors": [asdict(item) for item in pack_descriptors],
                "valid_widths": pack_widths,
                "target_descriptor": asdict(post_descriptors["input"][0]),
                "scale": float(scale), "heads": heads,
            },
            {"op": "device_write", "inputs": post_write_inputs,
             "addresses": post_write_addresses},
            {"op": "npu_chain", "programs": programs},
            {"op": "device_read",
             "outputs": [*post_output_names, *norm_output_names],
             "requests": [*post_output_requests, *norm_output_requests]},
        ])

        if resident_post and any(
                any(self._overlaps(
                    post_residual.bank_addresses[bank],
                    post_residual.bytes_per_bank, addresses[bank], size,
                ) for bank in range(2))
                for addresses, size in touched_ranges):
            raise RuntimeError(
                "attention/post frame graph overwrites its forwarded residual"
            )
        self._invalidate_ranges(touched_ranges)
        self._python_transport_api_calls += 1
        fetch_names = [*post_output_names, *norm_output_names]
        if capture_post_code:
            fetch_names.append("post.code")
        try:
            result = dict(method(
                initial, nodes, fetch_names, int(timeout_ms), self.safe_dma
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        expected_programs = calls + len(programs)
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        if (int(result["nodes"]) != len(nodes)
                or int(result["programs"]) != expected_programs
                or len(npu_seconds) != expected_programs):
            raise RuntimeError("multi-attention6 frame graph count mismatch")
        returned = dict(result["outputs"])
        outputs = [tuple(np.ascontiguousarray(bank) for bank in returned[name])
                   for name in fetch_names]
        opcode_seconds = dict(result["opcode_seconds"])
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        self.groups += 1
        self.dispatches += expected_programs
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        self._cpp_resident_transaction_calls += 1
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        metadata = {
            "submission_group_size": expected_programs,
            "h2c_ms": h2c_ms, "c2h_ms": c2h_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "bridge_ms": float(opcode_seconds.get(
                "attention_pack_bf16_heads", 0.0
            )) * 1000.0,
            "cpp_frame_graph": True, "frame_graph_nodes": len(nodes),
            "attention6_calls": calls,
            "post_output_count": len(post_output_names),
            "resident_post_norm": resident_post,
        }
        if capture_post_code:
            metadata["captured_post_code"] = tuple(
                np.ascontiguousarray(bank) for bank in returned["post.code"]
            )
        return outputs, metadata

    def run_qkv_attention6_post_frame_graph(
        self,
        qkv_record: dict,
        qkv_input: PhysicalTensor,
        attention_record: dict,
        post_record: dict,
        post_residual: PhysicalTensor,
        qkv_scales: list[float],
        valid_widths: list[int],
        post_scale: float,
        heads: int,
        timeout_ms: int,
    ) -> tuple[list[PhysicalTensor], dict]:
        """Run QKV, exact BF16/A8 bridge, attention6 and post in one call."""
        method = getattr(self.transport, "run_frame_graph", None)
        if not callable(method):
            raise RuntimeError("loaded extension has no C++ frame-graph interpreter")
        qkv_name = qkv_record["name"]
        attention_name = attention_record["name"]
        post_name = post_record["name"]
        qkv_descriptors = self.codec_selection.descriptors[qkv_name]
        attention_descriptors = self.codec_selection.descriptors[attention_name]
        post_descriptors = self.codec_selection.descriptors[post_name]
        if (len(qkv_record["inputs"]) != 1
                or len(qkv_record["outputs"]) != 3
                or len(qkv_descriptors["input"]) != 1
                or len(qkv_descriptors["output"]) != 3
                or not self.codec_selection.native_for(qkv_name, "input")
                or not self.codec_selection.native_for(qkv_name, "output")):
            raise RuntimeError(f"{qkv_name}: QKV frame-graph ABI is not supported")
        if (len(attention_record["inputs"]) != heads * 4
                or len(attention_record["outputs"]) != heads * 2
                or len(attention_descriptors["input"]) != heads * 4
                or len(attention_descriptors["output"]) != heads * 2
                or not self.codec_selection.native_for(attention_name, "input")
                or not self.codec_selection.native_for(attention_name, "output")):
            raise RuntimeError(
                f"{attention_name}: attention6 frame-graph ABI is not supported"
            )
        if (len(post_record["inputs"]) != 2
                or len(post_descriptors["input"]) != 2
                or not post_record["outputs"]
                or not self.codec_selection.native_for(post_name, "input")
                or not self.codec_selection.native_for(post_name, "output")):
            raise RuntimeError(f"{post_name}: post frame-graph ABI is not supported")
        if len(qkv_scales) != heads * 3 or len(valid_widths) != heads * 2:
            raise ValueError("QKV scales or attention valid widths are not head-aligned")

        touched_ranges = []
        qkv_input_address = self._tensor_addresses(
            qkv_record, qkv_record["inputs"][0], 0
        )
        touched_ranges.append((
            qkv_input_address, int(qkv_record["inputs"][0]["size_per_bank"]) // 2
        ))
        qkv_output_names, qkv_output_requests = [], []
        for index, tensor in enumerate(qkv_record["outputs"]):
            location = self._tensor_addresses(qkv_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            qkv_output_names.append(f"qkv.output.{index}")
            qkv_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))

        attention_input_names = [
            f"attention6.input.{index}"
            for index in range(len(attention_record["inputs"]))
        ]
        attention_input_addresses = []
        for tensor in attention_record["inputs"]:
            location = self._tensor_addresses(attention_record, tensor, 0)
            attention_input_addresses.append(location)
            touched_ranges.append((location, int(tensor["size_per_bank"]) // 2))
        attention_output_names, attention_output_requests = [], []
        for index, tensor in enumerate(attention_record["outputs"]):
            location = self._tensor_addresses(attention_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            attention_output_names.append(f"attention6.output.{index}")
            attention_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))

        post_input_addresses = []
        for tensor in post_record["inputs"]:
            location = self._tensor_addresses(post_record, tensor, 0)
            post_input_addresses.append(location)
            touched_ranges.append((location, int(tensor["size_per_bank"]) // 2))
        post_output_names, post_output_requests = [], []
        for index, tensor in enumerate(post_record["outputs"]):
            location = self._tensor_addresses(post_record, tensor, 0)
            size = int(tensor["size_per_bank"]) // 2
            post_output_names.append(f"post.output.{index}")
            post_output_requests.append([
                (bank, location[bank], size) for bank in range(2)
            ])
            touched_ranges.append((location, size))

        def program(record: dict) -> dict:
            return {
                "stage_id": record["name"],
                "base_addresses": [int(value) for value in record["base_addresses"]],
                "isa_ranges": [int(value) for value in record["isa_ranges"]],
            }

        nodes = [
            {"op": "device_write", "inputs": ["qkv.input"],
             "addresses": [qkv_input_address]},
            {"op": "npu_chain", "programs": [program(qkv_record)]},
            {"op": "device_read", "outputs": qkv_output_names,
             "requests": qkv_output_requests},
            {"op": "qkv_pack_bf16_attention6", "inputs": qkv_output_names,
             "outputs": attention_input_names,
             "source_descriptors": [asdict(item)
                                    for item in qkv_descriptors["output"]],
             "target_descriptors": [asdict(item)
                                    for item in attention_descriptors["input"]],
             "scales": [float(value) for value in qkv_scales],
             "heads": int(heads)},
            {"op": "device_write", "inputs": attention_input_names,
             "addresses": attention_input_addresses},
            {"op": "npu_chain", "programs": [program(attention_record)]},
            {"op": "device_read", "outputs": attention_output_names,
             "requests": attention_output_requests},
            {"op": "attention_pack_bf16_heads",
             "inputs": attention_output_names, "output": "post.code",
             "source_descriptors": [asdict(item)
                                    for item in attention_descriptors["output"]],
             "valid_widths": [int(value) for value in valid_widths],
             "target_descriptor": asdict(post_descriptors["input"][0]),
             "scale": float(post_scale), "heads": int(heads)},
            {"op": "device_write", "inputs": ["post.code", "post.residual"],
             "addresses": post_input_addresses},
            {"op": "npu_chain", "programs": [program(post_record)]},
            {"op": "device_read", "outputs": post_output_names,
             "requests": post_output_requests},
        ]
        self._invalidate_ranges(touched_ranges)
        self._python_transport_api_calls += 1
        try:
            result = dict(method(
                {"qkv.input": qkv_input, "post.residual": post_residual},
                nodes, post_output_names, int(timeout_ms), self.safe_dma,
            ))
        except Exception:
            self._invalidate_all_handles()
            raise
        if (int(result["nodes"]) != len(nodes)
                or int(result["programs"]) != 3
                or len(result["npu_seconds"]) != 3):
            raise RuntimeError("QKV/attention6/post frame graph count mismatch")
        returned = dict(result["outputs"])
        outputs = [tuple(np.ascontiguousarray(bank)
                         for bank in returned[name])
                   for name in post_output_names]
        npu_seconds = [float(value) for value in result["npu_seconds"]]
        opcode_seconds = dict(result["opcode_seconds"])
        h2c_ms = float(opcode_seconds.get("device_write", 0.0)) * 1000.0
        c2h_ms = float(opcode_seconds.get("device_read", 0.0)) * 1000.0
        qkv_bridge_ms = float(opcode_seconds.get(
            "qkv_pack_bf16_attention6", 0.0
        )) * 1000.0
        post_bridge_ms = float(opcode_seconds.get(
            "attention_pack_bf16_heads", 0.0
        )) * 1000.0
        self.groups += 1
        self.dispatches += 3
        self.h2c_seconds += h2c_ms / 1000.0
        self.c2h_seconds += c2h_ms / 1000.0
        self._cpp_resident_transaction_calls += 1
        self._frame_graph_nodes += len(nodes)
        self._frame_graph_peak_tensors = max(
            self._frame_graph_peak_tensors, int(result["peak_tensors"])
        )
        self._frame_graph_wall_seconds += float(result["wall_seconds"])
        return outputs, {
            "submission_group_size": 3, "h2c_ms": h2c_ms,
            "c2h_ms": c2h_ms,
            "npu_ms": [value * 1000.0 for value in npu_seconds],
            "bridge_ms": qkv_bridge_ms + post_bridge_ms,
            "qkv_bridge_ms": qkv_bridge_ms,
            "post_bridge_ms": post_bridge_ms,
            "cpp_frame_graph": True, "frame_graph_nodes": len(nodes),
        }

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
        upload_ranges: list[tuple[tuple[int, int], int]] = []
        programs = []
        output_requests: list[tuple[int, int, int]] = []
        output_ranges: list[tuple[tuple[int, int], int]] = []
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
                addresses = self._tensor_addresses(record, tensor, slot_units)
                upload_ranges.append((addresses, combined_size // 2))
                for bank, half in enumerate(halves):
                    h2c_requests.append((bank, addresses[bank], half))
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
                addresses = self._tensor_addresses(record, tensor, slot_units)
                output_ranges.append((addresses, half_size))
                for bank in range(2):
                    output_requests.append((bank, addresses[bank], half_size))
                    downloaded_bytes += half_size

        self._invalidate_ranges(upload_ranges)
        try:
            raw, npu_seconds, h2c_ms, c2h_ms, cpp_transaction = (
                self._execute_transaction(
                    h2c_requests, programs, output_requests, timeout_ms
                )
            )
        except Exception:
            self._invalidate_all_handles()
            raise
        self._invalidate_ranges(output_ranges)

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
            "cpp_resident_transaction": cpp_transaction,
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
            "device_tensor_live_handles": len(self._live_handles),
            "device_tensor_handle_creations": self._resident_handle_creations,
            "device_tensor_handle_invalidations": self._resident_handle_invalidations,
            "device_tensor_forwarded_inputs": self._resident_forwarded_inputs,
            "device_tensor_connections": self._resident_connections,
            "device_tensor_frame_epoch": self._frame_epoch,
            "python_transport_api_calls": self._python_transport_api_calls,
            "cpp_resident_transaction_calls":
                self._cpp_resident_transaction_calls,
            "frame_graph_nodes": self._frame_graph_nodes,
            "frame_graph_peak_tensors": self._frame_graph_peak_tensors,
            "frame_graph_wall_ms": self._frame_graph_wall_seconds * 1000.0,
        })
        return result

    def reset_frame_stats(self) -> None:
        self._invalidate_all_handles()
        self._frame_epoch += 1
        self.codec_selection.reset_stats()
        self.groups = 0
        self.dispatches = 0
        self.h2c_seconds = 0.0
        self.c2h_seconds = 0.0
        self._resident_handle_creations = 0
        self._resident_handle_invalidations = 0
        self._resident_forwarded_inputs = 0
        self._resident_connections = 0
        self._python_transport_api_calls = 0
        self._cpp_resident_transaction_calls = 0
        self._frame_graph_nodes = 0
        self._frame_graph_peak_tensors = 0
        self._frame_graph_wall_seconds = 0.0
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
