#!/usr/bin/env python3
"""Execute the Depth Anything V2 ViT-S token tail with one resident U250 bank."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

if __package__:
    from . import u250_cpp_mapped_runtime as mapped_runtime
    from .u250_cpp_mapped_runtime import LayoutCodecSelection, get_cached_cpp_runtime
    from .u250_host_executor import HostExecutorSelection
    from .u250_host_profile import HostProfiler
    from .u250_layout_descriptors import build_case_descriptors
else:
    import u250_cpp_mapped_runtime as mapped_runtime
    from u250_cpp_mapped_runtime import LayoutCodecSelection, get_cached_cpp_runtime
    from u250_host_executor import HostExecutorSelection
    from u250_host_profile import HostProfiler
    from u250_layout_descriptors import build_case_descriptors


_CFG_REGISTRY_CACHE: dict[str, "CfgCodecRegistry"] = {}
_STATIC_JSON_CACHE: dict[str, tuple[tuple[int, ...], bytes, object]] = {}
_HOST_PARAMS_CACHE: dict[str, tuple[tuple[int, ...], dict[str, np.ndarray]]] = {}
_BANK_IMAGE_CACHE: dict[str, tuple[tuple[int, ...], np.ndarray]] = {}
_NPZ_YAML_PATHS: tuple[str, str] | None = None
ENCODER_INTERNAL_STAGES = (
    "norm1", "qkv", "attention", "attention_branch", "post",
    "norm2", "fc1", "gelu", "fc2",
)


def _file_identity(path: Path) -> tuple[int, ...]:
    """Return a cache identity that changes on an ordinary file replacement."""
    status = path.stat()
    return (
        int(status.st_dev), int(status.st_ino), int(status.st_size),
        int(status.st_mtime_ns), int(status.st_ctime_ns),
    )


def load_json_cached(path: Path) -> tuple[object, bytes, bool]:
    """Load immutable JSON once per resident process, with stat invalidation."""
    resolved = path.resolve()
    key = str(resolved)
    identity = _file_identity(resolved)
    cached = _STATIC_JSON_CACHE.get(key)
    if cached is not None and cached[0] == identity:
        return cached[2], cached[1], True
    raw = resolved.read_bytes()
    value = json.loads(raw)
    _STATIC_JSON_CACHE[key] = (identity, raw, value)
    return value, raw, False


def load_host_params_cached(path: Path) -> tuple[dict[str, np.ndarray], bool]:
    """Reuse constant arrays while returning a fresh frame-local environment."""
    resolved = path.resolve()
    key = str(resolved)
    identity = _file_identity(resolved)
    cached = _HOST_PARAMS_CACHE.get(key)
    reused = cached is not None and cached[0] == identity
    if not reused:
        with np.load(resolved, allow_pickle=False) as archive:
            values = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        _HOST_PARAMS_CACHE[key] = (identity, values)
        cached = _HOST_PARAMS_CACHE[key]
    # Decoder steps install frame-local tensors into this mapping.
    return dict(cached[1]), reused


def load_bank_image_cached(path: Path) -> tuple[np.ndarray, bool]:
    """Reuse linked bank bytes after the runtime has made the image resident."""
    resolved = path.resolve()
    key = str(resolved)
    identity = _file_identity(resolved)
    cached = _BANK_IMAGE_CACHE.get(key)
    reused = cached is not None and cached[0] == identity
    if not reused:
        value = np.fromfile(resolved, dtype=np.uint8)
        value.flags.writeable = False
        _BANK_IMAGE_CACHE[key] = (identity, value)
        cached = _BANK_IMAGE_CACHE[key]
    return cached[1], reused


def parse_encoder_internal_target(value: str) -> tuple[int, str]:
    """Parse LAYER:STAGE for one causal encoder intervention."""
    try:
        layer_text, stage = value.split(":", 1)
        layer = int(layer_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "encoder internal target must be LAYER:STAGE"
        ) from exc
    if not 0 <= layer <= 11:
        raise argparse.ArgumentTypeError("encoder internal layer must be within [0, 11]")
    if stage not in ENCODER_INTERNAL_STAGES:
        raise argparse.ArgumentTypeError(
            "encoder internal stage must be one of "
            + ", ".join(ENCODER_INTERNAL_STAGES)
        )
    return layer, stage


def parse_encoder_attention_head_target(value: str) -> tuple[int, int]:
    """Parse LAYER:HEAD for one single-head attention intervention."""
    try:
        layer_text, head_text = value.split(":", 1)
        layer, head = int(layer_text), int(head_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "encoder attention-head target must be LAYER:HEAD"
        ) from exc
    if not 0 <= layer <= 11:
        raise argparse.ArgumentTypeError("encoder attention layer must be within [0, 11]")
    if not 0 <= head <= 5:
        raise argparse.ArgumentTypeError("encoder attention head must be within [0, 5]")
    return layer, head


def load_reference_replacement(
    archive: np.lib.npyio.NpzFile, key: str, expected: np.ndarray
) -> np.ndarray:
    """Load one finite FP32 capture and enforce the live tensor contract."""
    if key not in archive.files:
        raise KeyError(f"replacement trace does not contain {key}")
    reference = np.asarray(archive[key], dtype=np.float32)
    if reference.shape != expected.shape:
        raise ValueError(
            f"replacement {key} shape {reference.shape} != live {expected.shape}"
        )
    if not np.isfinite(reference).all():
        raise ValueError(f"replacement {key} contains non-finite values")
    return np.ascontiguousarray(reference)


def pad_nchw_width(value: np.ndarray, width: int) -> np.ndarray:
    """Right-pad one logical NCHW tensor with zeros to a physical width."""
    if value.ndim != 4:
        raise ValueError(f"width padding requires NCHW rank 4, got {value.shape}")
    logical_width = int(value.shape[3])
    if width < logical_width:
        raise ValueError(
            f"physical width {width} is smaller than logical width {logical_width}"
        )
    if width == logical_width:
        return np.ascontiguousarray(value)
    result = np.zeros((*value.shape[:3], width), dtype=value.dtype)
    result[:, :, :, :logical_width] = value
    return result


def crop_nchw_width(value: np.ndarray, width: int) -> np.ndarray:
    """Crop a width-padded NCHW hardware result back to its logical width."""
    if value.ndim != 4 or not 0 < width <= value.shape[3]:
        raise ValueError(f"invalid NCHW width crop {width} for {value.shape}")
    return np.ascontiguousarray(value[:, :, :, :width])


def fp32_attention_head(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                        rows_per_chunk: int = 256) -> np.ndarray:
    """Execute one [tokens, channels] attention head in bounded FP32 chunks."""
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 2:
        raise ValueError("FP32 attention Q/K/V must have identical rank-2 shapes")
    if not 0 < rows_per_chunk <= 256:
        raise ValueError("FP32 attention chunk rows must be within [1, 256]")
    key = k.T
    chunks = []
    for begin in range(0, q.shape[0], rows_per_chunk):
        logits = q[begin:begin + rows_per_chunk] @ key
        logits -= np.max(logits, axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        chunks.append(probability @ v)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0)[None, None])


def bf16_quantization_surrogate(value: np.ndarray, scale: float,
                                quantize) -> np.ndarray:
    """Encode FP32 values so a BF16 bridge preserves their target INT8 code.

    Physical attention fusion consumes compiler BF16 output descriptors.  A
    host-computed head must therefore cross a BF16 container before the fused
    post-attention quantizer.  Encoding the already selected INT8 code at the
    centre of its quantization bin prevents that container conversion from
    changing the legacy FP32-to-INT8 result.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("BF16 quantization surrogate scale must be positive")
    code = np.ascontiguousarray(quantize(value, scale), dtype=np.int8)
    surrogate = np.ascontiguousarray(
        code.astype(np.float32) * np.float32(scale), dtype=np.float32
    )
    # Mirror fpgaDmaBatch's round-to-nearest-even FP32 -> BF16 conversion and
    # fail closed if an unusual scale cannot preserve every selected code.
    bits = surrogate.view(np.uint32)
    rounded_bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(
        0xFFFF0000
    )
    rounded = rounded_bits.view(np.float32)
    roundtrip = np.clip(np.rint(rounded / np.float32(scale)), -128, 127).astype(
        np.int8
    )
    if not np.array_equal(roundtrip, code):
        raise RuntimeError("BF16 surrogate does not preserve target INT8 codes")
    return surrogate


def encoder_resident_offset_plan(records: dict[str, dict], block: dict,
                                 workspace_bytes_per_bank: int,
                                 x_begin: int | None = None) -> dict[str, int]:
    """Derive and validate a relocated encoder residual/post/norm2 plan."""
    norm = records[block["host_norm1"]["npu_core"]]
    post = records[block["post_attention"]["kernel"]]
    low_workspace_spans = [
        mapped_runtime.record_span_per_bank(records[block["qkv"]["kernel"]]),
        *(mapped_runtime.record_span_per_bank(records[head["kernel"]])
          for head in block["attention"]["heads"]),
    ]
    # The fused FC1/GELU/FC2 frame graph assigns every FC1 shard a distinct
    # low-workspace slot before it materializes the outputs on the host.  A
    # decoder capture must live above the complete sharded span, not merely
    # above one FC1 record.  Otherwise the last shard silently overwrites (and
    # correctly invalidates) an earlier block's retained DeviceTensorHandle.
    mlp = block.get("mlp", {})
    fc1_names = list(mlp.get("fc1_kernels", ()))
    if fc1_names:
        fc1_stride = max(
            mapped_runtime.record_span_per_bank(records[name])
            for name in fc1_names
        )
        fc1_slots = min(
            len(fc1_names), workspace_bytes_per_bank // fc1_stride
        )
        low_workspace_spans.append(fc1_slots * fc1_stride)
    fc2_name = mlp.get("fc2_kernel")
    if fc2_name:
        low_workspace_spans.append(
            mapped_runtime.record_span_per_bank(records[fc2_name])
        )
    low_end = max(low_workspace_spans) // mapped_runtime.ADDRESS_UNIT_BYTES_PER_BANK
    x_units = (int(norm["inputs"][0]["size_per_bank"]) // 2
               // mapped_runtime.ADDRESS_UNIT_BYTES_PER_BANK)
    selected_x = low_end if x_begin is None else int(x_begin)
    post_base = selected_x - int(post["inputs"][1]["address"])
    post_begin = post_base + int(post["outputs"][0]["address"])
    norm2_base = post_begin - int(norm["inputs"][0]["address"])
    norm2_end = norm2_base + int(norm["outputs"][0]["address"]) + x_units
    if (selected_x < low_end or post_base < 0
            or post_begin != selected_x + x_units
            or norm2_end * mapped_runtime.ADDRESS_UNIT_BYTES_PER_BANK
            > workspace_bytes_per_bank):
        raise RuntimeError(
            f"layer {block['layer']}: manifest is incompatible with resident FM plan"
        )
    return {
        "norm1": selected_x, "post": post_base, "norm2": norm2_base,
        "x_begin": selected_x, "x_end": selected_x + x_units,
        "scratch_end": norm2_end, "low_end": low_end, "tensor_units": x_units,
    }


def decoder_capture_offset_plan(records: dict[str, dict], contract: dict,
                                workspace_bytes_per_bank: int) -> dict[int, int]:
    """Place four persistent capture tensors around three-layer scratch spans."""
    captures = [int(block["layer"]) for block in contract["encoder"]
                if block.get("capture_for_decoder")]
    if captures != [2, 5, 8, 11]:
        raise RuntimeError(f"decoder capture layers are not qualified: {captures}")
    reference = contract["encoder"][3]
    standard = encoder_resident_offset_plan(
        records, reference, workspace_bytes_per_bank)
    low_end = standard["low_end"]
    units = standard["tensor_units"]
    plan = {
        2: low_end + 3 * units,
        5: low_end + 6 * units,
        8: low_end + 9 * units,
        11: low_end + 11 * units,
    }
    norm_name = reference["host_norm1"]["npu_core"]
    norm = records[norm_name]
    output_units = int(norm["outputs"][0]["address"]) + units
    end = plan[11] + output_units
    capacity = workspace_bytes_per_bank // mapped_runtime.ADDRESS_UNIT_BYTES_PER_BANK
    intervals = [(layer, begin, begin + units) for layer, begin in plan.items()]
    for index, (layer, begin, finish) in enumerate(intervals):
        if finish > capacity or any(
                max(begin, other_begin) < min(finish, other_finish)
                for _, other_begin, other_finish in intervals[index + 1:]):
            raise RuntimeError(f"decoder capture layer {layer} has an invalid FM interval")
    if end > capacity:
        raise RuntimeError("decoder LayerNorm output exceeds resident FM workspace")
    return plan


_CFG_TENSOR_PATTERN = re.compile(
    r"^(?P<output>Output )?Address: (?P<address>\d+) \([^)]*\) "
    r"Size: (?P<size>\d+) Layout: (?P<layout>\S+) "
    r"Dims: (?P<dims>\[[^]]*\]).*?c_align: (?P<c_align>\d+) "
    r"w_align: (?P<w_align>\d+) bitdepth: (?P<bitdepth>\d+)"
)


def enrich_cfg_tensors(path: Path, case_name: str, record: dict,
                       source: str | None = None) -> dict:
    """Copy manifest tensors and add cfg-only alignment metadata."""
    cfg_tensors = {"input": [], "output": []}
    for line in (path.read_text() if source is None else source).splitlines():
        match = _CFG_TENSOR_PATTERN.match(line)
        if match is None:
            continue
        parsed = match.groupdict()
        direction = "output" if parsed["output"] else "input"
        cfg_tensors[direction].append({
            "layout": parsed["layout"],
            "dims": json.loads(parsed["dims"]),
            "bitdepth": int(parsed["bitdepth"]),
            "size_per_bank": int(parsed["size"]),
            "c_align": int(parsed["c_align"]),
            "w_align": int(parsed["w_align"]),
        })

    enriched = dict(record)
    for direction in ("input", "output"):
        manifest_tensors = record[f"{direction}s"]
        if len(cfg_tensors[direction]) != len(manifest_tensors):
            raise ValueError(
                f"{case_name}: cfg {direction} count {len(cfg_tensors[direction])} "
                f"!= manifest {len(manifest_tensors)}"
            )
        tensors = []
        for index, (manifest, cfg) in enumerate(
                zip(manifest_tensors, cfg_tensors[direction])):
            for field in ("layout", "dims", "bitdepth", "size_per_bank"):
                if manifest[field] != cfg[field]:
                    raise ValueError(
                        f"{case_name}: {direction} {index} {field} mismatch"
                    )
            tensors.append({**manifest, "c_align": cfg["c_align"],
                            "w_align": cfg["w_align"]})
        enriched[f"{direction}s"] = tensors
    return enriched


@contextlib.contextmanager
def quiet_native_stdout(enabled: bool):
    """Silence verbose vendor codec callbacks, including C/C++ stdout."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        yield
        ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(sink)


def _cfg_registry_fingerprint(records: dict[str, dict], cfg_sources: dict[str, str]) -> str:
    """Bind parsed layouts to every manifest tensor field and the cfg snapshot."""
    tensors = {name: {key: record[key] for key in ("inputs", "outputs")}
               for name, record in records.items()}
    payload = json.dumps({"tensors": tensors, "cfg_sources": cfg_sources},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _cfg_manifest_tensor_fingerprint(records: dict[str, dict]) -> str:
    tensors = {name: {key: record[key] for key in ("inputs", "outputs")}
               for name, record in records.items()}
    payload = json.dumps(tensors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class CfgCodecRegistry:
    """Pre-parse all cfg layout identities and minimize vendor activations."""

    def __init__(self, cfg_dir: Path, records: dict[str, dict], npz2bin: object,
                 quiet: bool):
        started = time.perf_counter()
        self.cfg_dir = cfg_dir
        self.npz2bin = npz2bin
        self.quiet = quiet
        self.signatures = {}
        self.representatives = {}
        cfg_sources = {name: (cfg_dir / f"{name}_cfg.txt").read_text() for name in records}
        self.cfg_file_identities = {
            name: _file_identity(cfg_dir / f"{name}_cfg.txt") for name in records
        }
        self.manifest_tensor_fingerprint = _cfg_manifest_tensor_fingerprint(records)
        self.input_fingerprint = _cfg_registry_fingerprint(records, cfg_sources)
        enriched_records = {}
        for name in records:
            path = cfg_dir / f"{name}_cfg.txt"
            enriched_records[name] = enrich_cfg_tensors(path, name, records[name], cfg_sources[name])
        self.descriptors = build_case_descriptors(enriched_records)
        for name in records:
            signature = tuple(
                descriptor.identity()
                for direction in ("input", "output")
                for descriptor in self.descriptors[name][direction]
            )
            self.signatures[name] = signature
            self.representatives.setdefault(signature, name)
        self.preparse_ms = (time.perf_counter() - started) * 1000.0
        self.active_signature = None
        self.activations = 0
        self.activation_ms = 0.0

    def signature(self, name: str) -> tuple[str, ...]:
        return self.signatures[name]

    def activate(self, name: str) -> None:
        signature = self.signatures[name]
        if signature == self.active_signature:
            return
        representative = self.representatives[signature]
        stem = self.cfg_dir / representative
        started = time.perf_counter()
        with quiet_native_stdout(self.quiet):
            self.npz2bin.read_cfg(str(stem))
        self.activation_ms += (time.perf_counter() - started) * 1000.0
        self.activations += 1
        self.active_signature = signature


def get_cached_cfg_registry(cfg_dir: Path, records: dict[str, dict],
                            npz2bin: object, quiet: bool) -> tuple[CfgCodecRegistry, bool]:
    key = str(cfg_dir.resolve())
    cached = _CFG_REGISTRY_CACHE.get(key)
    if cached is not None:
        if set(cached.signatures) != set(records):
            raise RuntimeError("cfg registry case set changed inside resident process")
        if cached.manifest_tensor_fingerprint != _cfg_manifest_tensor_fingerprint(records):
            raise RuntimeError(
                "cfg registry tensor metadata or cfg contents changed inside resident process"
            )
        cfg_identities = {
            name: _file_identity(cfg_dir / f"{name}_cfg.txt") for name in records
        }
        if cached.cfg_file_identities == cfg_identities:
            return cached, True
        cfg_sources = {name: (cfg_dir / f"{name}_cfg.txt").read_text() for name in records}
        if cached.input_fingerprint != _cfg_registry_fingerprint(records, cfg_sources):
            raise RuntimeError("cfg registry tensor metadata or cfg contents changed inside resident process")
        cached.cfg_file_identities = cfg_identities
        return cached, True
    registry = CfgCodecRegistry(cfg_dir, records, npz2bin, quiet)
    _CFG_REGISTRY_CACHE[key] = registry
    return registry, False


def active_codec_cases(contract: dict, plan: dict, args: argparse.Namespace) -> set[str]:
    """Collect boundaries reachable by the existing full/capture/resume modes."""
    names = set()
    if ("frontend" in contract and args.encoder_captures is None
            and args.encoder_resume is None):
        names.update(plan["frontend"]["projection_kernels"])
    if args.encoder_captures is None:
        start = args.encoder_start_layer if args.encoder_resume is not None else 0
        for block in contract["encoder"][start:]:
            fused_layers = getattr(args, "fused_qkv_attention_layer_set", set())
            if (getattr(args, "fused_qkv_attention", False)
                    and int(block["layer"]) in fused_layers):
                fused = block.get("qkv_attention_fused")
                if fused is None:
                    raise ValueError(
                        f"layer {block['layer']}: fused QKV/attention contract is missing"
                    )
                names.add(fused["kernel"])
            attention6_layers = getattr(
                args, "fused_attention6_layer_set", set()
            )
            if (getattr(args, "fused_attention6", False)
                    and int(block["layer"]) in attention6_layers):
                fused = block.get("attention_fused")
                if fused is None:
                    raise ValueError(
                        f"layer {block['layer']}: fused six-head attention "
                        "contract is missing"
                    )
                names.add(fused["kernel"])
            for norm in ("host_norm1", "host_norm2"):
                if "npu_core" in block[norm]:
                    names.add(block[norm]["npu_core"])
            names.add(block["qkv"]["kernel"])
            names.update(head["kernel"] for head in block["attention"]["heads"])
            names.add(block["post_attention"]["kernel"])
            names.update(block["mlp"]["fc1_kernels"])
            names.add(block["mlp"]["fc2_kernel"])
    lowered = {step["source_node"]: step["kernel"] for step in contract["decoder"]
               if step["backend"] == "npu_layernorm"}
    contracted = {step["source_node"]: step for step in contract["decoder"]
                  if step["backend"] == "npu"}
    for step in plan["decoder_steps"]:
        if step["backend"] == "host":
            if step["name"] in lowered:
                names.add(lowered[step["name"]])
        else:
            expected = contracted.get(step["name"])
            kernels = (expected["kernels"] if expected is not None
                       and expected.get("fused_decoder_stem")
                       else step["kernels"])
            names.update(kernel["name"] for kernel in kernels)
    return names


class RuntimeTensorCodec:
    """Route complete cfg directions to their qualified codec backend."""

    def __init__(self, registry, selection, runtime, create_tensor, export_tensor,
                 preferred_output_key):
        self.registry = registry
        self.selection = selection
        self.runtime = runtime
        self.create_tensor = create_tensor
        self.export_tensor = export_tensor
        self.preferred_output_key = preferred_output_key
        self.reusable_pack_hits = 0
        self.reusable_pack_logical_bytes_saved = 0
        self.reusable_pack_physical_bytes_saved = 0
        self.prepacked_input_calls = 0
        self.prepacked_input_physical_bytes = 0

    class ReusableInput:
        """Explicitly immutable logical input with descriptor-specific packs."""

        def __init__(self, value: np.ndarray):
            self.value = np.ascontiguousarray(value)
            self.packed: dict[str, object] = {}

    class QuantizedInput:
        """Immutable FP32 input lazily quantized into each native descriptor."""

        def __init__(self, value: np.ndarray, scale: float):
            self.value = np.ascontiguousarray(value, dtype=np.float32)
            self.scale = float(scale)
            if not np.isfinite(self.scale) or self.scale <= 0.0:
                raise ValueError("quantized input scale must be finite and positive")
            self.packed: dict[str, object] = {}

    class PrepackedInput:
        """A qualified physical bank pair for one exact input descriptor."""

        def __init__(self, physical, descriptor):
            if not isinstance(physical, tuple) or len(physical) != 2:
                raise ValueError("prepacked input must be an (even, odd) bank pair")
            expected = descriptor.combined_bytes // 2
            banks = tuple(np.ascontiguousarray(bank, dtype=np.uint8)
                          for bank in physical)
            if any(bank.ndim != 1 or bank.size != expected for bank in banks):
                raise ValueError("prepacked bank size does not match descriptor")
            self.physical = banks
            self.descriptor_identity = descriptor.identity()
            self.combined_bytes = descriptor.combined_bytes

    def reusable(self, value: np.ndarray) -> "RuntimeTensorCodec.ReusableInput":
        return self.ReusableInput(value)

    def quantized(self, value: np.ndarray, scale: float) -> "RuntimeTensorCodec.QuantizedInput":
        return self.QuantizedInput(value, scale)

    def prepacked(self, physical, descriptor) -> "RuntimeTensorCodec.PrepackedInput":
        return self.PrepackedInput(physical, descriptor)

    def _native(self, name, descriptor, operation, *arrays):
        try:
            return getattr(self.runtime, f"{operation}_tensor")(*arrays, descriptor)
        except Exception as error:
            raise RuntimeError(
                f"{name}: {descriptor.direction} {descriptor.index} descriptor "
                f"{descriptor.identity()}: native {operation} failed: {error}"
            ) from error

    def _pack_native_input(self, name: str, item,
                           desc: "TensorLayoutDescriptor"):
        if isinstance(item, self.PrepackedInput):
            if item.descriptor_identity != desc.identity():
                raise ValueError(
                    f"{name}: prepacked input descriptor identity mismatch"
                )
            self.prepacked_input_calls += 1
            self.prepacked_input_physical_bytes += item.combined_bytes
            return item.physical
        if isinstance(item, self.QuantizedInput):
            identity = desc.identity()
            if identity in item.packed:
                self.reusable_pack_hits += 1
                self.reusable_pack_logical_bytes_saved += item.value.nbytes
                self.reusable_pack_physical_bytes_saved += desc.combined_bytes
                return item.packed[identity]
            physical = self.runtime.quantize_pack_tensor(
                item.value, desc, item.scale
            )
            item.packed[identity] = physical
            return physical
        if isinstance(item, self.ReusableInput):
            identity = desc.identity()
            if identity in item.packed:
                self.reusable_pack_hits += 1
                self.reusable_pack_logical_bytes_saved += item.value.nbytes
                self.reusable_pack_physical_bytes_saved += desc.combined_bytes
                return item.packed[identity]
            physical = self._native(name, desc, "pack", item.value)
            item.packed[identity] = physical
            return physical
        return self._native(name, desc, "pack", item)

    def pack_inputs(self, name: str, logical_inputs: list) -> list:
        descriptors = self.registry.descriptors[name]["input"]
        if len(logical_inputs) != len(descriptors):
            raise ValueError(f"{name}: logical input count does not match cfg")
        if self.selection.native_for(name, "input"):
            return [self._pack_native_input(name, item, desc)
                    for item, desc in zip(logical_inputs, descriptors)]
        self.registry.activate(name)
        if any(isinstance(value, self.PrepackedInput) for value in logical_inputs):
            raise RuntimeError(f"{name}: prepacked input requires native layout codec")
        if any(isinstance(value, self.QuantizedInput) for value in logical_inputs):
            raise RuntimeError(f"{name}: fused quantize-pack requires native layout codec")
        started = time.perf_counter()
        with quiet_native_stdout(self.registry.quiet):
            logical = [np.ascontiguousarray(
                value.value if isinstance(value, self.ReusableInput) else value
            ) for value in logical_inputs]
            values = self.registry.npz2bin.read_npz_dict(
                self.create_tensor, "input", [{"input": value} for value in logical])
        elapsed = (time.perf_counter() - started) * 1000.0
        packed = [np.ascontiguousarray(value).reshape(-1).view(np.uint8) for value in values]
        expected = [desc.combined_bytes for desc in descriptors]
        actual = [int(value.size) for value in packed]
        if actual != expected:
            raise ValueError(f"{name}: packed byte sizes {actual} != cfg {expected}")
        self.selection.record("vendor", "pack", descriptors, [v.nbytes for v in logical], elapsed)
        return packed

    def pack_resident_inputs(self, name: str, logical_inputs: list) -> list:
        """Pack host inputs while preserving device handles and connections."""
        descriptors = self.registry.descriptors[name]["input"]
        if len(logical_inputs) != len(descriptors):
            raise ValueError(f"{name}: resident input count does not match cfg")
        if not self.selection.native_for(name, "input"):
            raise RuntimeError(f"{name}: resident inputs require native codec")
        result = []
        for item, desc in zip(logical_inputs, descriptors):
            if item is None or isinstance(item, mapped_runtime.DeviceTensorHandle):
                result.append(item)
            else:
                result.append(self._pack_native_input(name, item, desc))
        return result

    def decode_outputs(self, name: str, physical: list) -> list[np.ndarray]:
        descriptors = self.registry.descriptors[name]["output"]
        if len(physical) != len(descriptors):
            raise ValueError(f"{name}: physical output count does not match cfg")
        if self.selection.native_for(name, "output"):
            return [self._native(name, desc, "unpack", *banks)
                    for banks, desc in zip(physical, descriptors)]
        self.registry.activate(name)
        started = time.perf_counter()
        with quiet_native_stdout(self.registry.quiet):
            decoded = self.registry.npz2bin.buffer_to_npz_dict(
                self.export_tensor, "output", physical)
        elapsed = (time.perf_counter() - started) * 1000.0
        values = [np.ascontiguousarray(item[self.preferred_output_key(item)]) for item in decoded]
        if len(values) != len(descriptors):
            raise ValueError(f"{name}: decoded output count does not match cfg")
        self.selection.record("vendor", "unpack", descriptors, [v.nbytes for v in values], elapsed)
        return values

    def reuse_stats(self) -> dict[str, int]:
        return {
            "native_pack_cache_hits": self.reusable_pack_hits,
            "native_pack_cache_logical_bytes_saved":
                self.reusable_pack_logical_bytes_saved,
            "native_pack_cache_physical_bytes_saved":
                self.reusable_pack_physical_bytes_saved,
            "native_prepacked_input_calls": self.prepacked_input_calls,
            "native_prepacked_input_physical_bytes":
                self.prepacked_input_physical_bytes,
        }


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(value, dtype=np.float32) / scale),
                   -128, 127).astype(np.int8)


def attention_head_inputs(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    head: dict,
    quantizer=quantize,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice one head and bridge either INT8 or BF16 QKV into INT8 attention.

    Legacy QKV kernels requantize all six heads with one scale and already
    return INT8.  Accuracy-calibrated kernels return BF16 so each 64-channel
    head can use its own static Q/K/V scale before the INT8 QK/AV kernels.
    Mixed Q/K/V dtypes are rejected because they do not describe either ABI.
    """
    arrays = (np.asarray(q), np.asarray(k), np.asarray(v))
    integer = tuple(value.dtype == np.int8 for value in arrays)
    if any(integer) and not all(integer):
        raise ValueError("Q/K/V outputs must be uniformly INT8 or BF16-decoded float")
    begin = int(head["head"]) * 64
    end = begin + 64
    sliced = tuple(value[0, 0, :, begin:end] for value in arrays)
    if all(integer):
        return sliced
    scales = head.get("scales_bf16", {})
    missing = [name for name in ("q", "k", "v")
               if not float(scales.get(name, 0.0)) > 0.0]
    if missing:
        raise ValueError(
            f"head {head['head']}: BF16 QKV requires positive per-head scales: {missing}"
        )
    return tuple(
        np.ascontiguousarray(
            quantizer(np.ascontiguousarray(value, dtype=np.float32),
                      float(scales[name])),
            dtype=np.int8,
        )
        for name, value in zip(("q", "k", "v"), sliced)
    )


def validate_attention_amplitude_contract(contract: dict) -> None:
    """Reject hidden V/AV amplitude compensation in encoder attention."""
    for layer in contract.get("encoder", []):
        layer_index = int(layer["layer"])
        for head in layer["attention"]["heads"]:
            scales = head.get("scales_bf16", {})
            if not scales:
                continue
            head_index = int(head["head"])
            v_scale = float(scales["v"])
            av_scale = float(scales.get("av_v", v_scale))
            gain = float(scales.get("av_output_gain", 1.0))
            if not np.isclose(gain, 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"encoder layer {layer_index} head {head_index}: "
                    f"av_output_gain must be 1, got {gain}"
                )
            if not np.isclose(av_scale, v_scale, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"encoder layer {layer_index} head {head_index}: "
                    f"AV scale {av_scale} must equal V scale {v_scale}"
                )


def calibration_stats(value: np.ndarray) -> dict:
    array = np.asarray(value, dtype=np.float32)
    absolute = np.abs(array).reshape(-1)
    return {
        "shape": list(array.shape),
        "elements": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "std": float(np.std(array, dtype=np.float64)),
        "max_abs": float(np.max(absolute)),
        "abs_p999": float(np.percentile(absolute, 99.9)),
        "abs_p9999": float(np.percentile(absolute, 99.99)),
    }


def layer_norm(value: np.ndarray, scale: np.ndarray, bias: np.ndarray,
               axis: int, epsilon: float) -> np.ndarray:
    axes = tuple(range(axis % value.ndim, value.ndim))
    mean = np.mean(value, axis=axes, keepdims=True, dtype=np.float32)
    variance = np.mean((value - mean) ** 2, axis=axes,
                       keepdims=True, dtype=np.float32)
    return ((value - mean) / np.sqrt(variance + epsilon) * scale + bias).astype(np.float32)


def gelu(value: np.ndarray) -> np.ndarray:
    # Vectorized erf approximation; maximum absolute erf error is ~1.5e-7.
    x = np.asarray(value, dtype=np.float32) / np.float32(np.sqrt(2.0))
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    polynomial = (((((1.061405429 * t - 1.453152027) * t)
                    + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    erf = sign * (1.0 - polynomial * np.exp(-(a * a)))
    return (0.5 * value * (1.0 + erf)).astype(np.float32)


def resize_align_corners(value: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    output_shape = tuple(int(x) for x in np.asarray(sizes).reshape(-1))
    if value.ndim != 4 or len(output_shape) != 4:
        raise ValueError(f"only NCHW Resize is supported: {value.shape}, {output_shape}")
    out_h, out_w = output_shape[2:]
    in_h, in_w = value.shape[2:]
    ys = np.linspace(0.0, in_h - 1, out_h, dtype=np.float32) if out_h > 1 else np.zeros(1)
    xs = np.linspace(0.0, in_w - 1, out_w, dtype=np.float32) if out_w > 1 else np.zeros(1)
    y0 = np.floor(ys).astype(np.int64); y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(xs).astype(np.int64); x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0).reshape(1, 1, out_h, 1)
    wx = (xs - x0).reshape(1, 1, 1, out_w)
    vertical = value[:, :, y0, :] * (1.0 - wy) + value[:, :, y1, :] * wy
    return (vertical[:, :, :, x0] * (1.0 - wx)
            + vertical[:, :, :, x1] * wx).astype(np.float32)


def depth_to_space(value: np.ndarray, block: int, mode: str) -> np.ndarray:
    n, channels, height, width = value.shape
    output_channels = channels // (block * block)
    if mode == "CRD":
        shaped = value.reshape(n, output_channels, block, block, height, width)
        return shaped.transpose(0, 1, 4, 2, 5, 3).reshape(
            n, output_channels, height * block, width * block)
    shaped = value.reshape(n, block, block, output_channels, height, width)
    return shaped.transpose(0, 3, 4, 1, 5, 2).reshape(
        n, output_channels, height * block, width * block)


def execute_host(step: dict, env: dict[str, np.ndarray],
                 timing: dict[str, dict] | None = None,
                 executor=None) -> None:
    started = time.perf_counter()
    values = [env[name] if name else None for name in step["inputs"]]
    attrs = step["attrs"]
    op = step["op_type"]
    if op == "Constant":
        result = env[attrs["value"]["param"]]
    elif op == "LayerNormalization":
        result = layer_norm(values[0], values[1], values[2],
                            int(attrs.get("axis", -1)), float(attrs.get("epsilon", 1e-5)))
    elif op == "Slice":
        starts = np.asarray(values[1]).reshape(-1).astype(np.int64)
        ends = np.asarray(values[2]).reshape(-1).astype(np.int64)
        axes = (np.asarray(values[3]).reshape(-1).astype(np.int64)
                if len(values) > 3 and values[3] is not None else np.arange(starts.size))
        steps = (np.asarray(values[4]).reshape(-1).astype(np.int64)
                 if len(values) > 4 and values[4] is not None else np.ones(starts.size, np.int64))
        slices = [slice(None)] * values[0].ndim
        for start, end, axis, stride in zip(starts, ends, axes, steps):
            slices[int(axis)] = slice(int(start), int(end), int(stride))
        result = values[0][tuple(slices)]
    elif op == "Transpose":
        result = np.transpose(values[0], attrs.get("perm"))
    elif op == "Reshape":
        shape = np.asarray(values[1]).reshape(-1).astype(np.int64).tolist()
        if not int(attrs.get("allowzero", 0)):
            shape = [values[0].shape[i] if x == 0 else x for i, x in enumerate(shape)]
        result = np.reshape(values[0], shape)
    elif op == "DepthToSpace":
        result = depth_to_space(values[0], int(attrs["blocksize"]), attrs.get("mode", "DCR"))
    elif op == "Relu":
        result = np.maximum(values[0], 0).astype(np.float32)
    elif op == "Add":
        native = executor is not None and all(
            value.dtype == np.float32 and value.flags.c_contiguous
            for value in values[:2]
        )
        result = (executor.add(values[0], values[1]) if native
                  else values[0] + values[1])
    elif op == "Shape":
        result = np.asarray(values[0].shape, dtype=np.int64)
    elif op == "Concat":
        native = (executor is not None and bool(values)
                  and values[0].dtype in (np.dtype(np.float32), np.dtype(np.int8))
                  and all(value.dtype == values[0].dtype
                          and value.flags.c_contiguous for value in values))
        result = (executor.concatenate(values, axis=int(attrs["axis"]))
                  if native
                  else np.concatenate(values, axis=int(attrs["axis"])))
    elif op == "Resize":
        if attrs.get("mode", "nearest") != "linear" or attrs.get(
                "coordinate_transformation_mode", "half_pixel") != "align_corners":
            raise ValueError(f"unsupported Resize attributes: {attrs}")
        sizes = values[3] if len(values) > 3 and values[3] is not None else np.rint(
            np.asarray(values[0].shape) * np.asarray(values[2])).astype(np.int64)
        native = (executor is not None and values[0].dtype == np.float32
                  and values[0].flags.c_contiguous)
        result = (executor.resize_align_corners(values[0], sizes) if native
                  else resize_align_corners(values[0], sizes))
    elif op == "Squeeze":
        axes = (tuple(int(x) for x in np.asarray(values[1]).reshape(-1))
                if len(values) > 1 else None)
        result = np.squeeze(values[0], axis=axes)
    else:
        raise NotImplementedError(f"host op {op}: {step['name']}")
    env[step["outputs"][0]] = np.ascontiguousarray(result)
    if timing is not None:
        aggregate = timing.setdefault(op, {"calls": 0, "ms": 0.0})
        aggregate["calls"] += 1
        aggregate["ms"] += (time.perf_counter() - started) * 1000.0


def main() -> int:
    global _NPZ_YAML_PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--host-params", type=Path, required=True)
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--encoder-captures", type=Path,
                        help="skip the encoder and load capture_l02/l05/l08/l11")
    parser.add_argument("--encoder-resume", type=Path,
                        help="resume the encoder from block_lXX in a prior board trace")
    parser.add_argument("--encoder-start-layer", type=int,
                        help="first encoder layer to execute with --encoder-resume")
    parser.add_argument(
        "--replacement-trace", type=Path,
        help="FP32 NPZ used by one capture-and-replace intervention",
    )
    parser.add_argument(
        "--replace-encoder-block", type=int,
        help="replace block_lXX after this encoder block with its FP32 capture",
    )
    parser.add_argument(
        "--replace-encoder-internal", type=parse_encoder_internal_target,
        metavar="LAYER:STAGE",
        help=("replace one FP32 encoder boundary; stages: "
              + ", ".join(ENCODER_INTERNAL_STAGES)),
    )
    parser.add_argument(
        "--replace-encoder-attention-head",
        type=parse_encoder_attention_head_target, metavar="LAYER:HEAD",
        help="replace one 64-channel raw-attention head from the FP32 trace",
    )
    parser.add_argument(
        "--replace-decoder-conv", type=int,
        help="replace decoder_conv_XX after this NPU Conv with its FP32 capture",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-only", action="store_true",
                        help="save only the final depth tensor, not intermediate traces")
    parser.add_argument(
        "--trace-attention-layers",
        help=("comma-separated encoder layers whose Q/K/V, attention, and block "
              "outputs are retained even with --depth-only"),
    )
    parser.add_argument(
        "--trace-encoder-internal-layers",
        help=("comma-separated encoder layers whose Norm, post-attention, MLP, "
              "and block boundaries are retained even with --depth-only"),
    )
    parser.add_argument(
        "--trace-decoder-convs",
        help=("comma-separated decoder Conv indices whose inputs and outputs "
              "are retained even with --depth-only"),
    )
    parser.add_argument(
        "--collect-calibration", action="store_true",
        help="compute expensive percentile statistics for calibration workflows",
    )
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--dma-runtime", choices=("legacy", "cpp_mapped"), default="cpp_mapped",
        help="use the mapped-BAR C++ transport or the legacy per-transfer bridge",
    )
    parser.add_argument(
        "--layout-codec", choices=("vendor", "auto", "native"), default="vendor",
        help="physical tensor codec; native requires a complete qualification report",
    )
    parser.add_argument("--layout-codec-report", type=Path,
                        help="single qualification JSON matching the manifest and active descriptors")
    parser.add_argument(
        "--fpga-dma-batch", type=Path,
        help="fpgaDmaBatch extension file or directory (required if not importable)",
    )
    parser.add_argument(
        "--host-executor", choices=("python", "auto", "cpp"), default="python",
        help="host graph backend; cpp requires a matching qualification report",
    )
    parser.add_argument(
        "--host-executor-report", type=Path,
        help="qualification report for --host-executor auto/cpp",
    )
    parser.add_argument(
        "--cpp-persistent-dma", action="store_true",
        help="reuse XDMA descriptors; default safe mode reopens each DMA segment",
    )
    parser.add_argument(
        "--attention-launch-group", type=int, default=3,
        help="maximum independent attention calls per C++ submission",
    )
    parser.add_argument(
        "--decoder-launch-group", type=int, default=32,
        help="maximum independent decoder calls per C++ submission",
    )
    parser.add_argument(
        "--encoder-fc1-launch-group", type=int, default=1,
        help="maximum independent encoder FC1 channel shards per C++ submission",
    )
    parser.add_argument(
        "--frontend-launch-group", type=int, default=1,
        help="maximum independent patch-projection shards per C++ submission",
    )
    parser.add_argument(
        "--encoder-fc-frame-graph", action="store_true",
        help=("execute each sharded FC1, physical GELU/quantize bridge, and "
              "FC2 as one C++ frame-graph transaction"),
    )
    parser.add_argument(
        "--verbose-vendor-codec", action="store_true",
        help="retain npz2bin's very verbose cfg/pack/unpack stdout",
    )
    parser.add_argument(
        "--attention-resident-kv", action="store_true",
        help="upload K/V once per head and retain them at the shared attention IO addresses",
    )
    parser.add_argument(
        "--attention-post-frame-graph", action="store_true",
        help=("execute all six 280-token attention heads, the physical output "
              "bridge, and post projection in one C++ frame-graph call"),
    )
    parser.add_argument(
        "--fused-qkv-attention", action="store_true",
        help=("replace each standalone QKV plus six attention launches with "
              "one compiler-fused program inside the attention/post frame graph"),
    )
    parser.add_argument(
        "--fused-qkv-attention-layers",
        help=("comma-separated encoder layers to fuse; defaults to all 12 "
              "when --fused-qkv-attention is enabled"),
    )
    parser.add_argument(
        "--fused-attention6", action="store_true",
        help=("preserve the BF16 QKV to host-A8 boundary and replace six "
              "attention programs with one compiler-fused program"),
    )
    parser.add_argument(
        "--fused-attention6-layers",
        help=("comma-separated encoder layers to fuse; defaults to all 12 "
              "when --fused-attention6 is enabled"),
    )
    parser.add_argument(
        "--qkv-attention6-frame-graph", action="store_true",
        help=("execute QKV, an exact physical BF16-to-head-A8 bridge, "
              "attention6, and post in one C++ frame-graph call"),
    )
    parser.add_argument(
        "--encoder-resident-intermediates", action="store_true",
        help="retain qualified encoder residual/post tensors in relocated shared FM",
    )
    parser.add_argument(
        "--decoder-resident-captures", action="store_true",
        help="retain block 2/5/8 captures and batch four decoder SPU LayerNorms",
    )
    parser.add_argument(
        "--decoder-fused-stems", action="store_true",
        help="run resident LayerNorm/layout/project-Conv stems from capture handles",
    )
    parser.add_argument(
        "--decoder-native-boundary", action="store_true",
        help=("run the four capture LayerNorm/layout/project boundaries as one "
              "C++ physical frame-graph call using the qualified Conv BINs"),
    )
    parser.add_argument(
        "--decoder-quantize-pack", action="store_true",
        help="fuse decoder FP32 quantization and native physical input packing",
    )
    parser.add_argument(
        "--encoder-quantize-pack", action="store_true",
        help=("fuse encoder/frontend FP32 quantization with native physical "
              "input packing on production paths"),
    )
    parser.add_argument(
        "--cpp-mixed-signature-groups", action="store_true",
        help=("allow one mapped C++ transaction to contain independent calls "
              "with different physical tensor extents"),
    )
    args = parser.parse_args()
    try:
        fused_qkv_attention_layer_set = (
            set(range(12))
            if args.fused_qkv_attention
            and args.fused_qkv_attention_layers is None
            else {
                int(item)
                for item in (args.fused_qkv_attention_layers or "").split(",")
                if item
            }
        )
    except ValueError:
        parser.error(
            "--fused-qkv-attention-layers must be comma-separated integers"
        )
    if not fused_qkv_attention_layer_set <= set(range(12)):
        parser.error("--fused-qkv-attention-layers must be within [0, 11]")
    if (args.fused_qkv_attention_layers is not None
            and not args.fused_qkv_attention):
        parser.error(
            "--fused-qkv-attention-layers requires --fused-qkv-attention"
        )
    args.fused_qkv_attention_layer_set = fused_qkv_attention_layer_set
    try:
        fused_attention6_layer_set = (
            set(range(12))
            if args.fused_attention6
            and args.fused_attention6_layers is None
            else {
                int(item)
                for item in (args.fused_attention6_layers or "").split(",")
                if item
            }
        )
    except ValueError:
        parser.error(
            "--fused-attention6-layers must be comma-separated integers"
        )
    if not fused_attention6_layer_set <= set(range(12)):
        parser.error("--fused-attention6-layers must be within [0, 11]")
    if (args.fused_attention6_layers is not None
            and not args.fused_attention6):
        parser.error(
            "--fused-attention6-layers requires --fused-attention6"
        )
    args.fused_attention6_layer_set = fused_attention6_layer_set
    try:
        attention_trace_layers = (
            set() if args.trace_attention_layers is None else
            {int(item) for item in args.trace_attention_layers.split(",") if item}
        )
    except ValueError:
        parser.error("--trace-attention-layers must be comma-separated integers")
    if not attention_trace_layers <= set(range(12)):
        parser.error("--trace-attention-layers must be within [0, 11]")
    try:
        encoder_internal_trace_layers = (
            set() if args.trace_encoder_internal_layers is None else
            {int(item) for item in args.trace_encoder_internal_layers.split(",")
             if item}
        )
    except ValueError:
        parser.error(
            "--trace-encoder-internal-layers must be comma-separated integers"
        )
    if not encoder_internal_trace_layers <= set(range(12)):
        parser.error("--trace-encoder-internal-layers must be within [0, 11]")
    # Internal boundary tracing does not require materialized Q/K/V or raw
    # attention tensors.  Keep it separate from attention tracing so a fused
    # QKV/attention/post frame graph can still expose post, Norm2, MLP and
    # complete-block diagnostics without silently falling back to the legacy
    # attention path.
    try:
        decoder_trace_convs = (
            set() if args.trace_decoder_convs is None else
            {int(item) for item in args.trace_decoder_convs.split(",") if item}
        )
    except ValueError:
        parser.error("--trace-decoder-convs must be comma-separated integers")
    if not decoder_trace_convs <= set(range(32)):
        parser.error("--trace-decoder-convs must be within [0, 31]")
    if (args.encoder_resume is None) != (args.encoder_start_layer is None):
        parser.error("--encoder-resume and --encoder-start-layer must be used together")
    if args.encoder_captures is not None and args.encoder_resume is not None:
        parser.error("--encoder-captures and --encoder-resume are mutually exclusive")
    if args.encoder_start_layer is not None and not 1 <= args.encoder_start_layer <= 11:
        parser.error("--encoder-start-layer must be in [1, 11]")
    replacement_targets = sum(value is not None for value in (
        args.replace_encoder_block, args.replace_encoder_internal,
        args.replace_encoder_attention_head,
        args.replace_decoder_conv,
    ))
    if (args.replacement_trace is None) != (replacement_targets == 0):
        parser.error(
            "--replacement-trace requires exactly one replacement target and vice versa"
        )
    if replacement_targets > 1:
        parser.error("only one capture-and-replace target is allowed per run")
    if (args.replace_encoder_block is not None
            and not 0 <= args.replace_encoder_block <= 11):
        parser.error("--replace-encoder-block must be within [0, 11]")
    if (args.replace_decoder_conv is not None
            and not 0 <= args.replace_decoder_conv <= 31):
        parser.error("--replace-decoder-conv must be within [0, 31]")
    if (args.replace_encoder_internal is not None
            and args.encoder_resident_intermediates):
        parser.error(
            "--replace-encoder-internal is incompatible with resident encoder "
            "chaining because the replacement invalidates its device handle"
        )
    if (args.replace_encoder_attention_head is not None
            and args.encoder_resident_intermediates):
        parser.error(
            "--replace-encoder-attention-head is incompatible with resident "
            "encoder chaining"
        )
    if (args.attention_launch_group < 1 or args.decoder_launch_group < 1
            or args.encoder_fc1_launch_group < 1
            or args.frontend_launch_group < 1):
        parser.error("launch group sizes must be positive")
    if args.layout_codec == "native" and args.dma_runtime != "cpp_mapped":
        parser.error("--layout-codec native requires --dma-runtime cpp_mapped")
    if args.host_executor != "python" and args.dma_runtime != "cpp_mapped":
        parser.error("--host-executor auto/cpp requires --dma-runtime cpp_mapped")
    if (args.encoder_resident_intermediates
            and (args.dma_runtime != "cpp_mapped" or args.layout_codec != "native")):
        parser.error(
            "--encoder-resident-intermediates requires cpp_mapped DMA and native codec"
        )
    if args.attention_post_frame_graph and (
            args.dma_runtime != "cpp_mapped"
            or args.layout_codec != "native"
            or args.host_executor != "cpp"
            or (args.encoder_resident_intermediates
                and not args.fused_attention6)):
        parser.error(
            "--attention-post-frame-graph requires native codec, cpp_mapped DMA, "
            "the C++ host executor; resident encoder residuals require "
            "--fused-attention6"
        )
    if args.fused_qkv_attention and (
            not args.attention_post_frame_graph
            or not args.depth_only
            or args.collect_calibration
            or attention_trace_layers
            or args.replace_encoder_internal is not None
            or args.replace_encoder_attention_head is not None):
        parser.error(
            "--fused-qkv-attention requires the depth-only attention/post "
            "frame graph without attention diagnostics or replacements"
        )
    if args.fused_attention6 and (
            not args.attention_post_frame_graph
            or not args.depth_only
            or args.collect_calibration
            or attention_trace_layers
            or args.replace_encoder_internal is not None
            or args.replace_encoder_attention_head is not None):
        parser.error(
            "--fused-attention6 requires the depth-only attention/post frame "
            "graph without attention diagnostics or replacements"
        )
    # The two switches may be combined for a selective rollout: layers in
    # fused_qkv_attention_layer_set execute the QKV+attention program, while
    # every other layer keeps the qualified r168 attention6 path.  A fused-QKV
    # layer clears attention_heads before attention6 scheduling, so the two
    # programs cannot execute for the same block.
    if args.qkv_attention6_frame_graph and not args.fused_attention6:
        parser.error("--qkv-attention6-frame-graph requires --fused-attention6")
    if args.decoder_resident_captures:
        if not args.encoder_resident_intermediates:
            parser.error(
                "--decoder-resident-captures requires --encoder-resident-intermediates"
            )
        if args.encoder_captures is not None or args.encoder_resume is not None:
            parser.error(
                "--decoder-resident-captures requires a full encoder execution"
            )
    if args.decoder_native_boundary:
        if (args.layout_codec != "native"
                or args.dma_runtime != "cpp_mapped"
                or args.host_executor != "cpp"):
            parser.error(
                "--decoder-native-boundary requires native codec, cpp_mapped "
                "DMA, and the C++ host executor"
            )
        if args.decoder_fused_stems:
            parser.error(
                "--decoder-native-boundary and --decoder-fused-stems are mutually exclusive"
            )
    if args.decoder_quantize_pack and (
            args.layout_codec != "native" or args.dma_runtime != "cpp_mapped"):
        parser.error(
            "--decoder-quantize-pack requires native codec and cpp_mapped DMA"
        )
    if args.encoder_quantize_pack and (
            args.layout_codec != "native" or args.dma_runtime != "cpp_mapped"):
        parser.error(
            "--encoder-quantize-pack requires native codec and cpp_mapped DMA"
        )
    if args.encoder_fc_frame_graph and (
            args.layout_codec != "native"
            or args.dma_runtime != "cpp_mapped"
            or args.host_executor != "cpp"):
        parser.error(
            "--encoder-fc-frame-graph requires native codec, cpp_mapped DMA, "
            "and the C++ host executor"
        )
    process_started = time.perf_counter()
    host_profiler = HostProfiler()
    replacement_archive = (
        np.load(args.replacement_trace, allow_pickle=False)
        if args.replacement_trace is not None else None
    )

    def profiled_quantize(
        category: str, value: np.ndarray, scale: float
    ) -> np.ndarray:
        with host_profiler.measure(
            category, elements=int(value.size), nbytes=int(value.nbytes)
        ):
            return host_executor.quantize(value, scale)

    case_dir = args.case_dir.resolve(); runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(case_dir)); sys.path.insert(1, str(runtime_dir))
    import npz2bin  # type: ignore
    from npz_util import createBF16TensorFromDict  # type: ignore
    from run_u250_resident_compiled_case import (  # type: ignore
        C2H_DEVICES, DDR_BASES, EventWaiter, H2C_DEVICES,
        clear_interrupt, configure_npu, export_npz_allow_nonfinite,
        merge_2ddr, preferred_output_key, reg_write, sha256_array,
        split_2ddr, tensor_metrics,
    )
    yaml_paths = (
        str(runtime_dir / "arch_16_mono.yaml"),
        str(runtime_dir / "arch_256_mono.yaml"),
    )
    yaml_reused = _NPZ_YAML_PATHS is not None
    if _NPZ_YAML_PATHS is None:
        with quiet_native_stdout(not args.verbose_vendor_codec):
            npz2bin.read_yaml(list(yaml_paths))
        _NPZ_YAML_PATHS = yaml_paths
    elif _NPZ_YAML_PATHS != yaml_paths:
        raise RuntimeError(
            "resident process cannot switch npz2bin architecture YAML files"
        )
    manifest, manifest_bytes, manifest_reused = load_json_cached(args.manifest)
    contract, _, contract_reused = load_json_cached(args.contract)
    validate_attention_amplitude_contract(contract)
    plan, _, host_plan_reused = load_json_cached(args.host_plan)
    contract_decoder = {
        step["source_node"]: step for step in contract["decoder"]
        if step["backend"] == "npu"
    }
    contract_decoder_layernorm = {
        step["source_node"]: step for step in contract["decoder"]
        if step["backend"] == "npu_layernorm"
    }
    plan_decoder = {
        step["name"]: step for step in plan["decoder_steps"]
        if step["backend"] == "npu"
    }
    if contract_decoder.keys() != plan_decoder.keys():
        raise ValueError("host plan and runtime contract decoder nodes differ")
    fused_contract_nodes = {
        name for name, step in contract_decoder.items()
        if step.get("fused_decoder_stem")
    }
    if bool(fused_contract_nodes) != bool(args.decoder_fused_stems):
        raise ValueError(
            "decoder fused-stem flag and runtime contract must be enabled together"
        )
    valid_fused_nodes = {
        f"/depth_head/projects.{index}/Conv" for index in range(4)
    }
    if not fused_contract_nodes <= valid_fused_nodes:
        raise ValueError("runtime contract contains an unknown decoder stem")
    fused_capture_nodes = {
        name for name in fused_contract_nodes
        if contract_decoder[name].get("stem_input", "capture") == "capture"
    }
    if fused_capture_nodes and not args.decoder_resident_captures:
        raise ValueError("capture-input decoder stems require resident captures")
    for name, step in plan_decoder.items():
        expected = contract_decoder[name]
        if expected.get("fused_decoder_stem"):
            continue
        expected_scale = float(expected["input_quantization"]["scale"])
        if not np.isclose(float(step["input_scale"]), expected_scale,
                          rtol=0.0, atol=1e-12):
            raise ValueError(
                f"decoder input scale mismatch for {name}: "
                f"plan={step['input_scale']} contract={expected_scale}"
            )
        if ([item["name"] for item in step["kernels"]]
                != [item["name"] for item in expected["kernels"]]):
            raise ValueError(f"decoder kernel mismatch for {name}")
    records = {item["name"]: item for item in manifest["cases"]}
    cfg_registry, cfg_registry_reused = get_cached_cfg_registry(
        args.cfg_dir, records, npz2bin, quiet=not args.verbose_vendor_codec
    )
    active_cases = active_codec_cases(contract, plan, args)
    codec_selection = LayoutCodecSelection(
        args.layout_codec, hashlib.sha256(manifest_bytes).hexdigest(), args.layout_codec_report,
        {name: cfg_registry.descriptors[name] for name in sorted(active_cases)},
    )
    if args.dma_runtime == "legacy":
        codec_selection.prepare(None)
    host_extension = None
    host_extension_path = None
    host_extension_sha256 = None
    if args.host_executor != "python":
        # Qualification is deliberately completed before get_cached_cpp_runtime()
        # can construct DmaBatch and open any XDMA device.
        host_extension = mapped_runtime.load_fpga_dma_batch(args.fpga_dma_batch)
        host_extension_path = Path(host_extension.__file__).resolve()
        host_extension_sha256 = getattr(
            host_extension, "_u250_extension_sha256", None
        )
    host_selection = HostExecutorSelection(
        args.host_executor, args.host_executor_report,
        host_extension_path, host_extension_sha256,
    )
    host_executor = host_selection.create(host_extension)
    cfg_activations_at_start = cfg_registry.activations
    cfg_activation_ms_at_start = cfg_registry.activation_ms
    env, host_params_reused = load_host_params_cached(args.host_params)
    loaded_input = np.load(args.input, allow_pickle=False)
    if isinstance(loaded_input, np.ndarray):
        x = np.ascontiguousarray(loaded_input, dtype=np.float32)
    else:
        with loaded_input as archive:
            x = np.ascontiguousarray(archive["input"], dtype=np.float32)

    bank_path = case_dir / manifest["bank_file"]
    linked, bank_image_reused = load_bank_image_cached(bank_path)
    started = time.perf_counter()
    cpp_runtime = None
    cpp_runtime_reused = False
    resident_bank_reused = False
    if args.dma_runtime == "cpp_mapped":
        cpp_runtime, cpp_runtime_reused = get_cached_cpp_runtime(
            str(args.manifest.resolve()), manifest, args.fpga_dma_batch,
            safe_dma=not args.cpp_persistent_dma,
            codec_selection=codec_selection,
        )
        load_ms, resident_bank_reused = cpp_runtime.ensure_bank(
            linked, str(manifest["bank_sha256"])
        )
        cpp_runtime.reset_frame_stats()
        waiter = None
    else:
        import fpgaDma  # type: ignore
        for bank, half in enumerate(split_2ddr(linked)):
            fpgaDma.np2card(H2C_DEVICES[bank], DDR_BASES[bank], half.size, half)
        reg_write(fpgaDma, 0x2C, 1)
        load_ms = (time.perf_counter() - started) * 1000.0
        waiter = EventWaiter(); waiter.drain()
    timings = []
    decoder_checkpoints = {}
    hybrid_calibration = {}
    decoder_calibration = {}
    frontend_captures = {}
    h2c_skipped_bytes = 0
    submission_groups = []
    decoder_host_ops = {}
    light_attention_trace = {}
    light_encoder_internal_trace = {}

    tensor_codec = RuntimeTensorCodec(
        cfg_registry, codec_selection, cpp_runtime, createBF16TensorFromDict,
        export_npz_allow_nonfinite, preferred_output_key,
    )
    pack_inputs = tensor_codec.pack_inputs
    decode_outputs = tensor_codec.decode_outputs

    def run_kernel_group(
        names: list[str], logical_calls: list[list],
        upload_masks: list[list[bool] | None] | None = None,
        decode_outputs_flag: bool = True,
    ) -> list[list]:
        """Run one codec-compatible group and preserve per-call outputs."""
        nonlocal h2c_skipped_bytes
        if len(names) != len(logical_calls) or not names:
            raise ValueError("kernel group names/inputs are not aligned")
        signatures = {cfg_registry.signature(name) for name in names}
        if len(signatures) != 1 and (
                cpp_runtime is None or not args.cpp_mixed_signature_groups):
            raise ValueError("one kernel group must use one physical codec signature")
        if upload_masks is None:
            upload_masks = [None] * len(names)
        masks = []
        packed_calls = []
        for name, logical, mask in zip(names, logical_calls, upload_masks):
            selected = [True] * len(logical) if mask is None else list(mask)
            if len(selected) != len(logical):
                raise ValueError(f"{name}: upload mask does not match input count")
            packed_calls.append(pack_inputs(name, logical))
            masks.append(selected)

        if cpp_runtime is not None:
            # A grouped launch uses disjoint FM slots.  K/V therefore have to
            # be copied into each slot; single-call groups can retain them.
            effective_masks = (masks if len(names) == 1 else
                               [[True] * len(values) for values in logical_calls])
            physical_calls, group = cpp_runtime.run_group(
                [records[name] for name in names], packed_calls,
                effective_masks, args.timeout_ms,
            )
            h2c_skipped_bytes += int(group["h2c_skipped_bytes"])
            submission_groups.append({
                "kind": "cpp_mapped", "kernels": list(names),
                **{key: value for key, value in group.items() if key != "npu_ms"},
            })
            results = ([decode_outputs(name, physical)
                        for name, physical in zip(names, physical_calls)]
                       if decode_outputs_flag else physical_calls)
            for index, (name, npu_ms) in enumerate(zip(names, group["npu_ms"])):
                timings.append({
                    "kernel": name, "event": 0,
                    "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                    "npu_ms": float(npu_ms),
                    "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                    "submission_group_size": len(names),
                })
            return results

        results = []
        for name, packed, mask in zip(names, packed_calls, masks):
            record = records[name]
            configure_npu(fpgaDma, record)
            reg_write(fpgaDma, 0x00, 0x3F)
            waiter.drain()
            reg_write(fpgaDma, 0x34, 1)
            h2c_start = time.perf_counter()
            for tensor, combined, upload in zip(record["inputs"], packed, mask):
                if not upload:
                    h2c_skipped_bytes += int(combined.size)
                    continue
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
            c2h_start = time.perf_counter()
            physical = []
            for tensor in record["outputs"]:
                combined_size = int(tensor["size_per_bank"])
                if combined_size % 2:
                    raise ValueError(f"{name}: odd combined output byte count")
                halves = []
                for bank in range(2):
                    address = ((record["base_addresses"][4] + tensor["address"])
                               * 0x80 + DDR_BASES[bank])
                    halves.append(np.ascontiguousarray(fpgaDma.card2np(
                        C2H_DEVICES[bank], address, combined_size // 2)))
                physical.append(merge_2ddr(halves[0], halves[1]))
            c2h_ms = (time.perf_counter() - c2h_start) * 1000.0
            if not decode_outputs_flag:
                raise RuntimeError(
                    "physical output forwarding requires cpp_mapped runtime"
                )
            results.append(decode_outputs(name, physical))
            timings.append({"kernel": name, "event": int(event),
                            "h2c_ms": h2c_ms, "npu_ms": npu_ms,
                            "c2h_ms": c2h_ms, "submission_group_size": 1})
            submission_groups.append({
                "kind": "legacy", "kernels": [name], "submission_group_size": 1,
                "h2c_ms": h2c_ms, "c2h_ms": c2h_ms,
            })
        return results

    def run_compatible_groups(
        names: list[str], logical_calls: list[list], max_group: int,
        upload_masks: list[list[bool] | None] | None = None,
        decode_outputs_flag: bool = True,
    ) -> list[list]:
        """Group independent calls by codec ABI and restore logical order."""
        if upload_masks is None:
            upload_masks = [None] * len(names)
        outputs: list[list[np.ndarray] | None] = [None] * len(names)
        buckets = {}
        for index, name in enumerate(names):
            signature = (
                "cpp_mixed" if cpp_runtime is not None
                and args.cpp_mixed_signature_groups
                else cfg_registry.signature(name)
            )
            buckets.setdefault(signature, []).append(index)
        for indices in buckets.values():
            cursor = 0
            while cursor < len(indices):
                limit = min(len(indices), cursor + max_group)
                if cpp_runtime is not None:
                    while limit > cursor + 1 and cpp_runtime.max_group_size(
                            [records[names[i]] for i in indices[cursor:limit]]) < limit - cursor:
                        limit -= 1
                chosen = indices[cursor:limit]
                values = run_kernel_group(
                    [names[i] for i in chosen], [logical_calls[i] for i in chosen],
                    [upload_masks[i] for i in chosen],
                    decode_outputs_flag=decode_outputs_flag,
                )
                for index, value in zip(chosen, values):
                    outputs[index] = value
                cursor = limit
        if any(value is None for value in outputs):
            raise RuntimeError("grouped execution did not produce every output")
        return list(outputs)  # type: ignore[arg-type]

    def run_kernel(name: str, logical_inputs: list,
                   upload_mask: list[bool] | None = None) -> list[np.ndarray]:
        return run_kernel_group([name], [logical_inputs], [upload_mask])[0]

    def run_kernel_physical(name: str, logical_inputs: list) -> list:
        return run_kernel_group(
            [name], [logical_inputs], [None], decode_outputs_flag=False
        )[0]

    def run_device_chain(
        names: list[str], logical_calls: list[list], offsets: list[int],
        connections: dict[tuple[int, int], tuple[int, int]],
        download_masks: list[list[bool]] | None = None,
        input_scales: list[list[float | None]] | None = None,
        output_scales: list[list[float | None]] | None = None,
    ):
        """Execute a qualified dependent chain and return decoded outputs/handles."""
        nonlocal h2c_skipped_bytes
        if cpp_runtime is None:
            raise RuntimeError("device tensor chaining requires cpp_mapped runtime")
        packed = [tensor_codec.pack_resident_inputs(name, values)
                  for name, values in zip(names, logical_calls)]
        physical, input_handles, output_handles, group = (
            cpp_runtime.run_resident_chain(
                [records[name] for name in names], packed, offsets,
                connections, args.timeout_ms, download_masks=download_masks,
                input_scales=input_scales, output_scales=output_scales,
            )
        )
        h2c_skipped_bytes += int(group["h2c_skipped_bytes"])
        submission_groups.append({
            "kind": "cpp_mapped_resident", "kernels": list(names),
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        decoded = []
        for name, outputs in zip(names, physical):
            if all(value is None for value in outputs):
                decoded.append([None] * len(outputs))
            elif any(value is None for value in outputs):
                raise RuntimeError(f"{name}: partial resident output decode is unsupported")
            else:
                decoded.append(decode_outputs(name, outputs))
        for index, (name, npu_ms) in enumerate(zip(names, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(names),
            })
        return decoded, input_handles, output_handles

    def run_fc_frame_graph(
        fc1_names: list[str], logical_input, fc2_name: str, scale: float,
    ) -> list[np.ndarray]:
        """Keep the complete FC1/GELU/FC2 schedule inside one C++ call."""
        if cpp_runtime is None:
            raise RuntimeError("FC frame graph requires cpp_mapped runtime")
        expected_input = cfg_registry.descriptors[fc1_names[0]]["input"][0]
        for name in fc1_names[1:]:
            if (cfg_registry.descriptors[name]["input"][0].identity()
                    != expected_input.identity()):
                raise RuntimeError("FC1 frame graph input descriptors differ")
        packed = tensor_codec.pack_resident_inputs(
            fc1_names[0], [logical_input]
        )[0]
        physical, group = cpp_runtime.run_fc1_gelu_fc2(
            [records[name] for name in fc1_names], packed,
            records[fc2_name], scale, args.timeout_ms,
        )
        names = [*fc1_names, fc2_name]
        submission_groups.append({
            "kind": "cpp_fc_frame_graph", "kernels": names,
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        for index, (name, npu_ms) in enumerate(zip(names, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(names),
            })
        return decode_outputs(fc2_name, physical)

    def run_attention_post_frame_graph(
        names: list[str], logical_calls: list[list], metadata: list[tuple],
        post_name: str, residual: np.ndarray, scale: float,
    ) -> list[np.ndarray]:
        """Keep one complete 280 attention/post boundary inside C++."""
        if cpp_runtime is None:
            raise RuntimeError("attention/post frame graph requires cpp_mapped runtime")
        packed = [tensor_codec.pack_inputs(name, values)
                  for name, values in zip(names, logical_calls)]
        residual_descriptor = cfg_registry.descriptors[post_name]["input"][1]
        post_residual = cpp_runtime.pack_tensor(
            np.ascontiguousarray(residual[:, None]), residual_descriptor
        )
        valid_widths = [
            width for _, q0_length, q1_length in metadata
            for width in (q0_length, q1_length)
        ]
        physical, group = cpp_runtime.run_attention_post_frame_graph(
            [records[name] for name in names], packed, records[post_name],
            post_residual, valid_widths, scale,
            args.attention_launch_group, args.timeout_ms,
        )
        kernels = [*names, post_name]
        submission_groups.append({
            "kind": "cpp_attention_post_frame_graph", "kernels": kernels,
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        for index, (name, npu_ms) in enumerate(zip(kernels, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(kernels),
            })
        return decode_outputs(post_name, physical)

    def run_fused_qkv_attention_post_frame_graph(
        fused_name: str, logical_input, post_name: str, residual: np.ndarray,
        valid_widths: list[int], scale: float, heads: int,
        capture_post_code: bool = False,
    ) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
        """Run one fused QKV/attention program and post in one C++ call."""
        if cpp_runtime is None:
            raise RuntimeError("fused QKV/attention requires cpp_mapped runtime")
        packed = tensor_codec.pack_inputs(fused_name, [logical_input])[0]
        residual_descriptor = cfg_registry.descriptors[post_name]["input"][1]
        post_residual = cpp_runtime.pack_tensor(
            np.ascontiguousarray(residual[:, None]), residual_descriptor
        )
        physical, group = (
            cpp_runtime.run_fused_qkv_attention_post_frame_graph(
                records[fused_name], packed, records[post_name], post_residual,
                valid_widths, scale, heads, args.timeout_ms,
                capture_post_code=capture_post_code,
            )
        )
        captured_post_code = group.pop("captured_post_code", None)
        kernels = [fused_name, post_name]
        submission_groups.append({
            "kind": "cpp_fused_qkv_attention_post_frame_graph",
            "kernels": kernels,
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        for index, (name, npu_ms) in enumerate(zip(kernels, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(kernels),
            })
        return decode_outputs(post_name, physical), captured_post_code

    def run_attention6_post_frame_graph(
        fused_name: str, logical_calls: list[list], post_name: str,
        residual, valid_widths: list[int], scale: float,
        heads: int, capture_post_code: bool = False,
        resident_offsets: dict[str, int] | None = None,
        norm_name: str | None = None,
    ) -> tuple[
        list[np.ndarray], tuple[np.ndarray, np.ndarray] | None,
        list[np.ndarray] | None,
    ]:
        """Run calibrated host-A8 six-head attention and post in one C++ call."""
        if cpp_runtime is None:
            raise RuntimeError("six-head attention fusion requires cpp_mapped runtime")
        if len(logical_calls) % heads:
            raise RuntimeError("six-head attention calls are not head-aligned")
        calls_per_head = len(logical_calls) // heads
        resident_post = isinstance(residual, mapped_runtime.DeviceTensorHandle)
        if resident_post:
            if resident_offsets is None or norm_name is None:
                raise RuntimeError(
                    "resident attention6 requires post/norm offsets and norm kernel"
                )
            post_residual = residual
        else:
            if resident_offsets is not None or norm_name is not None:
                raise RuntimeError(
                    "host-resident attention6 cannot use resident post metadata"
                )
            residual_descriptor = cfg_registry.descriptors[post_name]["input"][1]
            post_residual = cpp_runtime.pack_tensor(
                np.ascontiguousarray(residual[:, None]), residual_descriptor
            )
        if calls_per_head == 1 and not resident_post:
            flattened = [value for call in logical_calls for value in call]
            packed = tensor_codec.pack_inputs(fused_name, flattened)
            physical, group = cpp_runtime.run_fused_qkv_attention_post_frame_graph(
                records[fused_name], packed, records[post_name], post_residual,
                valid_widths, scale, heads, args.timeout_ms,
                output_order="head-major",
                capture_post_code=capture_post_code,
            )
            kernels = [fused_name, post_name]
        else:
            physical_calls = []
            for call_index in range(calls_per_head):
                call_values = [
                    value
                    for head_index in range(heads)
                    for value in logical_calls[
                        head_index * calls_per_head + call_index
                    ]
                ]
                physical_calls.append(
                    tensor_codec.pack_inputs(fused_name, call_values)
                )
            physical, group = cpp_runtime.run_multi_attention6_post_frame_graph(
                records[fused_name], physical_calls, records[post_name],
                post_residual, valid_widths, scale, heads, args.timeout_ms,
                capture_post_code=capture_post_code,
                post_offset_units=(0 if resident_offsets is None
                                   else resident_offsets["post"]),
                norm_record=(None if norm_name is None else records[norm_name]),
                norm_offset_units=(0 if resident_offsets is None
                                   else resident_offsets["norm2"]),
            )
            kernels = [fused_name] * calls_per_head + [post_name]
            if norm_name is not None:
                kernels.append(norm_name)
        captured_post_code = group.pop("captured_post_code", None)
        submission_groups.append({
            "kind": "cpp_attention6_post_frame_graph",
            "kernels": kernels,
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        for index, (name, npu_ms) in enumerate(zip(kernels, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(kernels),
            })
        post_output_count = int(group.get("post_output_count", len(physical)))
        post_outputs = decode_outputs(post_name, physical[:post_output_count])
        norm_outputs = (
            None if norm_name is None
            else decode_outputs(norm_name, physical[post_output_count:])
        )
        return post_outputs, captured_post_code, norm_outputs

    def run_qkv_attention6_post_frame_graph(
        qkv_name: str, logical_input, attention_name: str, post_name: str,
        residual: np.ndarray, qkv_scales: list[float],
        valid_widths: list[int], post_scale: float, heads: int,
    ) -> list[np.ndarray]:
        """Run the exact r168 QKV/attention6/post path in one C++ call."""
        if cpp_runtime is None:
            raise RuntimeError("QKV/attention6 frame graph requires cpp_mapped runtime")
        packed_qkv = tensor_codec.pack_inputs(qkv_name, [logical_input])[0]
        residual_descriptor = cfg_registry.descriptors[post_name]["input"][1]
        post_residual = cpp_runtime.pack_tensor(
            np.ascontiguousarray(residual[:, None]), residual_descriptor
        )
        physical, group = cpp_runtime.run_qkv_attention6_post_frame_graph(
            records[qkv_name], packed_qkv, records[attention_name],
            records[post_name], post_residual, qkv_scales, valid_widths,
            post_scale, heads, args.timeout_ms,
        )
        kernels = [qkv_name, attention_name, post_name]
        submission_groups.append({
            "kind": "cpp_qkv_attention6_post_frame_graph", "kernels": kernels,
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        for index, (name, npu_ms) in enumerate(zip(kernels, group["npu_ms"])):
            timings.append({
                "kernel": name, "event": 0,
                "h2c_ms": float(group["h2c_ms"]) if index == 0 else 0.0,
                "npu_ms": float(npu_ms),
                "c2h_ms": float(group["c2h_ms"]) if index == 0 else 0.0,
                "submission_group_size": len(kernels),
            })
        return decode_outputs(post_name, physical)

    try:
        captures = []
        block_outputs = []
        post_outputs = []
        qkv_outputs = []
        attention_outputs = []
        activation_outputs = []
        executed_layers = []
        resident_capture_handles = {}
        capture_offsets = (decoder_capture_offset_plan(
            records, contract, cpp_runtime.workspace_bytes_per_bank)
            if args.decoder_resident_captures else {})
        if ("frontend" in contract and args.encoder_captures is None
                and args.encoder_resume is None):
            frontend = plan.get("frontend")
            if frontend is None:
                raise ValueError("runtime contract enables frontend but host plan does not")
            if list(x.shape) != frontend["input_shape"]:
                raise ValueError(f"RGB input shape {x.shape} does not match frontend")
            n, channels, height, width = x.shape
            patch = int(frontend["patch_size"])
            grid_h, grid_w = frontend["patch_grid"]
            with host_profiler.measure(
                "frontend.patchify", elements=int(x.size), nbytes=int(x.nbytes)
            ):
                patchified = x.reshape(
                    n, channels, grid_h, patch, grid_w, patch
                ).transpose(0, 3, 5, 1, 2, 4).reshape(
                    n, channels * patch * patch, grid_h, grid_w
                )
                projection = contract["frontend"]["patch_projection"]
            projection_input = patchified
            if projection["input_dtype"] == "INT8":
                if args.encoder_quantize_pack:
                    projection_input = tensor_codec.quantized(
                        patchified,
                        projection["input_quantization"]["scale"],
                    )
                else:
                    projection_input = host_executor.quantize(
                        patchified,
                        projection["input_quantization"]["scale"],
                    )
            reusable_projection_input = (
                projection_input
                if isinstance(projection_input, tensor_codec.QuantizedInput)
                else tensor_codec.reusable(projection_input)
            )
            projection_names = frontend["projection_kernels"]
            projection_outputs = [
                values[0] for values in run_compatible_groups(
                    projection_names,
                    [[reusable_projection_input] for _ in projection_names],
                    args.frontend_launch_group,
                )
            ]
            projection_bytes = sum(int(value.nbytes) for value in projection_outputs)
            with host_profiler.measure(
                "frontend.token_assembly",
                elements=sum(int(value.size) for value in projection_outputs),
                nbytes=projection_bytes,
            ):
                projected = host_executor.concatenate(projection_outputs, axis=1)
                patch_tokens = projected.transpose(0, 2, 3, 1).reshape(
                    n, grid_h * grid_w, projected.shape[1]
                )
                cls_token = np.ascontiguousarray(np.broadcast_to(
                    env[frontend["cls_token"]], (n, 1, projected.shape[1])
                ))
                x = np.ascontiguousarray(
                    host_executor.add(
                        host_executor.concatenate(
                            [cls_token, np.ascontiguousarray(patch_tokens)], axis=1
                        ),
                        env[frontend["pos_embed"]],
                    ), dtype=np.float32
                )
            if not args.depth_only:
                frontend_captures = {
                    "frontend_patch_projection": projected.copy(),
                    "frontend_tokens": x.copy(),
                }
        if args.encoder_captures is not None:
            with np.load(args.encoder_captures, allow_pickle=False) as archive:
                captures = [np.ascontiguousarray(archive[f"capture_l{layer:02d}"],
                                                dtype=np.float32)
                            for layer in plan["capture_layers"]]
            encoder_blocks = []
        elif args.encoder_resume is not None:
            start_layer = args.encoder_start_layer
            with np.load(args.encoder_resume, allow_pickle=False) as archive:
                x = np.ascontiguousarray(
                    archive[f"block_l{start_layer - 1:02d}"], dtype=np.float32
                )
                captures = [
                    np.ascontiguousarray(archive[f"capture_l{layer:02d}"],
                                         dtype=np.float32)
                    for layer in plan["capture_layers"] if layer < start_layer
                ]
            encoder_blocks = zip(
                contract["encoder"][start_layer:], plan["encoder"][start_layer:]
            )
        else:
            encoder_blocks = zip(contract["encoder"], plan["encoder"])
        for block, norm_specs in encoder_blocks:
            layer_index = int(block["layer"])
            trace_encoder_internal = layer_index in encoder_internal_trace_layers
            if trace_encoder_internal:
                light_encoder_internal_trace[
                    f"encoder_l{layer_index:02d}_input"
                ] = x.copy()
            internal_stage = (
                args.replace_encoder_internal[1]
                if (args.replace_encoder_internal is not None
                    and args.replace_encoder_internal[0] == layer_index)
                else None
            )
            attention_head_replacement = (
                args.replace_encoder_attention_head[1]
                if (args.replace_encoder_attention_head is not None
                    and args.replace_encoder_attention_head[0] == layer_index)
                else None
            )
            executed_layers.append(layer_index)
            norm1 = norm_specs["norm1"]
            norm1_contract = block["host_norm1"]
            fused_norm1_qkv = (
                norm1_contract.get("fused_into") == block["qkv"]["kernel"]
            )
            chained_norm1_qkv = (
                norm1_contract.get("device_chain_to") == block["qkv"]["kernel"]
            )
            resident_offsets = None
            resident_x_handle = None
            chained_qkv_values = None
            if chained_norm1_qkv:
                if cpp_runtime is None or args.layout_codec != "native":
                    raise RuntimeError(
                        "LayerNorm/QKV device chain requires cpp_mapped native runtime"
                    )
                scale = float(norm1_contract["output_quantization"]["scale"])
                norm_name = norm1_contract["npu_core"]
                norm_output_address = int(records[norm_name]["outputs"][0]["address"])
                qkv_offset = norm_output_address - int(
                    records[block["qkv"]["kernel"]]["inputs"][0]["address"]
                )
                values, _, _ = run_device_chain(
                    [norm_name, block["qkv"]["kernel"]],
                    [[tensor_codec.reusable(x[:, None])], [None]],
                    [0, qkv_offset], {(1, 0): (0, 0)},
                    download_masks=[[False], [True, True, True]],
                    input_scales=[[None], [scale]],
                    output_scales=[[scale], [None, None, None]],
                )
                chained_qkv_values = values[1]
                normalized = None
                if (internal_stage == "norm1" or trace_encoder_internal
                        or args.collect_calibration):
                    with host_profiler.measure(
                        "encoder.layernorm_affine_diagnostic",
                        elements=int(x.size), nbytes=int(x.nbytes),
                    ):
                        normalized = layer_norm(
                            x, env[norm1["scale"]], env[norm1["bias"]],
                            norm1["axis"], norm1["epsilon"]
                        )
            elif fused_norm1_qkv:
                # The compiled program consumes the residual in BF16, computes
                # the pure LayerNorm core, and applies gamma/beta through the
                # algebraically folded Q/K/V weights and biases.  Keep a host
                # normalization only when a diagnostic mode explicitly needs
                # the logical norm1 tensor.
                normalized = None
                if (internal_stage == "norm1" or trace_encoder_internal
                        or args.collect_calibration):
                    with host_profiler.measure(
                        "encoder.layernorm_affine_diagnostic",
                        elements=int(x.size), nbytes=int(x.nbytes),
                    ):
                        normalized = layer_norm(
                            x, env[norm1["scale"]], env[norm1["bias"]],
                            norm1["axis"], norm1["epsilon"]
                        )
            elif "npu_core" in norm1_contract:
                if args.encoder_resident_intermediates:
                    source_capture = int(block["layer"]) - 1
                    resident_offsets = encoder_resident_offset_plan(
                        records, block, cpp_runtime.workspace_bytes_per_bank,
                        capture_offsets.get(source_capture),
                    )
                    reusable_x = tensor_codec.reusable(x[:, None])
                    resident_values, resident_inputs, _ = run_device_chain(
                        [norm1_contract["npu_core"]], [[reusable_x]],
                        [resident_offsets["norm1"]], {},
                    )
                    core = resident_values[0][0][:, 0]
                    resident_x_handle = resident_inputs[0][0]
                    if resident_x_handle is None:
                        raise RuntimeError("norm1 did not retain its residual input")
                    if source_capture in capture_offsets:
                        resident_capture_handles[source_capture] = resident_x_handle
                else:
                    core = run_kernel(
                        norm1_contract["npu_core"], [x[:, None]]
                    )[0][:, 0]
                with host_profiler.measure(
                    "encoder.layernorm_affine",
                    elements=int(core.size), nbytes=int(core.nbytes),
                ):
                    if norm1_contract.get("affine") == "folded":
                        normalized = np.ascontiguousarray(core, dtype=np.float32)
                    else:
                        normalized = (core * env[norm1["scale"]]
                                      + env[norm1["bias"]]).astype(np.float32)
            else:
                with host_profiler.measure(
                    "encoder.layernorm_affine",
                    elements=int(x.size), nbytes=int(x.nbytes),
                ):
                    normalized = layer_norm(
                        x, env[norm1["scale"]], env[norm1["bias"]],
                        norm1["axis"], norm1["epsilon"]
                    )
            if internal_stage == "norm1":
                if normalized is None:
                    raise RuntimeError("norm1 replacement requires a logical norm1 tensor")
                normalized = load_reference_replacement(
                    replacement_archive,
                    f"encoder_l{layer_index:02d}_norm1", normalized,
                )
            if trace_encoder_internal:
                if normalized is None:
                    raise RuntimeError("norm1 tracing requires a logical norm1 tensor")
                light_encoder_internal_trace[
                    f"encoder_l{layer_index:02d}_norm1"
                ] = normalized.copy()
            if args.collect_calibration:
                if normalized is None:
                    raise RuntimeError("norm1 calibration requires a logical norm1 tensor")
                hybrid_calibration[f"/blocks.{block['layer']}/norm1/LayerNormalization_output_0"] = calibration_stats(normalized)
            if chained_norm1_qkv:
                qkv_input = None
            elif fused_norm1_qkv:
                qkv_input = tensor_codec.reusable(x[:, None])
            elif args.encoder_quantize_pack:
                qkv_input = tensor_codec.quantized(
                    normalized[:, None],
                    block["qkv"]["input_quantization"]["scale"],
                )
            else:
                code = profiled_quantize(
                    "encoder.quantize_qkv", normalized,
                    block["qkv"]["input_quantization"]["scale"],
                )
                qkv_input = code[:, None]
            fused_qkv_attention = (
                block.get("qkv_attention_fused")
                if (args.fused_qkv_attention
                    and layer_index in args.fused_qkv_attention_layer_set)
                else None
            )
            if fused_qkv_attention is not None and (
                    chained_norm1_qkv or fused_norm1_qkv or qkv_input is None):
                raise RuntimeError(
                    f"layer {layer_index}: fused QKV/attention requires the "
                    "calibrated host-normalized A8 QKV input ABI"
                )
            qkv_values = (
                [] if (fused_qkv_attention is not None
                       or args.qkv_attention6_frame_graph) else
                (chained_qkv_values if chained_qkv_values is not None
                 else run_kernel(block["qkv"]["kernel"], [qkv_input]))
            )
            qkv_attention_values = None
            if (fused_qkv_attention is not None
                    or args.qkv_attention6_frame_graph):
                q = k = v = None
            elif block["qkv"].get("output_layout") == "concatenated_qkv":
                if len(qkv_values) != 1 or qkv_values[0].shape[-1] % 3:
                    raise RuntimeError("concatenated QKV kernel returned an invalid shape")
                q, k, v = (
                    np.ascontiguousarray(value)
                    for value in np.split(qkv_values[0], 3, axis=-1)
                )
            elif block["qkv"].get("output_layout") == "head_split_qkv":
                heads = block["attention"]["heads"]
                head_count = len(heads)
                if len(qkv_values) != head_count * 3:
                    raise RuntimeError(
                        "head-split QKV kernel returned an invalid output count"
                    )
                split = {
                    branch: qkv_values[index * head_count:(index + 1) * head_count]
                    for index, branch in enumerate(("q", "k", "v"))
                }
                if any(value.dtype != np.int8 for values in split.values()
                       for value in values):
                    raise RuntimeError("head-split QKV outputs must be INT8")
                qkv_attention_values = tuple(
                    np.ascontiguousarray(np.concatenate(split[branch], axis=3))
                    for branch in ("q", "k", "v")
                )
                # Preserve diagnostic trace semantics: Q/K/V captures remain
                # dequantized tensors comparable with the FP32 reference,
                # while attention consumes the original INT8 codes.
                logical = {}
                for branch in ("q", "k", "v"):
                    logical[branch] = np.ascontiguousarray(np.concatenate([
                        value.astype(np.float32)
                        * float(head["scales_bf16"][branch])
                        for value, head in zip(split[branch], heads)
                    ], axis=3))
                q, k, v = logical["q"], logical["k"], logical["v"]
            elif block["qkv"].get("output_layout") == "attention_ready_qkv":
                heads = block["attention"]["heads"]
                head_count = len(heads)
                if len(qkv_values) != head_count * 4:
                    raise RuntimeError(
                        "attention-ready QKV kernel returned an invalid output count"
                    )
                q0 = qkv_values[0:head_count]
                kt = qkv_values[head_count:2 * head_count]
                vh = qkv_values[2 * head_count:3 * head_count]
                q1 = qkv_values[3 * head_count:4 * head_count]
                if any(value.dtype != np.int8
                       for values in (q0, kt, vh, q1) for value in values):
                    raise RuntimeError(
                        "attention-ready QKV outputs must be INT8"
                    )
                qh = [
                    np.ascontiguousarray(np.concatenate((first, second), axis=2))
                    for first, second in zip(q0, q1)
                ]
                kh = [
                    np.ascontiguousarray(value.transpose(0, 2, 3, 1))
                    for value in kt
                ]
                qkv_attention_values = tuple(
                    np.ascontiguousarray(np.concatenate(values, axis=3))
                    for values in (qh, kh, vh)
                )
                logical = {}
                for branch, values in zip(("q", "k", "v"),
                                          (qh, kh, vh)):
                    logical[branch] = np.ascontiguousarray(np.concatenate([
                        value.astype(np.float32)
                        * float(head["scales_bf16"][branch])
                        for value, head in zip(values, heads)
                    ], axis=3))
                q, k, v = logical["q"], logical["k"], logical["v"]
            else:
                q, k, v = qkv_values
            if internal_stage == "qkv":
                if fused_qkv_attention is not None:
                    raise RuntimeError(
                        "QKV replacement is unavailable on the fused production path"
                    )
                q = load_reference_replacement(
                    replacement_archive, f"encoder_l{layer_index:02d}_q", q
                )
                k = load_reference_replacement(
                    replacement_archive, f"encoder_l{layer_index:02d}_k", k
                )
                v = load_reference_replacement(
                    replacement_archive, f"encoder_l{layer_index:02d}_v", v
                )
            if not args.depth_only:
                qkv_outputs.append((q.copy(), k.copy(), v.copy()))
            if layer_index in attention_trace_layers:
                light_attention_trace.update({
                    f"q_l{layer_index:02d}": q.copy(),
                    f"k_l{layer_index:02d}": k.copy(),
                    f"v_l{layer_index:02d}": v.copy(),
                })
            post_name = block["post_attention"]["kernel"]
            attention_fusion = (
                args.depth_only and not args.collect_calibration
                and internal_stage != "attention"
                and attention_head_replacement is None
                and layer_index not in attention_trace_layers
                and cpp_runtime is not None and host_executor.backend == "cpp"
                and codec_selection.native_for(post_name, "input")
                and all(codec_selection.native_for(head["kernel"], "output")
                        for head in block["attention"]["heads"])
            )
            head_outputs = []
            attention_physical_by_head = {}
            attention_source_descriptors_by_head = {}
            attention_valid_widths_by_head = {}
            attention_output_by_head = {}
            attention_names = []
            attention_call_inputs = []
            attention_call_masks = []
            attention_call_metadata = []
            fused_post = None
            fused_norm2_core = None
            attention_heads = block["attention"]["heads"]
            if fused_qkv_attention is not None:
                if not attention_fusion:
                    raise RuntimeError(
                        f"layer {layer_index}: fused QKV/attention requires "
                        "the qualified physical post bridge"
                    )
                heads = int(fused_qkv_attention.get("heads", 0))
                valid_widths = [int(value) for value in
                                fused_qkv_attention.get(
                                    "valid_widths_head_major", [])]
                if (heads != len(attention_heads)
                        or fused_qkv_attention.get("output_order")
                        != "chunk-major"
                        or float(fused_qkv_attention.get(
                            "amplitude_gain", 1.0)) != 1.0):
                    raise RuntimeError(
                        f"layer {layer_index}: fused attention contract is invalid"
                    )
                fused_outputs, captured_post_code = (
                    run_fused_qkv_attention_post_frame_graph(
                    fused_qkv_attention["kernel"], qkv_input, post_name, x,
                    valid_widths,
                    block["post_attention"]["input_quantization"]["scale"],
                    heads,
                    capture_post_code=trace_encoder_internal,
                ))
                fused_post = fused_outputs[0][:, 0]
                if captured_post_code is not None:
                    for bank, payload in enumerate(captured_post_code):
                        light_encoder_internal_trace[
                            f"encoder_l{layer_index:02d}_post_input_bank{bank}"
                        ] = np.ascontiguousarray(payload)
                attention_heads = []
            attention6 = (
                block.get("attention_fused")
                if (args.fused_attention6
                    and layer_index in args.fused_attention6_layer_set)
                else None
            )
            if args.qkv_attention6_frame_graph:
                if not attention_fusion or attention6 is None:
                    raise RuntimeError(
                        f"layer {layer_index}: QKV/attention6 frame graph "
                        "requires the qualified fused attention contract"
                    )
                heads = int(attention6.get("heads", 0))
                valid_widths = [int(value) for value in
                                attention6.get("valid_widths_head_major", [])]
                if (heads != len(attention_heads)
                        or attention6.get("input_order")
                        != "head-major-q0-k-v-q1"
                        or attention6.get("output_order") != "head-major"
                        or float(attention6.get("amplitude_gain", 1.0)) != 1.0):
                    raise RuntimeError(
                        f"layer {layer_index}: fused attention6 contract is invalid"
                    )
                qkv_scales = [
                    float(head["scales_bf16"][branch])
                    for head in attention_heads for branch in ("q", "k", "v")
                ]
                fused_post = run_qkv_attention6_post_frame_graph(
                    block["qkv"]["kernel"], qkv_input,
                    attention6["kernel"], post_name, x, qkv_scales,
                    valid_widths,
                    block["post_attention"]["input_quantization"]["scale"],
                    heads,
                )[0][:, 0]
                attention_heads = []
            for head in attention_heads:
                head_index = int(head["head"])
                if head_index in block.get("host_attention_heads", []):
                    if q.dtype == np.int8 or k.dtype == np.int8 or v.dtype == np.int8:
                        raise RuntimeError(
                            "host FP32 attention requires BF16-decoded Q/K/V"
                        )
                    begin = head_index * 64
                    end = begin + 64
                    with host_profiler.measure(
                        "encoder.attention_host_fp32",
                        elements=int(q.shape[2]) * 64 * 3,
                        nbytes=int(q.shape[2]) * 64 * 3 * 4,
                    ):
                        host_attention = fp32_attention_head(
                            q[0, 0, :, begin:end],
                            k[0, 0, :, begin:end],
                            v[0, 0, :, begin:end],
                        )
                    if attention_fusion:
                        host_attention = bf16_quantization_surrogate(
                            host_attention,
                            block["post_attention"]["input_quantization"]["scale"],
                            host_executor.quantize,
                        )
                        descriptors = cfg_registry.descriptors[
                            head["kernel"]
                        ]["output"]
                        for call in head["calls"]:
                            for descriptor, key in zip(
                                    descriptors, ("q0_rows", "q1_rows")):
                                start, stop = call[key]
                                logical = np.zeros(
                                    descriptor.dims, dtype=np.float32
                                )
                                logical[:, :, :stop - start] = (
                                    host_attention[:, :, start:stop]
                                )
                                attention_physical_by_head.setdefault(
                                    head_index, []
                                ).append(
                                    cpp_runtime.pack_tensor_symmetric(
                                        logical, descriptor
                                    )
                                )
                                attention_source_descriptors_by_head.setdefault(
                                    head_index, []
                                ).append(descriptor)
                                attention_valid_widths_by_head.setdefault(
                                    head_index, []
                                ).append(stop - start)
                    else:
                        attention_output_by_head[head_index] = host_attention
                    continue
                qkv_elements = int(q.shape[2]) * 64 * 3
                with host_profiler.measure(
                    "encoder.attention_qkv_head_quantize",
                    elements=qkv_elements,
                    nbytes=qkv_elements * (1 if q.dtype == np.int8 else 5),
                ):
                    attention_values = (
                        qkv_attention_values
                        if qkv_attention_values is not None else (q, k, v)
                    )
                    qh, kh, vh = attention_head_inputs(
                        *attention_values, head, host_executor.quantize
                    )
                with host_profiler.measure(
                    "encoder.attention_input_assembly",
                    elements=int(qh.size + kh.size + vh.size),
                    nbytes=int(qh.nbytes + kh.nbytes + vh.nbytes),
                ):
                    call_inputs = []
                    call_masks = []
                    q0_lengths = []
                    q1_lengths = []
                    reusable_k = tensor_codec.reusable(kh.T[None, None])
                    reusable_v = tensor_codec.reusable(vh[None, None])
                    for call_index, call in enumerate(head["calls"]):
                        q0 = qh[call["q0_rows"][0]:call["q0_rows"][1]]
                        q1 = np.zeros((256, 64), np.int8)
                        q1_values = qh[call["q1_rows"][0]:call["q1_rows"][1]]
                        q1[:q1_values.shape[0]] = q1_values
                        call_inputs.append([
                            q0[None, None], reusable_k,
                            reusable_v, q1[None, None]
                        ])
                        call_masks.append(
                            [True, call_index == 0, call_index == 0, True]
                            if args.attention_resident_kv else None
                        )
                        q0_lengths.append(q0.shape[0])
                        q1_lengths.append(q1_values.shape[0])
                for inputs, mask, q0_length, q1_length in zip(
                        call_inputs, call_masks, q0_lengths, q1_lengths):
                    attention_names.append(head["kernel"])
                    attention_call_inputs.append(inputs)
                    attention_call_masks.append(mask)
                    attention_call_metadata.append(
                        (head_index, q0_length, q1_length)
                    )

            # Schedule all hardware heads for this block together.  At 280x280
            # every head has one two-chunk attention call, so the production
            # launch group of three reduces six Python/C++ transactions to two
            # without changing the physical NPU program order or tensor ABI.
            # At 518x518 each head has three calls; a six-head program is reused
            # three times and the C++ frame graph preserves head/call order.
            use_attention6 = (
                attention6 is not None and attention_fusion
                and len(attention_call_inputs) >= len(block["attention"]["heads"])
                and len(attention_call_inputs) % len(block["attention"]["heads"]) == 0
                and len(attention_call_metadata) == len(attention_call_inputs)
            )
            if trace_encoder_internal and attention_call_inputs:
                for head_index, values in enumerate(attention_call_inputs):
                    for branch, value in zip(("q0", "k", "v", "q1"), values):
                        logical_value = (
                            value.value
                            if isinstance(value, RuntimeTensorCodec.ReusableInput)
                            else value
                        )
                        light_encoder_internal_trace[
                            f"encoder_l{layer_index:02d}_attention_input_"
                            f"h{head_index:02d}_{branch}"
                        ] = np.ascontiguousarray(logical_value)
            use_attention_post_graph = (
                args.attention_post_frame_graph and attention_fusion
                and not args.encoder_resident_intermediates
                and len(attention_names) == len(block["attention"]["heads"])
                and len({metadata[0] for metadata in attention_call_metadata})
                == len(block["attention"]["heads"])
            )
            if use_attention6:
                heads = int(attention6.get("heads", 0))
                valid_widths = [int(value) for value in
                                attention6.get("valid_widths_head_major", [])]
                if (heads != len(block["attention"]["heads"])
                        or attention6.get("input_order")
                        != "head-major-q0-k-v-q1"
                        or attention6.get("output_order") != "head-major"
                        or float(attention6.get("amplitude_gain", 1.0)) != 1.0):
                    raise RuntimeError(
                        f"layer {layer_index}: fused attention6 contract is invalid"
                    )
                calls_per_head = len(attention_call_inputs) // heads
                if int(attention6.get("calls_per_head", calls_per_head)) != calls_per_head:
                    raise RuntimeError(
                        f"layer {layer_index}: fused attention6 call count is invalid"
                    )
                fused_outputs, captured_post_code, fused_norm_outputs = (
                    run_attention6_post_frame_graph(
                    attention6["kernel"], attention_call_inputs,
                    post_name,
                    (resident_x_handle
                     if args.encoder_resident_intermediates else x),
                    valid_widths,
                    block["post_attention"]["input_quantization"]["scale"],
                    heads, capture_post_code=trace_encoder_internal,
                    resident_offsets=(resident_offsets
                                      if args.encoder_resident_intermediates
                                      else None),
                    norm_name=(block["host_norm2"].get("npu_core")
                               if args.encoder_resident_intermediates else None),
                ))
                fused_post = fused_outputs[0][:, 0]
                if fused_norm_outputs is not None:
                    fused_norm2_core = fused_norm_outputs[0][:, 0]
                if captured_post_code is not None:
                    for bank, payload in enumerate(captured_post_code):
                        light_encoder_internal_trace[
                            f"encoder_l{layer_index:02d}_post_input_bank{bank}"
                        ] = np.ascontiguousarray(payload)
            elif use_attention_post_graph:
                fused_post = run_attention_post_frame_graph(
                    attention_names, attention_call_inputs,
                    attention_call_metadata, post_name, x,
                    block["post_attention"]["input_quantization"]["scale"],
                )[0][:, 0]
            elif attention_names:
                grouped = run_compatible_groups(
                    attention_names, attention_call_inputs,
                    args.attention_launch_group, attention_call_masks,
                    decode_outputs_flag=not attention_fusion,
                )
                grouped_by_head = {}
                for outputs, metadata, name in zip(
                        grouped, attention_call_metadata, attention_names):
                    head_index, q0_length, q1_length = metadata
                    grouped_by_head.setdefault(head_index, []).append(
                        (outputs, q0_length, q1_length, name)
                    )
                for head in block["attention"]["heads"]:
                    head_index = int(head["head"])
                    calls = grouped_by_head.get(head_index, [])
                    if not calls:
                        continue
                    if attention_fusion:
                        descriptors = cfg_registry.descriptors[
                            head["kernel"]
                        ]["output"]
                        for outputs, q0_length, q1_length, _ in calls:
                            attention_physical_by_head.setdefault(
                                head_index, []
                            ).extend(outputs)
                            attention_source_descriptors_by_head.setdefault(
                                head_index, []
                            ).extend(descriptors)
                            attention_valid_widths_by_head.setdefault(
                                head_index, []
                            ).extend([q0_length, q1_length])
                    else:
                        grouped_elements = sum(
                            int(value.size) for outputs, _, _, _ in calls
                            for value in outputs
                        )
                        grouped_bytes = sum(
                            int(value.nbytes) for outputs, _, _, _ in calls
                            for value in outputs
                        )
                        with host_profiler.measure(
                            "encoder.attention_output_assembly",
                            elements=grouped_elements, nbytes=grouped_bytes,
                        ):
                            chunks = []
                            for (out0, out1), _, q1_length, _ in calls:
                                chunks.extend([
                                    np.ascontiguousarray(out0),
                                    np.ascontiguousarray(
                                        out1[:, :, :q1_length, :]
                                    ),
                                ])
                            attention_output_by_head[head_index] = (
                                host_executor.concatenate(chunks, axis=2)
                            )
            if attention_fusion:
                if fused_post is None:
                    ordered_heads = [
                        int(head["head"])
                        for head in block["attention"]["heads"]
                    ]
                    attention_physical = [
                        value for head in ordered_heads
                        for value in attention_physical_by_head[head]
                    ]
                    attention_source_descriptors = [
                        value for head in ordered_heads
                        for value in attention_source_descriptors_by_head[head]
                    ]
                    attention_valid_widths = [
                        value for head in ordered_heads
                        for value in attention_valid_widths_by_head[head]
                    ]
                    target_descriptor = cfg_registry.descriptors[post_name]["input"][0]
                    logical_elements = int(np.prod(target_descriptor.dims))
                    physical_bytes = sum(
                        descriptor.combined_bytes
                        for descriptor in attention_source_descriptors
                    ) + target_descriptor.combined_bytes
                    with host_profiler.measure(
                        "encoder.attention_pack_bf16_heads",
                        elements=logical_elements, nbytes=physical_bytes,
                    ):
                        physical_post_code = host_executor.attention_pack_bf16_heads(
                            attention_physical, attention_source_descriptors,
                            attention_valid_widths, target_descriptor,
                            block["post_attention"]["input_quantization"]["scale"],
                            len(block["attention"]["heads"]),
                        )
                    post_code = tensor_codec.prepacked(
                        physical_post_code, target_descriptor
                    )
                attention = None
            else:
                head_outputs = [
                    attention_output_by_head[int(head["head"])]
                    for head in block["attention"]["heads"]
                ]
                with host_profiler.measure(
                    "encoder.attention_output_assembly",
                    elements=sum(int(value.size) for value in head_outputs),
                    nbytes=sum(int(value.nbytes) for value in head_outputs),
                ):
                    attention = host_executor.concatenate(head_outputs, axis=3)
            if internal_stage == "attention":
                attention = load_reference_replacement(
                    replacement_archive,
                    f"encoder_l{layer_index:02d}_attention", attention,
                )
            elif attention_head_replacement is not None:
                reference_attention = load_reference_replacement(
                    replacement_archive,
                    f"encoder_l{layer_index:02d}_attention", attention,
                )
                begin = attention_head_replacement * 64
                attention = attention.copy()
                attention[..., begin:begin + 64] = reference_attention[
                    ..., begin:begin + 64
                ]
            if not args.depth_only:
                attention_outputs.append(attention.copy())
            if layer_index in attention_trace_layers:
                light_attention_trace[f"attention_l{layer_index:02d}"] = (
                    attention.copy()
                )
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/attn/Concat_6_output_0"] = calibration_stats(attention)
            if not attention_fusion:
                if args.encoder_quantize_pack:
                    post_code = tensor_codec.quantized(
                        attention,
                        block["post_attention"]["input_quantization"]["scale"],
                    )
                else:
                    post_code = profiled_quantize(
                        "encoder.post_attention_quantize", attention,
                        block["post_attention"]["input_quantization"]["scale"],
                    )
            norm2 = norm_specs["norm2"]
            norm2_contract = block["host_norm2"]
            if args.encoder_resident_intermediates:
                if (resident_offsets is None or resident_x_handle is None
                        or "npu_core" not in norm2_contract):
                    raise RuntimeError(
                        "encoder residency requires NPU norm1/norm2 and a live residual handle"
                    )
                if fused_post is not None:
                    if fused_norm2_core is None:
                        raise RuntimeError(
                            "resident attention fusion did not return norm2"
                        )
                    post = fused_post
                    core = fused_norm2_core
                else:
                    resident_values, _, _ = run_device_chain(
                        [post_name, norm2_contract["npu_core"]],
                        [[post_code, resident_x_handle], [None]],
                        [resident_offsets["post"], resident_offsets["norm2"]],
                        {(1, 0): (0, 0)},
                    )
                    post = resident_values[0][0][:, 0]
                    core = resident_values[1][0][:, 0]
            else:
                post = (fused_post if fused_post is not None else run_kernel(
                    post_name, [post_code, x[:, None]]
                )[0][:, 0])
                if internal_stage == "attention_branch":
                    attention_branch = load_reference_replacement(
                        replacement_archive,
                        f"encoder_l{layer_index:02d}_attention_branch", x,
                    )
                    post = host_executor.add(x, attention_branch)
                elif internal_stage == "post":
                    post = load_reference_replacement(
                        replacement_archive,
                        f"encoder_l{layer_index:02d}_post", post,
                    )
                core = (run_kernel(
                    norm2_contract["npu_core"], [post[:, None]]
                )[0][:, 0] if "npu_core" in norm2_contract else None)
            if not args.depth_only:
                post_outputs.append(post.copy())
            if trace_encoder_internal:
                light_encoder_internal_trace.update({
                    f"encoder_l{layer_index:02d}_attention_branch": (
                        post - x
                    ).copy(),
                    f"encoder_l{layer_index:02d}_post": post.copy(),
                })
            if core is not None:
                with host_profiler.measure(
                    "encoder.layernorm_affine",
                    elements=int(core.size), nbytes=int(core.nbytes),
                ):
                    if norm2_contract.get("affine") == "folded":
                        normalized = np.ascontiguousarray(core, dtype=np.float32)
                    else:
                        normalized = (core * env[norm2["scale"]]
                                      + env[norm2["bias"]]).astype(np.float32)
            else:
                with host_profiler.measure(
                    "encoder.layernorm_affine",
                    elements=int(post.size), nbytes=int(post.nbytes),
                ):
                    normalized = layer_norm(
                        post, env[norm2["scale"]], env[norm2["bias"]],
                        norm2["axis"], norm2["epsilon"]
                    )
            if internal_stage == "norm2":
                normalized = load_reference_replacement(
                    replacement_archive,
                    f"encoder_l{layer_index:02d}_norm2", normalized,
                )
            if trace_encoder_internal:
                light_encoder_internal_trace[
                    f"encoder_l{layer_index:02d}_norm2"
                ] = normalized.copy()
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/norm2/LayerNormalization_output_0"] = calibration_stats(normalized)
            native_gelu = block["mlp"].get("npu_activation")
            fused_fc1_pack = (
                args.encoder_quantize_pack and native_gelu is None
                and args.depth_only and not args.collect_calibration
                and internal_stage not in ("fc1", "gelu")
                and not trace_encoder_internal
            )
            if fused_fc1_pack:
                fc1_input = tensor_codec.quantized(
                    normalized[:, None],
                    block["mlp"]["fc1_input_quantization"]["scale"],
                )
                fc1_code = None
            else:
                fc1_code = profiled_quantize(
                    "encoder.quantize_fc1", normalized,
                    block["mlp"]["fc1_input_quantization"]["scale"],
                )
                fc1_input = tensor_codec.reusable(fc1_code[:, None])
            fused_fc2 = None
            if native_gelu is None:
                fc2_scale = block["mlp"]["fc2_input_quantization"]["scale"]
                fc1_names = block["mlp"]["fc1_kernels"]
                fc2_name = block["mlp"]["fc2_kernel"]
                physical_fusion = (
                    args.depth_only and not args.collect_calibration
                    and internal_stage not in ("fc1", "gelu")
                    and not trace_encoder_internal
                    and cpp_runtime is not None and host_executor.backend == "cpp"
                    and all(codec_selection.native_for(name, "output")
                            for name in fc1_names)
                    and codec_selection.native_for(fc2_name, "input")
                )
                if physical_fusion:
                    if args.encoder_fc_frame_graph:
                        fused_fc2 = run_fc_frame_graph(
                            fc1_names, fc1_input, fc2_name, fc2_scale
                        )[0][:, 0]
                    else:
                        fc1_calls = run_compatible_groups(
                            fc1_names,
                            [[fc1_input] for _ in fc1_names],
                            args.encoder_fc1_launch_group,
                            decode_outputs_flag=False,
                        )
                        fc1_physical = [
                            output for outputs in fc1_calls for output in outputs
                        ]
                        source_descriptors = [
                            descriptor for name in fc1_names
                            for descriptor in cfg_registry.descriptors[name]["output"]
                        ]
                        if len(fc1_physical) != len(source_descriptors):
                            raise RuntimeError(
                                "FC1 physical outputs do not match cfg descriptors"
                            )
                        target_descriptor = cfg_registry.descriptors[fc2_name]["input"][0]
                        logical_elements = sum(
                            int(np.prod(descriptor.dims))
                            for descriptor in source_descriptors
                        )
                        physical_bytes = sum(
                            descriptor.combined_bytes
                            for descriptor in source_descriptors
                        ) + target_descriptor.combined_bytes
                        with host_profiler.measure(
                            "encoder.gelu_pack_bf16_concatenate",
                            elements=logical_elements, nbytes=physical_bytes,
                        ):
                            physical_fc2_input = (
                                host_executor.gelu_pack_bf16_concatenate(
                                    fc1_physical, source_descriptors,
                                    target_descriptor, fc2_scale,
                                )
                            )
                        fc2_input = tensor_codec.prepacked(
                            physical_fc2_input, target_descriptor
                        )
                    activated = None
                else:
                    fc1_outputs = [
                        output for name in fc1_names
                        for output in run_kernel(name, [fc1_input])
                    ]
                    with host_profiler.measure(
                        "encoder.mlp_assembly",
                        elements=sum(int(value.size) for value in fc1_outputs),
                        nbytes=sum(int(value.nbytes) for value in fc1_outputs),
                    ):
                        hidden = host_executor.concatenate(fc1_outputs, axis=3)
                    if internal_stage == "fc1":
                        hidden = load_reference_replacement(
                            replacement_archive,
                            f"encoder_l{layer_index:02d}_fc1", hidden,
                        )
                    if trace_encoder_internal:
                        light_encoder_internal_trace[
                            f"encoder_l{layer_index:02d}_fc1"
                        ] = hidden.copy()
                    if internal_stage == "gelu":
                        activated = load_reference_replacement(
                            replacement_archive,
                            f"encoder_l{layer_index:02d}_gelu", gelu(hidden),
                        )
                        fc2_input = profiled_quantize(
                            "encoder.quantize_fc2", activated, fc2_scale
                        )
                    else:
                        with host_profiler.measure(
                            "encoder.gelu_quantize",
                            elements=int(hidden.size), nbytes=int(hidden.nbytes),
                        ):
                            fc2_input = host_executor.gelu_quantize(
                                hidden, fc2_scale
                            )
                        if (args.depth_only and not args.collect_calibration
                                and not trace_encoder_internal):
                            activated = None
                        else:
                            with host_profiler.measure(
                                "encoder.gelu",
                                elements=int(hidden.size), nbytes=int(hidden.nbytes),
                            ):
                                activated = gelu(hidden)
            else:
                if internal_stage == "fc1":
                    raise RuntimeError(
                        "FC1 replacement is unavailable for fused NPU GELU"
                    )
                conv_input = np.ascontiguousarray(
                    # Native-GELU contracts deliberately keep their logical
                    # INT8 input because it is transposed into NCHW here.
                    fc1_code.transpose(0, 2, 1)[:, :, None, :]
                )
                activated_code = host_executor.concatenate([
                    run_kernel(name, [conv_input])[0]
                    for name in block["mlp"]["fc1_kernels"]
                ], axis=1)
                fc2_input = np.ascontiguousarray(
                    activated_code.transpose(0, 2, 3, 1)
                )
                activated = (fc2_input.astype(np.float32)
                             * float(native_gelu["output_step"]))
                if internal_stage == "gelu":
                    activated = load_reference_replacement(
                        replacement_archive,
                        f"encoder_l{layer_index:02d}_gelu", activated,
                    )
                    fc2_input = profiled_quantize(
                        "encoder.quantize_fc2", activated,
                        block["mlp"]["fc2_input_quantization"]["scale"],
                    )
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/mlp/act/Mul_1_output_0"] = calibration_stats(activated)
            if trace_encoder_internal and activated is not None:
                light_encoder_internal_trace[
                    f"encoder_l{layer_index:02d}_gelu"
                ] = activated.copy()
            if not args.depth_only:
                activation_outputs.append(activated.copy())
            fc2 = (fused_fc2 if fused_fc2 is not None else
                   run_kernel(block["mlp"]["fc2_kernel"], [fc2_input])[0][:, 0])
            if internal_stage == "fc2":
                fc2 = load_reference_replacement(
                    replacement_archive,
                    f"encoder_l{layer_index:02d}_fc2", fc2,
                )
            if trace_encoder_internal:
                light_encoder_internal_trace[
                    f"encoder_l{layer_index:02d}_fc2"
                ] = fc2.copy()
            with host_profiler.measure(
                "encoder.residual",
                elements=int(post.size), nbytes=int(post.nbytes + fc2.nbytes),
            ):
                x = host_executor.add(post, fc2)
            if args.replace_encoder_block == layer_index:
                x = load_reference_replacement(
                    replacement_archive, f"block_l{layer_index:02d}", x
                )
            if not args.depth_only:
                block_outputs.append(x.copy())
            if (layer_index in attention_trace_layers
                    or layer_index in encoder_internal_trace_layers):
                light_attention_trace[f"block_l{layer_index:02d}"] = x.copy()
            if block["capture_for_decoder"]:
                captures.append(x.copy())
        for name, value in zip(plan["capture_tensor_names"], captures):
            env[name] = value

        resident_decoder_norms = {}
        native_boundary_nodes = set()
        fused_host_nodes = {
            node for name in fused_contract_nodes
            for node in contract_decoder[name]["fused_host_nodes"]
        }
        if args.decoder_native_boundary:
            capture_layers = tuple(plan["capture_layers"])
            if capture_layers != (2, 5, 8, 11):
                raise RuntimeError(
                    f"decoder native boundary capture order is not qualified: {capture_layers}"
                )
            captures_by_layer = dict(zip(capture_layers, captures))
            project_steps = [
                step for step in plan["decoder_steps"]
                if step["backend"] == "npu"
                and step["name"].startswith("/depth_head/projects.")
            ]
            if len(project_steps) != 4:
                raise RuntimeError("decoder native boundary requires four project steps")
            boundary_stems = []
            for project, (layer, step) in enumerate(
                    zip(capture_layers, project_steps)):
                if layer in resident_capture_handles:
                    norm_name = contract["encoder"][layer]["host_norm1"].get(
                        "npu_core"
                    )
                    if norm_name is None:
                        raise RuntimeError(
                            "resident decoder capture has no NPU LayerNorm ABI"
                        )
                    source_descriptor = replace(
                        cfg_registry.descriptors[norm_name]["output"][0],
                        direction="output", index=0, matrix_role="output",
                    )
                    source = resident_capture_handles[layer]
                else:
                    # Host-resident captures can enter the same native decoder
                    # bridge without first executing an NPU LayerNorm.  Reuse
                    # the post-attention residual's qualified BF16 NDWC input
                    # ABI; the C++ bridge performs final LayerNorm, layout
                    # conversion and A8 packing before the project Convs.
                    post_name = contract["encoder"][layer][
                        "post_attention"
                    ]["kernel"]
                    source_pack_descriptor = cfg_registry.descriptors[
                        post_name
                    ]["input"][1]
                    source_descriptor = replace(
                        source_pack_descriptor,
                        direction="output", index=0, matrix_role="output",
                    )
                    if (source_pack_descriptor.storage_identity()
                            != source_descriptor.storage_identity()):
                        raise RuntimeError(
                            "decoder host capture input/output storage ABI mismatch"
                        )
                    source = cpp_runtime.pack_tensor(
                        np.ascontiguousarray(captures_by_layer[layer][:, None]),
                        source_pack_descriptor,
                    )
                kernel_names = [item["name"] for item in step["kernels"]]
                target_descriptor = cfg_registry.descriptors[
                    kernel_names[0]
                ]["input"][0]
                boundary_stems.append({
                    "source": source,
                    "source_descriptor": source_descriptor,
                    "target_descriptor": target_descriptor,
                    "gamma": env["pretrained.norm.weight"],
                    "beta": env["pretrained.norm.bias"],
                    "scale": float(step["input_scale"]),
                    "epsilon": 1.0e-6,
                    "records": [records[name] for name in kernel_names],
                })
            physical_projects, boundary_group = (
                cpp_runtime.run_decoder_capture_stems(
                    boundary_stems, args.timeout_ms
                )
            )
            timing_cursor = 0
            for project, (step, physical_calls) in enumerate(
                    zip(project_steps, physical_projects)):
                names = [item["name"] for item in step["kernels"]]
                decoded_calls = [
                    decode_outputs(name, outputs)
                    for name, outputs in zip(names, physical_calls)
                ]
                env[step["outputs"][0]] = host_executor.concatenate(
                    [outputs[0] for outputs in decoded_calls], axis=1
                )
                submission_groups.append({
                    "kind": "cpp_decoder_frame_graph",
                    "kernels": names,
                    "submission_group_size": len(names),
                    "cpp_frame_graph": True,
                    "source_c2h_ms": (
                        boundary_group["source_c2h_ms"] if project == 0 else 0.0
                    ),
                    "bridge_ms": (
                        boundary_group["bridge_ms"] if project == 0 else 0.0
                    ),
                    "h2c_ms": boundary_group["h2c_ms"] if project == 0 else 0.0,
                    "c2h_ms": boundary_group["c2h_ms"] if project == 0 else 0.0,
                })
                for kernel_index, name in enumerate(names):
                    timings.append({
                        "kernel": name, "event": 0,
                        "h2c_ms": (boundary_group["h2c_ms"]
                                   if project == 0 and kernel_index == 0 else 0.0),
                        "npu_ms": boundary_group["npu_ms"][timing_cursor],
                        "c2h_ms": (boundary_group["c2h_ms"]
                                   if project == 0 and kernel_index == 0 else 0.0),
                        "submission_group_size": len(names),
                    })
                    timing_cursor += 1
            if timing_cursor != len(boundary_group["npu_ms"]):
                raise RuntimeError("decoder native boundary timing plan mismatch")
            # The physical bridge subsumes the initial four LayerNorms, their
            # constants/Slices/Transpose/Reshape chains, and project Conv steps.
            boundary_host_indices = {
                *range(0, 27), *range(31, 34), *range(37, 40), *range(41, 44),
            }
            for index, step in enumerate(plan["decoder_steps"][:45]):
                if (index in boundary_host_indices
                        or (step["backend"] == "npu"
                            and step["name"].startswith(
                                "/depth_head/projects."))):
                    native_boundary_nodes.add(step["name"])
        elif args.decoder_fused_stems:
            fused_specs = [
                contract_decoder[name] for name in sorted(fused_capture_nodes)
            ]
            fused_layers = {int(item["capture_layer"]) for item in fused_specs}
            missing = sorted((fused_layers - {11}) - set(resident_capture_handles))
            if missing:
                raise RuntimeError(f"decoder capture handles were not retained: {missing}")
            captures_by_layer = dict(zip(plan["capture_layers"], captures))
            for source_node in sorted(fused_capture_nodes):
                project = int(source_node.split("projects.", 1)[1].split("/", 1)[0])
                specification = contract_decoder[source_node]
                layer = int(specification["capture_layer"])
                parts = []
                for kernel in specification["kernels"]:
                    source = resident_capture_handles.get(layer)
                    if source is None:
                        source = tensor_codec.reusable(
                            captures_by_layer[layer][:, None])
                    decoded, inputs, _ = run_device_chain(
                        [kernel["name"]], [[source]],
                        [capture_offsets[layer]], {},
                    )
                    if layer not in resident_capture_handles:
                        handle = inputs[0][0]
                        if handle is None:
                            raise RuntimeError(
                                f"decoder capture layer {layer} was not retained"
                            )
                        resident_capture_handles[layer] = handle
                    parts.append(decoded[0][0])
                output = host_executor.concatenate(parts, axis=1)
                env[specification["output_tensor"]] = output
                if not args.depth_only and project == 0:
                    decoder_checkpoints["decoder_conv_00"] = output.copy()
        elif args.decoder_resident_captures:
            norm_steps = plan["decoder_steps"][:4]
            lowered = [contract_decoder_layernorm.get(step["name"])
                       for step in norm_steps]
            if (any(item is None for item in lowered)
                    or len({item["kernel"] for item in lowered}) != 1):
                raise RuntimeError(
                    "resident decoder captures require four compatible NPU LayerNorm entries"
                )
            missing = sorted(set((2, 5, 8)) - set(resident_capture_handles))
            if missing:
                raise RuntimeError(f"decoder capture handles were not retained: {missing}")
            names = [item["kernel"] for item in lowered]
            logical = [[resident_capture_handles[layer]] for layer in (2, 5, 8)]
            logical.append([tensor_codec.reusable(captures[-1][:, None])])
            decoded, decoder_inputs, _ = run_device_chain(
                names, logical,
                [capture_offsets[layer] for layer in (2, 5, 8, 11)], {},
            )
            last_handle = decoder_inputs[-1][0]
            if last_handle is None:
                raise RuntimeError("decoder capture layer 11 was not retained")
            resident_capture_handles[11] = last_handle
            for step, values in zip(norm_steps, decoded):
                core = values[0][:, 0]
                resident_decoder_norms[step["name"]] = np.ascontiguousarray(
                    core * env[step["inputs"][1]] + env[step["inputs"][2]],
                    dtype=np.float32,
                )

        for step in plan["decoder_steps"]:
            if step["name"] in native_boundary_nodes:
                continue
            if step["backend"] == "host":
                if step["name"] in fused_host_nodes:
                    continue
                lowered_norm = contract_decoder_layernorm.get(step["name"])
                if lowered_norm is not None:
                    if step["name"] in resident_decoder_norms:
                        env[step["outputs"][0]] = resident_decoder_norms[step["name"]]
                        continue
                    value = env[step["inputs"][0]]
                    core = run_kernel(
                        lowered_norm["kernel"], [value[:, None]]
                    )[0][:, 0]
                    env[step["outputs"][0]] = np.ascontiguousarray(
                        core * env[step["inputs"][1]]
                        + env[step["inputs"][2]], dtype=np.float32
                    )
                    continue
                execute_host(step, env, decoder_host_ops, host_executor)
                continue
            expected_decoder = contract_decoder[step["name"]]
            if expected_decoder.get("fused_decoder_stem"):
                if (expected_decoder.get("stem_input", "capture") != "capture"
                        and step["outputs"][0] not in env):
                    source = tensor_codec.reusable(
                        np.ascontiguousarray(
                            env[expected_decoder["input_tensor"]][:, None]
                        )
                    )
                    parts = [
                        run_kernel(kernel["name"], [source])[0]
                        for kernel in expected_decoder["kernels"]
                    ]
                    env[step["outputs"][0]] = host_executor.concatenate(
                        parts, axis=1
                    )
                if step["outputs"][0] not in env:
                    raise RuntimeError(
                        f"fused decoder stem did not produce {step['name']}"
                    )
                continue
            value = env[step["inputs"][0]]
            if args.collect_calibration:
                decoder_calibration[step["name"]] = calibration_stats(value)
            physical_width = step.get("physical_input_width")
            if physical_width is not None:
                with host_profiler.measure(
                    "decoder.width_pad",
                    elements=int(value.size), nbytes=int(value.nbytes),
                ):
                    physical_value = pad_nchw_width(value, int(physical_width))
            else:
                physical_value = value
            decoder_scale = float(step["input_scale"])
            if args.decoder_quantize_pack:
                code = tensor_codec.quantized(physical_value, decoder_scale)
            else:
                code = profiled_quantize(
                    "decoder.quantize", physical_value, decoder_scale
                )
            if step.get("channel_sliced"):
                with host_profiler.measure(
                    "decoder.tile_assembly",
                    elements=int(physical_value.size),
                    nbytes=int(physical_value.nbytes),
                ):
                    names = [item["name"] for item in step["kernels"]]
                    reusable_code = (
                        code if args.decoder_quantize_pack
                        else tensor_codec.reusable(code)
                    )
                    logical_calls = [[reusable_code] for _ in names]
                grouped = run_compatible_groups(
                    names, logical_calls, args.decoder_launch_group
                )
                parts = [item[0] for item in grouped]
                with host_profiler.measure(
                    "decoder.output_assembly",
                    elements=sum(int(part.size) for part in parts),
                    nbytes=sum(int(part.nbytes) for part in parts),
                ):
                    output = host_executor.concatenate(parts, axis=1)
            elif int(step["row_tiles"]) == 1:
                output = run_kernel(step["kernels"][0]["name"], [code])[0]
            else:
                with host_profiler.measure(
                    "decoder.tile_assembly",
                    elements=int(physical_value.size),
                    nbytes=int(physical_value.nbytes),
                ):
                    rows = int(step["tile_output_rows"])
                    count = int(step["row_tiles"])
                    is_3x3 = len(step["kernels"]) == 3
                    tile_source = physical_value if args.decoder_quantize_pack else code
                    names = []
                    tile_inputs = []
                    for tile in range(count):
                        begin = tile * rows; end = begin + rows
                        if not is_3x3:
                            name = step["kernels"][0]["name"]
                            tile_input = tile_source[:, :, begin:end]
                        elif tile == 0:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "first"
                            )
                            tile_input = tile_source[:, :, :end + 1]
                        elif tile == count - 1:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "last"
                            )
                            tile_input = tile_source[:, :, begin - 1:end]
                        else:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "middle"
                            )
                            tile_input = tile_source[:, :, begin - 1:end + 1]
                        names.append(name)
                        tile_inputs.append([
                            tensor_codec.quantized(tile_input, decoder_scale)
                            if args.decoder_quantize_pack else tile_input
                        ])
                grouped = run_compatible_groups(
                    names, tile_inputs, args.decoder_launch_group
                )
                parts = [item[0] for item in grouped]
                with host_profiler.measure(
                    "decoder.output_assembly",
                    elements=sum(int(part.size) for part in parts),
                    nbytes=sum(int(part.nbytes) for part in parts),
                ):
                    output = host_executor.concatenate(parts, axis=2)
            logical_width = step.get("logical_output_width")
            if logical_width is not None:
                with host_profiler.measure(
                    "decoder.width_crop",
                    elements=int(output.size), nbytes=int(output.nbytes),
                ):
                    output = crop_nchw_width(output, int(logical_width))
            decoder_index = int(step["index"])
            if args.replace_decoder_conv == decoder_index:
                output = load_reference_replacement(
                    replacement_archive, f"decoder_conv_{decoder_index:02d}", output
                )
            env[step["outputs"][0]] = output
            checkpoint_index = int(step["index"])
            default_decoder_checkpoints = (
                tuple(range(11)) + (13, 18, 23, 28, 29, 30, 31)
            )
            retain_decoder_checkpoint = (
                checkpoint_index in decoder_trace_convs
                or (not args.depth_only
                    and checkpoint_index in default_decoder_checkpoints)
            )
            if retain_decoder_checkpoint:
                decoder_checkpoints[
                    f"decoder_conv_{checkpoint_index:02d}"
                ] = output.copy()
                decoder_checkpoints[
                    f"decoder_input_{checkpoint_index:02d}"
                ] = value.copy()
        output = env[plan["model_outputs"][0]]
    finally:
        if waiter is not None:
            waiter.close()

    with host_profiler.measure(
        "result.serialize", elements=int(output.size), nbytes=int(output.nbytes)
    ):
        saved = {"depth": np.ascontiguousarray(output)}
        saved.update(light_attention_trace)
        saved.update(light_encoder_internal_trace)
        if decoder_trace_convs:
            saved.update(decoder_checkpoints)
        if not args.depth_only:
            saved.update(frontend_captures)
            saved.update({f"capture_l{layer:02d}": value for layer, value in
                          zip(plan["capture_layers"], captures)})
            saved.update({f"block_l{layer:02d}": value
                          for layer, value in zip(executed_layers, block_outputs)})
            saved.update({f"post_l{layer:02d}": value
                          for layer, value in zip(executed_layers, post_outputs)})
            for layer, (q, k, v) in zip(executed_layers, qkv_outputs):
                saved[f"q_l{layer:02d}"] = q
                saved[f"k_l{layer:02d}"] = k
                saved[f"v_l{layer:02d}"] = v
            saved.update({f"attention_l{layer:02d}": value
                          for layer, value in zip(executed_layers, attention_outputs)})
            saved.update({f"activated_l{layer:02d}": value
                          for layer, value in zip(executed_layers, activation_outputs)})
            saved.update(decoder_checkpoints)
        np.savez(args.output, **saved)

    def kernel_stage(name: str) -> str:
        if name.startswith("patch_projection"):
            return "frontend_patch_projection"
        if name.startswith("tail_norm"):
            return "encoder_layernorm"
        if name.startswith("qkv_projection"):
            return "encoder_qkv"
        if name.startswith("attention2"):
            return "encoder_attention"
        if name.startswith("post_attention"):
            return "encoder_post_attention"
        if name.startswith("mlp_fc1"):
            return "encoder_mlp_fc1"
        if name.startswith("mlp_fc2"):
            return "encoder_mlp_fc2"
        if name.startswith("decoder_"):
            return "decoder"
        return "other"

    latency_by_stage = {}
    for timing in timings:
        stage = kernel_stage(timing["kernel"])
        aggregate = latency_by_stage.setdefault(stage, {
            "calls": 0, "h2c_ms": 0.0, "npu_ms": 0.0, "c2h_ms": 0.0,
        })
        aggregate["calls"] += 1
        for key in ("h2c_ms", "npu_ms", "c2h_ms"):
            aggregate[key] += float(timing[key])
    codec_stats = codec_selection.stats()  # Enforces zero vendor calls in native mode.
    codec_stats.update(tensor_codec.reuse_stats())
    codec_pack_ms = codec_stats["codec_pack_ms_total"]
    codec_unpack_ms = codec_stats["codec_unpack_ms_total"]
    output_finite = bool(np.isfinite(output).all())
    output_sha256 = sha256_array(output)
    resident_bank_sha256 = sha256_array(linked)
    process_wall_ms = (time.perf_counter() - process_started) * 1000.0
    summary = {
        **codec_stats,
        "summary_schema_version": (15 if args.qkv_attention6_frame_graph else
                                   14 if args.attention_post_frame_graph else
                                   13 if args.encoder_quantize_pack else
                                   12 if args.encoder_fc_frame_graph else
                                   11 if args.decoder_native_boundary else
                                   10 if contract.get("encoder_fc1_dispatch_policy") else
                                   9 if args.decoder_fused_stems else
                                   8 if args.decoder_resident_captures else
                                   7 if args.encoder_resident_intermediates else 6),
        "host_executor": {
            "requested": host_selection.mode,
            "backend": host_executor.backend,
            "qualification_sha256": host_selection.report_sha256,
            "extension_sha256": host_selection.extension_sha256,
            "fallback_reason": host_selection.fallback_reason,
            **host_executor.stats(),
        },
        "output_shape": list(output.shape), "finite": output_finite,
        "output_sha256": output_sha256, "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": resident_bank_sha256, "static_h2c_write_count": 2,
        "static_reloads": 0, "load_ms": load_ms, "npu_calls": len(timings),
        "npu_ms_total": sum(x["npu_ms"] for x in timings),
        "h2c_ms_total": sum(x["h2c_ms"] for x in timings),
        "c2h_ms_total": sum(x["c2h_ms"] for x in timings),
        "codec_cfg_preparse_ms": 0.0 if cfg_registry_reused else cfg_registry.preparse_ms,
        "codec_cfg_layouts": len(cfg_registry.representatives),
        "codec_yaml_reused": yaml_reused,
        "codec_cfg_registry_reused": cfg_registry_reused,
        "manifest_cache_reused": manifest_reused,
        "contract_cache_reused": contract_reused,
        "host_plan_cache_reused": host_plan_reused,
        "host_params_cache_reused": host_params_reused,
        "bank_image_cache_reused": bank_image_reused,
        "codec_cfg_vendor_activations": (
            cfg_registry.activations - cfg_activations_at_start
        ),
        "codec_cfg_vendor_activation_ms": (
            cfg_registry.activation_ms - cfg_activation_ms_at_start
        ),
        "codec_pack_ms_total": codec_pack_ms,
        "codec_unpack_ms_total": codec_unpack_ms,
        "dma_runtime": args.dma_runtime,
        "cpp_runtime_reused": cpp_runtime_reused,
        "resident_bank_reused": resident_bank_reused,
        "submission_groups": len(submission_groups),
        "submission_group_dispatches": sum(
            int(item["submission_group_size"]) for item in submission_groups
        ),
        "attention_launch_group": args.attention_launch_group,
        "decoder_launch_group": args.decoder_launch_group,
        "encoder_fc1_launch_group": args.encoder_fc1_launch_group,
        "frontend_launch_group": args.frontend_launch_group,
        "encoder_fc_frame_graph": bool(args.encoder_fc_frame_graph),
        "c2h_exact_half_size": True,
        "latency_by_stage": latency_by_stage,
        "decoder_host_ops": decoder_host_ops,
        "attention_resident_kv": bool(args.attention_resident_kv),
        "attention_post_frame_graph": bool(args.attention_post_frame_graph),
        "fused_qkv_attention": bool(args.fused_qkv_attention),
        "fused_qkv_attention_layers": sorted(
            args.fused_qkv_attention_layer_set
        ),
        "fused_attention6": bool(args.fused_attention6),
        "fused_attention6_layers": sorted(args.fused_attention6_layer_set),
        "qkv_attention6_frame_graph": bool(args.qkv_attention6_frame_graph),
        "encoder_resident_intermediates": bool(args.encoder_resident_intermediates),
        "decoder_resident_captures": bool(args.decoder_resident_captures),
        "decoder_fused_stems": bool(args.decoder_fused_stems),
        "decoder_native_boundary": bool(args.decoder_native_boundary),
        "decoder_quantize_pack": bool(args.decoder_quantize_pack),
        "encoder_quantize_pack": bool(args.encoder_quantize_pack),
        "cpp_mixed_signature_groups": bool(args.cpp_mixed_signature_groups),
        "decoder_capture_offsets_units": capture_offsets,
        "collect_calibration": bool(args.collect_calibration),
        "encoder_resume": str(args.encoder_resume) if args.encoder_resume else None,
        "encoder_start_layer": args.encoder_start_layer,
        "capture_and_replace": {
            "trace": (str(args.replacement_trace.resolve())
                      if args.replacement_trace else None),
            "encoder_block": args.replace_encoder_block,
            "encoder_internal": (
                {"layer": args.replace_encoder_internal[0],
                 "stage": args.replace_encoder_internal[1]}
                if args.replace_encoder_internal is not None else None
            ),
            "encoder_attention_head": (
                {"layer": args.replace_encoder_attention_head[0],
                 "head": args.replace_encoder_attention_head[1]}
                if args.replace_encoder_attention_head is not None else None
            ),
            "decoder_conv": args.replace_decoder_conv,
        },
        "h2c_skipped_bytes": h2c_skipped_bytes,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "process_wall_ms": process_wall_ms,
        "hybrid_calibration": hybrid_calibration,
        "decoder_calibration": decoder_calibration,
    }
    if cpp_runtime is not None:
        summary["cpp_runtime"] = cpp_runtime.stats()
    accounted = (
        float(load_ms) + summary["h2c_ms_total"] + summary["npu_ms_total"]
        + summary["c2h_ms_total"] + codec_pack_ms + codec_unpack_ms
        + summary["codec_cfg_preparse_ms"]
        + summary["codec_cfg_vendor_activation_ms"]
        + sum(float(item["ms"]) for item in decoder_host_ops.values())
    )
    host_summary = host_profiler.summary(
        process_wall_ms=summary["process_wall_ms"],
        externally_accounted_ms=accounted,
    )
    summary["host_profile"] = host_summary["host_profile"]
    compatibility_residual = (
        host_summary["host_profile_ms_total"]
        + host_summary["unattributed_host_residual_ms"]
    )
    summary["latency_breakdown"] = {
        "resident_bank_load_ms": float(load_ms),
        "cfg_preparse_ms": summary["codec_cfg_preparse_ms"],
        "cfg_vendor_activation_ms": summary["codec_cfg_vendor_activation_ms"],
        "input_pack_ms": codec_pack_ms,
        "native_input_pack_ms": codec_stats["native_pack_ms"],
        "vendor_input_pack_ms": codec_stats["vendor_pack_ms"],
        "h2c_ms": summary["h2c_ms_total"],
        "npu_ms": summary["npu_ms_total"],
        "c2h_ms": summary["c2h_ms_total"],
        "output_unpack_ms": codec_unpack_ms,
        "native_output_unpack_ms": codec_stats["native_unpack_ms"],
        "vendor_output_unpack_ms": codec_stats["vendor_unpack_ms"],
        "decoder_host_ops_ms": sum(
            float(item["ms"]) for item in decoder_host_ops.values()
        ),
        "host_profile_ms_total": host_summary["host_profile_ms_total"],
        "unattributed_host_residual_ms": host_summary[
            "unattributed_host_residual_ms"
        ],
        "host_graph_and_python_residual_ms": compatibility_residual,
        "accounted_ms": accounted + host_summary["host_profile_ms_total"],
        "measurement_note": (
            "group H2C/C2H time is charged to the first physical dispatch; "
            "hardware NPU time remains per BIN"
        ),
    }
    if args.golden:
        loaded_golden = np.load(args.golden, allow_pickle=False)
        if isinstance(loaded_golden, np.ndarray):
            golden = loaded_golden
        else:
            with loaded_golden as archive:
                golden = archive[archive.files[0]]
        summary["metrics"] = tensor_metrics(output, golden)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if replacement_archive is not None:
        replacement_archive.close()
    print("HYBRID_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
