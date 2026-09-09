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
_NPZ_YAML_PATHS: tuple[str, str] | None = None


def encoder_resident_offset_plan(records: dict[str, dict], block: dict,
                                 workspace_bytes_per_bank: int,
                                 x_begin: int | None = None) -> dict[str, int]:
    """Derive and validate a relocated encoder residual/post/norm2 plan."""
    norm = records[block["host_norm1"]["npu_core"]]
    post = records[block["post_attention"]["kernel"]]
    low_end = max(
        mapped_runtime.record_span_per_bank(records[block["qkv"]["kernel"]]),
        *(mapped_runtime.record_span_per_bank(records[head["kernel"]])
          for head in block["attention"]["heads"]),
    ) // mapped_runtime.ADDRESS_UNIT_BYTES_PER_BANK
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
        cfg_sources = {name: (cfg_dir / f"{name}_cfg.txt").read_text() for name in records}
        if cached.input_fingerprint != _cfg_registry_fingerprint(records, cfg_sources):
            raise RuntimeError("cfg registry tensor metadata or cfg contents changed inside resident process")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-only", action="store_true",
                        help="save only the final depth tensor, not intermediate traces")
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
        "--verbose-vendor-codec", action="store_true",
        help="retain npz2bin's very verbose cfg/pack/unpack stdout",
    )
    parser.add_argument(
        "--attention-resident-kv", action="store_true",
        help="upload K/V once per head and retain them at the shared attention IO addresses",
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
    args = parser.parse_args()
    if (args.encoder_resume is None) != (args.encoder_start_layer is None):
        parser.error("--encoder-resume and --encoder-start-layer must be used together")
    if args.encoder_captures is not None and args.encoder_resume is not None:
        parser.error("--encoder-captures and --encoder-resume are mutually exclusive")
    if args.encoder_start_layer is not None and not 1 <= args.encoder_start_layer <= 11:
        parser.error("--encoder-start-layer must be in [1, 11]")
    if args.attention_launch_group < 1 or args.decoder_launch_group < 1:
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
        if (not args.decoder_resident_captures
                or args.layout_codec != "native"
                or args.dma_runtime != "cpp_mapped"
                or args.host_executor != "cpp"):
            parser.error(
                "--decoder-native-boundary requires resident captures, native "
                "codec, cpp_mapped DMA, and the C++ host executor"
            )
        if args.decoder_fused_stems:
            parser.error(
                "--decoder-native-boundary and --decoder-fused-stems are mutually exclusive"
            )
    process_started = time.perf_counter()
    host_profiler = HostProfiler()

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
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
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
    env = {key: np.ascontiguousarray(value)
           for key, value in np.load(args.host_params, allow_pickle=False).items()}
    loaded_input = np.load(args.input, allow_pickle=False)
    if isinstance(loaded_input, np.ndarray):
        x = np.ascontiguousarray(loaded_input, dtype=np.float32)
    else:
        with loaded_input as archive:
            x = np.ascontiguousarray(archive["input"], dtype=np.float32)

    bank_path = case_dir / manifest["bank_file"]
    linked = np.fromfile(bank_path, dtype=np.uint8)
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
        if len(signatures) != 1:
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
            buckets.setdefault(cfg_registry.signature(name), []).append(index)
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
                connections, args.timeout_ms,
            )
        )
        h2c_skipped_bytes += int(group["h2c_skipped_bytes"])
        submission_groups.append({
            "kind": "cpp_mapped_resident", "kernels": list(names),
            **{key: value for key, value in group.items() if key != "npu_ms"},
        })
        decoded = []
        for name, outputs in zip(names, physical):
            if any(value is None for value in outputs):
                raise RuntimeError(f"{name}: runner requires downloaded resident outputs")
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
                    projection_input = host_executor.quantize(
                        patchified, projection["input_quantization"]["scale"])
            reusable_projection_input = tensor_codec.reusable(projection_input)
            projection_outputs = [
                run_kernel(name, [reusable_projection_input])[0]
                for name in frontend["projection_kernels"]
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
            executed_layers.append(int(block["layer"]))
            norm1 = norm_specs["norm1"]
            norm1_contract = block["host_norm1"]
            resident_offsets = None
            resident_x_handle = None
            if "npu_core" in norm1_contract:
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
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/norm1/LayerNormalization_output_0"] = calibration_stats(normalized)
            code = profiled_quantize(
                "encoder.quantize_qkv", normalized,
                block["qkv"]["input_quantization"]["scale"],
            )
            q, k, v = run_kernel(block["qkv"]["kernel"], [code[:, None]])
            if not args.depth_only:
                qkv_outputs.append((q.copy(), k.copy(), v.copy()))
            post_name = block["post_attention"]["kernel"]
            attention_fusion = (
                args.depth_only and not args.collect_calibration
                and cpp_runtime is not None and host_executor.backend == "cpp"
                and codec_selection.native_for(post_name, "input")
                and all(codec_selection.native_for(head["kernel"], "output")
                        for head in block["attention"]["heads"])
            )
            head_outputs = []
            attention_physical = []
            attention_source_descriptors = []
            attention_valid_widths = []
            for head in block["attention"]["heads"]:
                begin = head["head"] * 64; end = begin + 64
                qh = q[0, 0, :, begin:end]; kh = k[0, 0, :, begin:end]
                vh = v[0, 0, :, begin:end]
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
                grouped = run_compatible_groups(
                    [head["kernel"]] * len(call_inputs), call_inputs,
                    args.attention_launch_group, call_masks,
                    decode_outputs_flag=not attention_fusion,
                )
                if attention_fusion:
                    descriptors = cfg_registry.descriptors[head["kernel"]]["output"]
                    for outputs, q0_length, q1_length in zip(
                        grouped, q0_lengths, q1_lengths
                    ):
                        attention_physical.extend(outputs)
                        attention_source_descriptors.extend(descriptors)
                        attention_valid_widths.extend([q0_length, q1_length])
                else:
                    grouped_elements = sum(
                        int(value.size) for outputs in grouped for value in outputs
                    )
                    grouped_bytes = sum(
                        int(value.nbytes) for outputs in grouped for value in outputs
                    )
                    with host_profiler.measure(
                        "encoder.attention_output_assembly",
                        elements=grouped_elements, nbytes=grouped_bytes,
                    ):
                        chunks = []
                        for (out0, out1), q1_length in zip(grouped, q1_lengths):
                            chunks.extend([
                                np.ascontiguousarray(out0),
                                np.ascontiguousarray(out1[:, :, :q1_length, :]),
                            ])
                        head_outputs.append(
                            host_executor.concatenate(chunks, axis=2)
                        )
            if attention_fusion:
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
                with host_profiler.measure(
                    "encoder.attention_output_assembly",
                    elements=sum(int(value.size) for value in head_outputs),
                    nbytes=sum(int(value.nbytes) for value in head_outputs),
                ):
                    attention = host_executor.concatenate(head_outputs, axis=3)
            if not args.depth_only:
                attention_outputs.append(attention.copy())
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/attn/Concat_6_output_0"] = calibration_stats(attention)
            if not attention_fusion:
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
                resident_values, _, _ = run_device_chain(
                    [post_name, norm2_contract["npu_core"]],
                    [[post_code, resident_x_handle], [None]],
                    [resident_offsets["post"], resident_offsets["norm2"]],
                    {(1, 0): (0, 0)},
                )
                post = resident_values[0][0][:, 0]
                core = resident_values[1][0][:, 0]
            else:
                post = run_kernel(
                    post_name, [post_code, x[:, None]]
                )[0][:, 0]
                core = (run_kernel(
                    norm2_contract["npu_core"], [post[:, None]]
                )[0][:, 0] if "npu_core" in norm2_contract else None)
            if not args.depth_only:
                post_outputs.append(post.copy())
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
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/norm2/LayerNormalization_output_0"] = calibration_stats(normalized)
            fc1_code = profiled_quantize(
                "encoder.quantize_fc1", normalized,
                block["mlp"]["fc1_input_quantization"]["scale"],
            )
            native_gelu = block["mlp"].get("npu_activation")
            if native_gelu is None:
                reusable_fc1_code = tensor_codec.reusable(fc1_code[:, None])
                fc2_scale = block["mlp"]["fc2_input_quantization"]["scale"]
                fc1_names = block["mlp"]["fc1_kernels"]
                fc2_name = block["mlp"]["fc2_kernel"]
                physical_fusion = (
                    args.depth_only and not args.collect_calibration
                    and cpp_runtime is not None and host_executor.backend == "cpp"
                    and all(codec_selection.native_for(name, "output")
                            for name in fc1_names)
                    and codec_selection.native_for(fc2_name, "input")
                )
                if physical_fusion:
                    fc1_calls = [
                        run_kernel_physical(name, [reusable_fc1_code])
                        for name in fc1_names
                    ]
                    fc1_physical = [
                        output for outputs in fc1_calls for output in outputs
                    ]
                    source_descriptors = [
                        descriptor for name in fc1_names
                        for descriptor in cfg_registry.descriptors[name]["output"]
                    ]
                    if len(fc1_physical) != len(source_descriptors):
                        raise RuntimeError("FC1 physical outputs do not match cfg descriptors")
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
                        for output in run_kernel(name, [reusable_fc1_code])
                    ]
                    with host_profiler.measure(
                        "encoder.mlp_assembly",
                        elements=sum(int(value.size) for value in fc1_outputs),
                        nbytes=sum(int(value.nbytes) for value in fc1_outputs),
                    ):
                        hidden = host_executor.concatenate(fc1_outputs, axis=3)
                    with host_profiler.measure(
                        "encoder.gelu_quantize",
                        elements=int(hidden.size), nbytes=int(hidden.nbytes),
                    ):
                        fc2_input = host_executor.gelu_quantize(hidden, fc2_scale)
                    if args.depth_only and not args.collect_calibration:
                        activated = None
                    else:
                        with host_profiler.measure(
                            "encoder.gelu",
                            elements=int(hidden.size), nbytes=int(hidden.nbytes),
                        ):
                            activated = gelu(hidden)
            else:
                conv_input = np.ascontiguousarray(
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
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/mlp/act/Mul_1_output_0"] = calibration_stats(activated)
            if not args.depth_only:
                activation_outputs.append(activated.copy())
            fc2 = run_kernel(block["mlp"]["fc2_kernel"], [fc2_input])[0][:, 0]
            with host_profiler.measure(
                "encoder.residual",
                elements=int(post.size), nbytes=int(post.nbytes + fc2.nbytes),
            ):
                x = host_executor.add(post, fc2)
            if not args.depth_only:
                block_outputs.append(x.copy())
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
                if project < 3 and layer not in resident_capture_handles:
                    raise RuntimeError(
                        f"decoder capture layer {layer} was not retained"
                    )
                norm_name = contract["encoder"][layer]["host_norm1"]["npu_core"]
                source_descriptor = replace(
                    cfg_registry.descriptors[norm_name]["output"][0],
                    direction="output", index=0, matrix_role="output",
                )
                if layer in resident_capture_handles:
                    source = resident_capture_handles[layer]
                else:
                    # The final capture is materialized on the host because no
                    # following block needs a resident norm1 tensor.  Pack it
                    # through tail_norm1's qualified input route; input/output
                    # directions share the same physical storage ABI, while
                    # production qualification deliberately restricts pack to
                    # input descriptors and unpack to output descriptors.
                    source_pack_descriptor = cfg_registry.descriptors[
                        norm_name
                    ]["input"][0]
                    if (source_pack_descriptor.storage_identity()
                            != source_descriptor.storage_identity()):
                        raise RuntimeError(
                            "decoder final capture input/output storage ABI mismatch"
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
            code = profiled_quantize(
                "decoder.quantize", value, float(step["input_scale"])
            )
            if step.get("channel_sliced"):
                with host_profiler.measure(
                    "decoder.tile_assembly",
                    elements=int(code.size), nbytes=int(code.nbytes),
                ):
                    names = [item["name"] for item in step["kernels"]]
                    reusable_code = tensor_codec.reusable(code)
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
                    elements=int(code.size), nbytes=int(code.nbytes),
                ):
                    rows = int(step["tile_output_rows"])
                    count = int(step["row_tiles"])
                    is_3x3 = len(step["kernels"]) == 3
                    names = []
                    tile_inputs = []
                    for tile in range(count):
                        begin = tile * rows; end = begin + rows
                        if not is_3x3:
                            name = step["kernels"][0]["name"]
                            tile_input = code[:, :, begin:end]
                        elif tile == 0:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "first"
                            )
                            tile_input = code[:, :, :end + 1]
                        elif tile == count - 1:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "last"
                            )
                            tile_input = code[:, :, begin - 1:end]
                        else:
                            name = next(
                                item["name"] for item in step["kernels"]
                                if item["position"] == "middle"
                            )
                            tile_input = code[:, :, begin - 1:end + 1]
                        names.append(name)
                        tile_inputs.append([tile_input])
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
            env[step["outputs"][0]] = output
            if (not args.depth_only and int(step["index"])
                    in (0, 1, 7, 10, 13, 18, 23, 28, 29, 30, 31)):
                decoder_checkpoints[f"decoder_conv_{int(step['index']):02d}"] = output.copy()
            if not args.depth_only and int(step["index"]) == 7:
                decoder_checkpoints["decoder_input_07"] = value.copy()
        output = env[plan["model_outputs"][0]]
    finally:
        if waiter is not None:
            waiter.close()

    with host_profiler.measure(
        "result.serialize", elements=int(output.size), nbytes=int(output.nbytes)
    ):
        saved = {"depth": np.ascontiguousarray(output)}
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
        "summary_schema_version": (11 if args.decoder_native_boundary else
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
        "c2h_exact_half_size": True,
        "latency_by_stage": latency_by_stage,
        "decoder_host_ops": decoder_host_ops,
        "attention_resident_kv": bool(args.attention_resident_kv),
        "encoder_resident_intermediates": bool(args.encoder_resident_intermediates),
        "decoder_resident_captures": bool(args.decoder_resident_captures),
        "decoder_fused_stems": bool(args.decoder_fused_stems),
        "decoder_native_boundary": bool(args.decoder_native_boundary),
        "decoder_capture_offsets_units": capture_offsets,
        "collect_calibration": bool(args.collect_calibration),
        "encoder_resume": str(args.encoder_resume) if args.encoder_resume else None,
        "encoder_start_layer": args.encoder_start_layer,
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
    print("HYBRID_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
