#!/usr/bin/env python3
"""Execute the Depth Anything V2 ViT-S token tail with one resident U250 bank."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from u250_cpp_mapped_runtime import get_cached_cpp_runtime


_CFG_REGISTRY_CACHE: dict[str, "CfgCodecRegistry"] = {}
_NPZ_YAML_PATHS: tuple[str, str] | None = None


def codec_signature(path: Path) -> tuple[str, ...]:
    """Return the tensor-layout portion of a cfg, independent of addresses."""
    entries = []
    for line in path.read_text().splitlines():
        if line.startswith("Address: ") or line.startswith("Output Address: "):
            normalized = re.sub(r"Address: \d+ \([^)]*\)", "Address: <relocated>", line)
            entries.append(" ".join(normalized.split()))
    if not entries:
        raise ValueError(f"{path}: no tensor layout entries")
    return tuple(entries)


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
        for name in records:
            path = cfg_dir / f"{name}_cfg.txt"
            signature = codec_signature(path)
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
        return cached, True
    registry = CfgCodecRegistry(cfg_dir, records, npz2bin, quiet)
    _CFG_REGISTRY_CACHE[key] = registry
    return registry, False


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
                 timing: dict[str, dict] | None = None) -> None:
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
        result = (values[0] + values[1]).astype(np.float32)
    elif op == "Shape":
        result = np.asarray(values[0].shape, dtype=np.int64)
    elif op == "Concat":
        result = np.concatenate(values, axis=int(attrs["axis"]))
    elif op == "Resize":
        if attrs.get("mode", "nearest") != "linear" or attrs.get(
                "coordinate_transformation_mode", "half_pixel") != "align_corners":
            raise ValueError(f"unsupported Resize attributes: {attrs}")
        sizes = values[3] if len(values) > 3 and values[3] is not None else np.rint(
            np.asarray(values[0].shape) * np.asarray(values[2])).astype(np.int64)
        result = resize_align_corners(values[0], sizes)
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
        "--fpga-dma-batch", type=Path,
        help="fpgaDmaBatch extension file or directory (required if not importable)",
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
    args = parser.parse_args()
    if (args.encoder_resume is None) != (args.encoder_start_layer is None):
        parser.error("--encoder-resume and --encoder-start-layer must be used together")
    if args.encoder_captures is not None and args.encoder_resume is not None:
        parser.error("--encoder-captures and --encoder-resume are mutually exclusive")
    if args.encoder_start_layer is not None and not 1 <= args.encoder_start_layer <= 11:
        parser.error("--encoder-start-layer must be in [1, 11]")
    if args.attention_launch_group < 1 or args.decoder_launch_group < 1:
        parser.error("launch group sizes must be positive")

    process_started = time.perf_counter()

    case_dir = args.case_dir.resolve(); runtime_dir = args.runtime_dir.resolve()
    sys.path.insert(0, str(case_dir)); sys.path.insert(1, str(runtime_dir))
    import fpgaDma  # type: ignore
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
    manifest = json.loads(args.manifest.read_text())
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
    for name, step in plan_decoder.items():
        expected = contract_decoder[name]
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
        )
        load_ms, resident_bank_reused = cpp_runtime.ensure_bank(
            linked, str(manifest["bank_sha256"])
        )
        cpp_runtime.reset_frame_stats()
        waiter = None
    else:
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
    codec_pack_ms = 0.0
    codec_unpack_ms = 0.0
    submission_groups = []
    decoder_host_ops = {}

    def pack_inputs(name: str, logical_inputs: list[np.ndarray]) -> list[np.ndarray]:
        nonlocal codec_pack_ms
        cfg_registry.activate(name)
        pack_started = time.perf_counter()
        with quiet_native_stdout(not args.verbose_vendor_codec):
            packed_values = npz2bin.read_npz_dict(
                createBF16TensorFromDict, "input",
                [{"input": np.ascontiguousarray(value)} for value in logical_inputs],
            )
        codec_pack_ms += (time.perf_counter() - pack_started) * 1000.0
        packed = [np.ascontiguousarray(value).reshape(-1).view(np.uint8)
                  for value in packed_values]
        expected = [int(item["size_per_bank"]) for item in records[name]["inputs"]]
        actual = [int(value.size) for value in packed]
        if actual != expected:
            raise ValueError(f"{name}: packed byte sizes {actual} != cfg {expected}")
        return packed

    def decode_outputs(name: str, physical: list[np.ndarray]) -> list[np.ndarray]:
        nonlocal codec_unpack_ms
        cfg_registry.activate(name)
        unpack_started = time.perf_counter()
        with quiet_native_stdout(not args.verbose_vendor_codec):
            decoded = npz2bin.buffer_to_npz_dict(
                export_npz_allow_nonfinite, "output", physical)
        codec_unpack_ms += (time.perf_counter() - unpack_started) * 1000.0
        return [np.ascontiguousarray(item[preferred_output_key(item)])
                for item in decoded]

    def run_kernel_group(
        names: list[str], logical_calls: list[list[np.ndarray]],
        upload_masks: list[list[bool] | None] | None = None,
    ) -> list[list[np.ndarray]]:
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
            results = [decode_outputs(name, physical)
                       for name, physical in zip(names, physical_calls)]
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
        names: list[str], logical_calls: list[list[np.ndarray]], max_group: int,
        upload_masks: list[list[bool] | None] | None = None,
    ) -> list[list[np.ndarray]]:
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
                )
                for index, value in zip(chosen, values):
                    outputs[index] = value
                cursor = limit
        if any(value is None for value in outputs):
            raise RuntimeError("grouped execution did not produce every output")
        return list(outputs)  # type: ignore[arg-type]

    def run_kernel(name: str, logical_inputs: list[np.ndarray],
                   upload_mask: list[bool] | None = None) -> list[np.ndarray]:
        return run_kernel_group([name], [logical_inputs], [upload_mask])[0]

    try:
        captures = []
        block_outputs = []
        post_outputs = []
        qkv_outputs = []
        attention_outputs = []
        activation_outputs = []
        executed_layers = []
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
            patchified = x.reshape(
                n, channels, grid_h, patch, grid_w, patch
            ).transpose(0, 3, 5, 1, 2, 4).reshape(
                n, channels * patch * patch, grid_h, grid_w
            )
            projection = contract["frontend"]["patch_projection"]
            projection_input = patchified
            if projection["input_dtype"] == "INT8":
                projection_input = quantize(
                    patchified, projection["input_quantization"]["scale"])
            projected = np.concatenate([
                run_kernel(name, [projection_input])[0]
                for name in frontend["projection_kernels"]
            ], axis=1)
            patch_tokens = projected.transpose(0, 2, 3, 1).reshape(
                n, grid_h * grid_w, projected.shape[1]
            )
            cls_token = np.broadcast_to(
                env[frontend["cls_token"]], (n, 1, projected.shape[1])
            )
            x = np.ascontiguousarray(
                np.concatenate([cls_token, patch_tokens], axis=1)
                + env[frontend["pos_embed"]], dtype=np.float32
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
            if "npu_core" in norm1_contract:
                core = run_kernel(
                    norm1_contract["npu_core"], [x[:, None]]
                )[0][:, 0]
                if norm1_contract.get("affine") == "folded":
                    normalized = np.ascontiguousarray(core, dtype=np.float32)
                else:
                    normalized = (core * env[norm1["scale"]]
                                  + env[norm1["bias"]]).astype(np.float32)
            else:
                normalized = layer_norm(
                    x, env[norm1["scale"]], env[norm1["bias"]],
                    norm1["axis"], norm1["epsilon"]
                )
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/norm1/LayerNormalization_output_0"] = calibration_stats(normalized)
            code = quantize(normalized, block["qkv"]["input_quantization"]["scale"])
            q, k, v = run_kernel(block["qkv"]["kernel"], [code[:, None]])
            if not args.depth_only:
                qkv_outputs.append((q.copy(), k.copy(), v.copy()))
            head_outputs = []
            for head in block["attention"]["heads"]:
                begin = head["head"] * 64; end = begin + 64
                qh = q[0, 0, :, begin:end]; kh = k[0, 0, :, begin:end]
                vh = v[0, 0, :, begin:end]
                call_inputs = []
                call_masks = []
                q1_lengths = []
                for call_index, call in enumerate(head["calls"]):
                    q0 = qh[call["q0_rows"][0]:call["q0_rows"][1]]
                    q1 = np.zeros((256, 64), np.int8)
                    q1_values = qh[call["q1_rows"][0]:call["q1_rows"][1]]
                    q1[:q1_values.shape[0]] = q1_values
                    call_inputs.append([
                        q0[None, None], kh.T[None, None], vh[None, None], q1[None, None]
                    ])
                    call_masks.append(
                        [True, call_index == 0, call_index == 0, True]
                        if args.attention_resident_kv else None
                    )
                    q1_lengths.append(q1_values.shape[0])
                grouped = run_compatible_groups(
                    [head["kernel"]] * len(call_inputs), call_inputs,
                    args.attention_launch_group, call_masks,
                )
                chunks = []
                for (out0, out1), q1_length in zip(grouped, q1_lengths):
                    chunks.extend([out0, out1[:, :, :q1_length, :]])
                head_outputs.append(np.concatenate(chunks, axis=2))
            attention = np.concatenate(head_outputs, axis=3)
            if not args.depth_only:
                attention_outputs.append(attention.copy())
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/attn/Concat_6_output_0"] = calibration_stats(attention)
            post = run_kernel(block["post_attention"]["kernel"], [
                quantize(attention, block["post_attention"]["input_quantization"]["scale"]),
                x[:, None],
            ])[0][:, 0]
            if not args.depth_only:
                post_outputs.append(post.copy())
            norm2 = norm_specs["norm2"]
            norm2_contract = block["host_norm2"]
            if "npu_core" in norm2_contract:
                core = run_kernel(
                    norm2_contract["npu_core"], [post[:, None]]
                )[0][:, 0]
                if norm2_contract.get("affine") == "folded":
                    normalized = np.ascontiguousarray(core, dtype=np.float32)
                else:
                    normalized = (core * env[norm2["scale"]]
                                  + env[norm2["bias"]]).astype(np.float32)
            else:
                normalized = layer_norm(
                    post, env[norm2["scale"]], env[norm2["bias"]],
                    norm2["axis"], norm2["epsilon"]
                )
            if args.collect_calibration:
                hybrid_calibration[f"/blocks.{block['layer']}/norm2/LayerNormalization_output_0"] = calibration_stats(normalized)
            fc1_code = quantize(normalized, block["mlp"]["fc1_input_quantization"]["scale"])
            native_gelu = block["mlp"].get("npu_activation")
            if native_gelu is None:
                hidden = np.concatenate([
                    run_kernel(name, [fc1_code[:, None]])[0]
                    for name in block["mlp"]["fc1_kernels"]
                ], axis=3)
                activated = gelu(hidden)
                fc2_input = quantize(
                    activated, block["mlp"]["fc2_input_quantization"]["scale"]
                )
            else:
                conv_input = np.ascontiguousarray(
                    fc1_code.transpose(0, 2, 1)[:, :, None, :]
                )
                activated_code = np.concatenate([
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
            x = (post + fc2).astype(np.float32)
            if not args.depth_only:
                block_outputs.append(x.copy())
            if block["capture_for_decoder"]:
                captures.append(x.copy())
        for name, value in zip(plan["capture_tensor_names"], captures):
            env[name] = value

        for step in plan["decoder_steps"]:
            if step["backend"] == "host":
                lowered_norm = contract_decoder_layernorm.get(step["name"])
                if lowered_norm is not None:
                    value = env[step["inputs"][0]]
                    core = run_kernel(
                        lowered_norm["kernel"], [value[:, None]]
                    )[0][:, 0]
                    env[step["outputs"][0]] = np.ascontiguousarray(
                        core * env[step["inputs"][1]]
                        + env[step["inputs"][2]], dtype=np.float32
                    )
                    continue
                execute_host(step, env, decoder_host_ops)
                continue
            value = env[step["inputs"][0]]
            if args.collect_calibration:
                decoder_calibration[step["name"]] = calibration_stats(value)
            code = quantize(value, float(step["input_scale"]))
            if step.get("channel_sliced"):
                names = [item["name"] for item in step["kernels"]]
                grouped = run_compatible_groups(
                    names, [[code] for _ in names], args.decoder_launch_group
                )
                output = np.concatenate([item[0] for item in grouped], axis=1)
            elif int(step["row_tiles"]) == 1:
                output = run_kernel(step["kernels"][0]["name"], [code])[0]
            else:
                rows = int(step["tile_output_rows"]); count = int(step["row_tiles"])
                is_3x3 = len(step["kernels"]) == 3
                names = []
                tile_inputs = []
                for tile in range(count):
                    begin = tile * rows; end = begin + rows
                    if not is_3x3:
                        name = step["kernels"][0]["name"]; tile_input = code[:, :, begin:end]
                    elif tile == 0:
                        name = next(x["name"] for x in step["kernels"] if x["position"] == "first")
                        tile_input = code[:, :, :end + 1]
                    elif tile == count - 1:
                        name = next(x["name"] for x in step["kernels"] if x["position"] == "last")
                        tile_input = code[:, :, begin - 1:end]
                    else:
                        name = next(x["name"] for x in step["kernels"] if x["position"] == "middle")
                        tile_input = code[:, :, begin - 1:end + 1]
                    names.append(name)
                    tile_inputs.append([tile_input])
                grouped = run_compatible_groups(
                    names, tile_inputs, args.decoder_launch_group
                )
                parts = [item[0] for item in grouped]
                output = np.concatenate(parts, axis=2)
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
            return "encoder_mlp_fc1_gelu"
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
    summary = {
        "output_shape": list(output.shape), "finite": bool(np.isfinite(output).all()),
        "output_sha256": sha256_array(output), "resident_bank_bytes": int(linked.size),
        "resident_bank_sha256": sha256_array(linked), "static_h2c_write_count": 2,
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
        "collect_calibration": bool(args.collect_calibration),
        "encoder_resume": str(args.encoder_resume) if args.encoder_resume else None,
        "encoder_start_layer": args.encoder_start_layer,
        "h2c_skipped_bytes": h2c_skipped_bytes,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "process_wall_ms": (time.perf_counter() - process_started) * 1000.0,
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
    summary["latency_breakdown"] = {
        "resident_bank_load_ms": float(load_ms),
        "cfg_preparse_ms": summary["codec_cfg_preparse_ms"],
        "cfg_vendor_activation_ms": summary["codec_cfg_vendor_activation_ms"],
        "input_pack_ms": codec_pack_ms,
        "h2c_ms": summary["h2c_ms_total"],
        "npu_ms": summary["npu_ms_total"],
        "c2h_ms": summary["c2h_ms_total"],
        "output_unpack_ms": codec_unpack_ms,
        "decoder_host_ops_ms": sum(
            float(item["ms"]) for item in decoder_host_ops.values()
        ),
        "host_graph_and_python_residual_ms": max(
            0.0, summary["process_wall_ms"] - accounted
        ),
        "accounted_ms": accounted,
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
