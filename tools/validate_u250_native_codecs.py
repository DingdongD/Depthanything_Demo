#!/usr/bin/env python3
"""CPU-only, fail-closed qualification against the official DS tensor codec.

The vendor pack API always reads cfg inputs and its unpack API reads outputs.
For the symmetric operation, a temporary cfg mirrors the selected tensor into
both lists. Each probe also checks the descriptor's actual direction using the
unmodified original cfg. No DMA extension instance or device is opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

import numpy as np

if __package__:
    from .run_u250_depthanything_hybrid import CfgCodecRegistry, quiet_native_stdout
else:
    from run_u250_depthanything_hybrid import CfgCodecRegistry, quiet_native_stdout


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def deterministic_tensor(shape, bitdepth, probe):
    """Coordinate-sensitive INT8 and finite BF16 rounding/tail probes."""
    shape = tuple(shape)
    if probe == "padding":
        return np.full(shape, -1, np.int8 if bitdepth == 8 else np.float32)
    if probe == "permutation":
        # Fixed SplitMix64 sequence avoids the 256-channel period inherent in
        # an INT8 affine coordinate pattern, without relying on RNG versions.
        bits = np.arange(np.prod(shape), dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        bits = (bits ^ (bits >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        bits = (bits ^ (bits >> 27)) * np.uint64(0x94D049BB133111EB)
        bits ^= bits >> 31
        if bitdepth == 8:
            return (bits >> 56).astype(np.uint8).view(np.int8).reshape(shape)
        return ((bits >> 40).astype(np.float32) / np.float32(65536) - 128).reshape(shape)
    n, c, y, x = np.ogrid[tuple(slice(0, dim) for dim in shape)]
    coordinate = n * 101 + c * 37 + y * 17 + x * 73
    if bitdepth == 8:
        return np.ascontiguousarray((coordinate % 256 - 128).astype(np.int8))
    if probe == "boundaries":
        bits = np.array([
            0x00000000, 0x80000000, 0x00000001, 0x00007FFF,
            0x00008000, 0x00008001, 0x007FFFFF, 0x00800000,
            0x3F807FFF, 0x3F808000, 0x3F808001, 0x3F818000,
            0xBF808000, 0xBF818000, 0x7F7F0000, 0xFF7F0000,
        ], np.uint32)
        return np.ascontiguousarray(bits[coordinate % len(bits)].view(np.float32))
    return np.ascontiguousarray((coordinate % 65521 - 32760).astype(np.float32) / np.float32(127))


def first_mismatch(actual, expected):
    """Compare dtype, full extent, and every bit, including float signed zero."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {"kind": "shape", "actual": list(actual.shape), "expected": list(expected.shape)}
    if actual.dtype != expected.dtype:
        return {"kind": "dtype", "actual": str(actual.dtype), "expected": str(expected.dtype)}
    a, b = np.ascontiguousarray(actual), np.ascontiguousarray(expected)
    changed = np.flatnonzero(a.reshape(-1).view(np.uint8) != b.reshape(-1).view(np.uint8))
    if changed.size == 0:
        return None
    index = int(changed[0]) // a.itemsize
    result = {"kind": "value", "flat_index": index,
              "coordinate": list(np.unravel_index(index, a.shape)),
              "actual": a.flat[index].item(), "expected": b.flat[index].item()}
    result["coordinate"] = [int(value) for value in result["coordinate"]]
    if a.dtype.kind == "f":
        result["actual_bits"] = a.flat[index].tobytes()[::-1].hex()
        result["expected_bits"] = b.flat[index].tobytes()[::-1].hex()
        # Keep reports valid JSON even if a broken codec produces NaN or Inf.
        for key in ("actual", "expected"):
            if not np.isfinite(result[key]):
                result[key] = str(result[key])
    return result


def split_banks(combined):
    data = np.ascontiguousarray(combined).reshape(-1).view(np.uint8)
    if data.size % 256:
        raise ValueError(f"vendor extent {data.size} is not 256-byte aligned")
    stripes = data.reshape(-1, 128)
    return tuple(np.ascontiguousarray(stripes[bank::2]).reshape(-1) for bank in (0, 1))


def merge_banks(even, odd):
    if even.dtype != np.uint8 or odd.dtype != np.uint8 or even.shape != odd.shape:
        raise ValueError("native bank dtype/shape mismatch")
    combined = np.empty(even.size + odd.size, np.uint8).reshape(-1, 128)
    combined[0::2] = even.reshape(-1, 128)
    combined[1::2] = odd.reshape(-1, 128)
    return combined.reshape(-1)


def export_logical(data, bitdepth, dims, name, *unused):
    if bitdepth == 8:
        # Vendor returns signed logical integer values, including negative INT8.
        value = np.asarray(data, np.int8).reshape(dims)
    elif bitdepth == 16:
        raw = np.asarray(data, np.int16).view(np.uint16).astype(np.uint32) << 16
        value = raw.view(np.float32).reshape(dims)
    else:
        raise ValueError(f"unsupported vendor bitdepth {bitdepth}")
    return {name: np.ascontiguousarray(value)}


def mirrored_cfg(source, descriptor):
    lines = source.splitlines()
    prefix = "Address:" if descriptor.direction == "input" else "Output Address:"
    selected = [line for line in lines if line.startswith(prefix)][descriptor.index]
    tensor = re.sub(r"^(Output )?Address: \d+ \([^)]*\)", "Address: 0 (0x0)", selected)
    if "ifmap_4ch_en:" not in tensor:
        tensor = tensor.replace(" c_align:", " ifmap_4ch_en: false c_align:")
    metadata = [line for line in lines if not line.startswith(("Address:", "Output Address:"))]
    return "\n".join([metadata[0], tensor, "Output " + tensor, *metadata[1:]]) + "\n"


def qualify_descriptor(codec, vendor, callback, records, cfg_dir, scratch, desc, users):
    identity = desc.identity()
    result = {"identity": identity, **asdict(desc), "users": users,
              "pack_exact": False, "unpack_exact": False, "native_exact": False,
              "native_pack_ms": 0.0, "vendor_pack_ms": 0.0, "probes": []}
    case_name = users[0]["case"]
    original = cfg_dir / f"{case_name}_cfg.txt"
    source = original.read_text()
    result["cfg_sha256"] = sha256(source.encode())
    mirror = scratch / identity
    mirror_text = mirrored_cfg(source, desc)
    mirror.with_name(mirror.name + "_cfg.txt").write_text(mirror_text)
    result["symmetric_cfg_sha256"] = sha256(mirror_text.encode())
    descriptor = asdict(desc)
    probes = ["coordinates", "permutation", "padding"] + (["boundaries"] if desc.bitdepth == 16 else [])
    try:
        codec.validate_descriptor(descriptor)
        for probe in probes:
            entry = {"name": probe, "mismatches": {}, "exact": False}
            # Retain findings immediately: a later codec exception must not
            # discard an already-detected mismatch from the returned artifact.
            result["probes"].append(entry)
            logical = deterministic_tensor(desc.dims, desc.bitdepth, probe)
            with quiet_native_stdout(True):
                vendor.read_cfg(str(mirror))
                started = time.perf_counter()
                packed = vendor.read_npz_dict(callback, "input", [{"input": logical}])
                vendor_ms = (time.perf_counter() - started) * 1000
            physical = np.ascontiguousarray(packed[0]).reshape(-1).view(np.uint8)
            if physical.size != desc.combined_bytes:
                raise ValueError(f"vendor physical extent {physical.size} != descriptor {desc.combined_bytes}")
            started = time.perf_counter()
            even, odd = codec.pack_tensor(logical, descriptor)
            native_ms = (time.perf_counter() - started) * 1000
            native_physical = merge_banks(even, odd)
            pack_error = first_mismatch(native_physical, physical)
            if pack_error:
                pack_error["bank"] = (pack_error.get("flat_index", 0) // 128) % 2
                entry["mismatches"]["pack"] = pack_error
            with quiet_native_stdout(True):
                restored = next(iter(vendor.buffer_to_npz_dict(export_logical, "output", [physical])[0].values()))
            native_restored = codec.unpack_tensor(*split_banks(physical), descriptor)
            unpack_error = first_mismatch(native_restored, restored)
            if unpack_error:
                entry["mismatches"]["unpack"] = unpack_error
            expected = logical
            if desc.bitdepth == 16:
                bits = logical.view(np.uint32)
                expected = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).view(np.float32)
            logical_error = first_mismatch(restored, expected)
            if logical_error:
                entry["mismatches"]["vendor_round_trip"] = logical_error
            # Cross-check the real cfg direction, not just the mirrored fixture.
            with quiet_native_stdout(True):
                vendor.read_cfg(str(original).removesuffix("_cfg.txt"))
                if desc.direction == "input":
                    arrays = [np.zeros(t["dims"], np.int8 if t["bitdepth"] == 8 else np.float32)
                              for t in records[case_name]["inputs"]]
                    arrays[desc.index] = logical
                    actual = vendor.read_npz_dict(callback, "input", [{"input": a} for a in arrays])[desc.index]
                    original_error = first_mismatch(np.ascontiguousarray(actual).reshape(-1).view(np.uint8), physical)
                else:
                    arrays = [np.zeros(t["size_per_bank"], np.uint8) for t in records[case_name]["outputs"]]
                    arrays[desc.index] = physical
                    actual = vendor.buffer_to_npz_dict(export_logical, "output", arrays)[desc.index]
                    original_error = first_mismatch(next(iter(actual.values())), restored)
            if original_error:
                entry["mismatches"]["original_cfg_direction"] = original_error
            if probe == "padding":
                # -1 has no zero bytes in either INT8 or BF16. Thus vendor zeros
                # identify only physical padding, without consulting native indices.
                padding = physical == 0
                poisoned = physical.copy()
                poisoned[padding] = 0x3F
                with quiet_native_stdout(True):
                    vendor.read_cfg(str(mirror))
                    poison_expected = next(iter(vendor.buffer_to_npz_dict(export_logical, "output", [poisoned])[0].values()))
                poison_actual = codec.unpack_tensor(*split_banks(poisoned), descriptor)
                for label, actual in (("native_padding", poison_actual), ("vendor_padding", poison_expected)):
                    error = first_mismatch(actual, expected)
                    if error:
                        entry["mismatches"][label] = error
                entry["padding_bytes"] = int(padding.sum())
                entry["padding_zero"] = bool(np.all(native_physical[padding] == 0))
                if not entry["padding_zero"]:
                    entry["mismatches"]["padding_zero"] = first_mismatch(native_physical[padding], physical[padding])
            entry.update(native_pack_ms=native_ms, vendor_pack_ms=vendor_ms,
                         physical_bytes=int(physical.size), logical_bytes=logical.nbytes,
                         native_sha256=sha256(native_physical.tobytes()),
                         vendor_sha256=sha256(physical.tobytes()),
                         logical_sha256=sha256(native_restored.tobytes()),
                         vendor_logical_sha256=sha256(restored.tobytes()))
            entry["exact"] = not entry["mismatches"]
            result["native_pack_ms"] += native_ms
            result["vendor_pack_ms"] += vendor_ms
        # No descriptor is qualified if any probe, original cfg, or padding check fails.
        all_exact = len(result["probes"]) == len(probes) and all(p["exact"] for p in result["probes"])
        result.update(pack_exact=all_exact, unpack_exact=all_exact, native_exact=all_exact)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cfg-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True, help="r43 package containing npz_util.py")
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    # Imports from the existing package must not create __pycache__ there.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.case_dir.resolve()), str(args.runtime_dir.resolve())]
    import npz2bin
    from npz_util import createBF16TensorFromDict
    spec = importlib.util.spec_from_file_location("fpgaDmaBatch", args.extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension {args.extension}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    manifest_bytes = args.manifest.read_bytes()
    records = {record["name"]: record for record in json.loads(manifest_bytes)["cases"]}
    registry = CfgCodecRegistry(args.cfg_dir, records, npz2bin, quiet=True)
    unique = {}
    for name, directions in registry.descriptors.items():
        for direction, descriptors in directions.items():
            for desc in descriptors:
                if desc.layout != "NCHW":
                    continue
                item = unique.setdefault(desc.identity(), {"descriptor": desc, "users": []})
                item["users"].append({"case": name, "direction": direction, "index": desc.index})
    if not unique:
        raise ValueError("manifest contains no NCHW descriptors")
    yaml_paths = [args.runtime_dir / f"arch_{arch}_mono.yaml" for arch in (16, 256)]
    with quiet_native_stdout(True):
        npz2bin.read_yaml([str(path) for path in yaml_paths])
    report = {"manifest_sha256": sha256(manifest_bytes), "layout": "NCHW",
              "host": platform.node(), "python": sys.version, "python_executable": sys.executable,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "extension_sha256": sha256(args.extension.read_bytes()),
              "native_source_sha256": sha256(Path(__file__).with_name("fpga_dma_batch.cpp").read_bytes()),
              "validator_sha256": sha256(Path(__file__).read_bytes()),
              "vendor_sha256": sha256(Path(npz2bin.__file__).read_bytes()),
              "yaml_sha256": {p.name: sha256(p.read_bytes()) for p in yaml_paths},
              "oracle_method": "mirrored symmetric cfg plus unmodified cfg direction; full physical bytes and logical bits",
              "descriptors": []}
    with tempfile.TemporaryDirectory(prefix="nchw-oracle-", dir=args.work_dir) as scratch:
        for identity, item in sorted(unique.items()):
            result = qualify_descriptor(extension.DmaBatch, npz2bin, createBF16TensorFromDict,
                                        records, args.cfg_dir, Path(scratch),
                                        item["descriptor"], item["users"])
            report["descriptors"].append(result)
            print(f"{identity[:12]} {result['direction']} {result['dims']} b{result['bitdepth']}: "
                  f"{'exact' if result['native_exact'] else 'REJECTED'}", flush=True)
    report["qualified"] = all(d["native_exact"] for d in report["descriptors"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
