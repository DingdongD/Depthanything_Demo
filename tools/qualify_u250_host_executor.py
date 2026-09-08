#!/usr/bin/env python3
"""Generate deterministic, fail-closed evidence for the C++ host executor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .u250_host_executor import (OPERATIONS, PHYSICAL_FUSIONS,
                                     PythonHostExecutor, _sha256_file)
    from .u250_cpp_mapped_runtime import load_fpga_dma_batch, resolve_fpga_dma_batch
except ImportError:
    from u250_host_executor import (OPERATIONS, PHYSICAL_FUSIONS,
                                    PythonHostExecutor, _sha256_file)
    from u250_cpp_mapped_runtime import load_fpga_dma_batch, resolve_fpga_dma_batch


_SCALES = (0.00390625, 0.03993530943989754, 0.10580708831548691)


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite_bf16_domain() -> np.ndarray:
    bits = np.arange(65536, dtype=np.uint32) << np.uint32(16)
    values = bits.view(np.float32)
    return np.ascontiguousarray(values[np.isfinite(values)]).reshape(1, -1)


def _vectors(trace_path: Path | None) -> list[np.ndarray]:
    vectors = [np.array(
        [-128.0, -32.0, -16.0, -14.0, -12.0, -4.0, -1.5, -0.5,
         -0.0, 0.0, 0.5, 1.5, 4.0, 12.0, 32.0, 127.0], dtype=np.float32
    ).reshape(4, 4)]
    if trace_path is None:
        return vectors
    with np.load(trace_path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            value = np.asarray(archive[key])
            if value.dtype != np.float32 or value.size == 0 or not np.isfinite(value).all():
                continue
            # Keep qualification bounded while preserving a tensor-shaped case.
            flat = np.ascontiguousarray(value).reshape(-1)
            if flat.size > 4096:
                flat = flat[:4096]
            vectors.append(flat.reshape(1, -1))
            if len(vectors) >= 4:
                break
    return vectors


def _case(result: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(result)
    reference = np.asarray(expected)
    exact = bool(
        actual.dtype == reference.dtype
        and actual.shape == reference.shape
        and actual.flags.c_contiguous
        and reference.flags.c_contiguous
        and np.array_equal(actual, reference)
    )
    return {
        "exact": exact,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "c_contiguous": bool(actual.flags.c_contiguous),
        "actual_sha256": _digest(actual),
        "reference_sha256": _digest(reference),
    }


def _physical_case(result: tuple, expected: tuple) -> dict[str, Any]:
    actual = [np.asarray(bank) for bank in result]
    reference = [np.asarray(bank) for bank in expected]
    exact = bool(
        len(actual) == len(reference) == 2
        and all(a.dtype == r.dtype == np.uint8 for a, r in zip(actual, reference))
        and all(a.flags.c_contiguous and r.flags.c_contiguous
                for a, r in zip(actual, reference))
        and all(np.array_equal(a, r) for a, r in zip(actual, reference))
    )
    return {
        "exact": exact,
        "shape": [list(bank.shape) for bank in actual],
        "dtype": [str(bank.dtype) for bank in actual],
        "c_contiguous": all(bank.flags.c_contiguous for bank in actual),
        "actual_sha256": [_digest(bank) for bank in actual],
        "reference_sha256": [_digest(bank) for bank in reference],
    }


def _ndwc_descriptor(dims: tuple[int, int, int, int], bitdepth: int,
                     direction: str, index: int = 0) -> dict[str, Any]:
    element_bytes = bitdepth // 8
    c_align = dims[1] * ((dims[3] + 15) // 16) * element_bytes
    w_align = ((dims[2] + 15) // 16) * c_align
    return {
        "layout": "NDWC", "dims": list(dims), "bitdepth": bitdepth,
        "c_align": c_align, "w_align": w_align,
        "combined_bytes": w_align * 256, "direction": direction,
        "index": index,
        "matrix_role": "output" if direction == "output" else "left",
    }


def qualify_host_executor(
    extension: Any, extension_path: Path, trace_path: Path | None = None
) -> dict[str, Any]:
    """Run the exact vector suite without constructing ``DmaBatch``."""
    extension_path = Path(extension_path).resolve()
    extension_sha256 = _sha256_file(extension_path)
    native = extension.HostGraphExecutor()
    python = PythonHostExecutor()
    vectors = _vectors(trace_path)
    cases: dict[str, list[dict[str, Any]]] = {name: [] for name in OPERATIONS}
    physical_cases: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PHYSICAL_FUSIONS
    }
    for value in vectors:
        for scale in _SCALES:
            cases["quantize"].append(_case(
                native.quantize(value, scale), python.quantize(value, scale)))
            cases["gelu_quantize"].append(_case(
                native.gelu_quantize(value, scale),
                python.gelu_quantize(value, scale)))
        right = np.ascontiguousarray(np.flip(value, axis=-1))
        cases["add"].append(_case(native.add(value, right), python.add(value, right)))
        for scale in _SCALES:
            cases["add_quantize"].append(_case(
                native.add_quantize(value, right, scale),
                python.add_quantize(value, right, scale)))
        pieces = [np.ascontiguousarray(value[..., : value.shape[-1] // 2]),
                  np.ascontiguousarray(value[..., value.shape[-1] // 2 :])]
        cases["concatenate"].append(_case(
            native.concatenate(pieces, -1), python.concatenate(pieces, -1)))
        cases["concatenate"].append(_case(
            native.concatenate([value, value], 0),
            python.concatenate([value, value], 0)))

    bf16_domain = _finite_bf16_domain()
    with np.errstate(over="ignore", invalid="ignore"):
        for scale in _SCALES:
            cases["gelu_quantize"].append(_case(
                native.gelu_quantize(bf16_domain, scale),
                python.gelu_quantize(bf16_domain, scale)))

    resize_rng = np.random.default_rng(250)
    for input_shape, output_shape in (
        ((1, 3, 2, 3), (1, 3, 1, 1)),
        ((1, 8, 19, 37), (1, 8, 37, 74)),
        ((1, 4, 37, 74), (1, 4, 74, 148)),
        ((1, 2, 74, 148), (1, 2, 148, 296)),
        ((1, 1, 148, 296), (1, 1, 296, 518)),
        ((1, 2, 19, 37), (1, 2, 75, 518)),
    ):
        value = resize_rng.standard_normal(input_shape, dtype=np.float32)
        sizes = np.asarray(output_shape, dtype=np.int64)
        cases["resize_align_corners"].append(_case(
            native.resize_align_corners(value, output_shape[2], output_shape[3]),
            python.resize_align_corners(value, sizes)))

    fusion_rng = np.random.default_rng(6250)
    for channels in ((16, 32, 16, 48, 16, 32), (16, 16, 16, 16, 16, 16)):
        logical_sources = []
        source_descriptors = []
        physical_sources = []
        for index, channel_count in enumerate(channels):
            descriptor = _ndwc_descriptor(
                (1, 1, 16, channel_count), 16, "output", index
            )
            value = fusion_rng.standard_normal(
                descriptor["dims"], dtype=np.float32
            )
            physical = extension.DmaBatch.pack_tensor(value, descriptor)
            logical_sources.append(extension.DmaBatch.unpack_tensor(
                physical[0], physical[1], descriptor
            ))
            source_descriptors.append(descriptor)
            physical_sources.append(physical)
        target_descriptor = _ndwc_descriptor(
            (1, 1, 16, sum(channels)), 8, "input"
        )
        scale = _SCALES[len(physical_cases["gelu_pack_bf16_concatenate"])
                        % len(_SCALES)]
        logical = np.concatenate(logical_sources, axis=3)
        expected = extension.DmaBatch.pack_tensor(
            python.gelu_quantize(logical, scale), target_descriptor
        )
        actual = native.gelu_pack_bf16_concatenate(
            physical_sources, source_descriptors, target_descriptor, scale
        )
        physical_cases["gelu_pack_bf16_concatenate"].append(
            _physical_case(actual, expected)
        )

    finite_domain = _finite_bf16_domain().reshape(1, 1, 1, -1)
    domain_source = _ndwc_descriptor(tuple(finite_domain.shape), 16, "output")
    domain_target = _ndwc_descriptor(tuple(finite_domain.shape), 8, "input")
    domain_physical = extension.DmaBatch.pack_tensor(finite_domain, domain_source)
    domain_scale = _SCALES[1]
    with np.errstate(over="ignore", invalid="ignore"):
        domain_expected = extension.DmaBatch.pack_tensor(
            python.gelu_quantize(finite_domain, domain_scale), domain_target
        )
    domain_actual = native.gelu_pack_bf16_concatenate(
        [domain_physical], [domain_source], domain_target, domain_scale
    )
    physical_cases["gelu_pack_bf16_concatenate"].append(
        _physical_case(domain_actual, domain_expected)
    )

    operations = {
        name: {
            "exact": all(item["exact"] for item in values),
            "cases": len(values),
            "vectors": values,
        }
        for name, values in cases.items()
    }
    physical_fusions = {
        name: {
            "exact": all(item["exact"] for item in values),
            "cases": len(values),
            "vectors": values,
        }
        for name, values in physical_cases.items()
    }
    qualified = (all(item["exact"] for item in operations.values())
                 and all(item["exact"] for item in physical_fusions.values()))
    report: dict[str, Any] = {
        "schema": "u250-host-executor-qualification-v1",
        "qualified": qualified,
        "source_sha256": _sha256_file(Path(__file__).with_name("u250_host_graph.hpp")),
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha256,
        "trace_path": str(Path(trace_path).resolve()) if trace_path else None,
        "trace_sha256": _sha256_file(Path(trace_path)) if trace_path else None,
        "operations": operations,
        "physical_fusions": physical_fusions,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga-dma-batch", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extension_path = resolve_fpga_dma_batch(args.fpga_dma_batch)
    extension = load_fpga_dma_batch(extension_path)
    report = qualify_host_executor(extension, extension_path, args.trace)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualified": report["qualified"],
                      "output": str(args.output),
                      "extension_sha256": report["extension_sha256"]},
                     sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
