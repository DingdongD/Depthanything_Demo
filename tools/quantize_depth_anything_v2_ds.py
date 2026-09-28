#!/usr/bin/env python3
"""Run the shared DS quantizer with an Erf PWL legalization shim.

The bundled quantizer already handles GELU, but its PWL helper omitted the
``Erf`` branch emitted by the legacy PyTorch ONNX exporter.  Depth Anything
V2 uses GELU, so this wrapper supplies only that missing numerical branch and
leaves all DS quantization and serialization code unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


def erf_pwl(input_tensor: torch.Tensor, segments: int = 32):
    """Return finite piecewise-linear coefficients approximating ``erf``."""

    values = torch.as_tensor(input_tensor, dtype=torch.float32)
    input_range = float(torch.max(torch.abs(values)).item())
    if not torch.isfinite(torch.tensor(input_range)) or input_range <= 0.0:
        input_range = 1.0

    edges = torch.linspace(-input_range, input_range, segments + 1, dtype=torch.float32)
    x0, x1 = edges[:-1], edges[1:]
    y0, y1 = torch.erf(x0), torch.erf(x1)
    slopes = (y1 - y0) / (x1 - x0)
    intercepts = y0 - slopes * x0
    return slopes.tolist(), intercepts.tolist(), edges[1:].tolist()


def quantize(input_path: Path, data_range: int,
             toolchain_root: Path | None = None) -> Path:
    """Quantize an ONNX model in-place and return the generated DS model path."""

    if toolchain_root is None:
        configured = os.environ.get("DS_TOOLCHAIN_ROOT")
        if not configured:
            raise ValueError(
                "set DS_TOOLCHAIN_ROOT or pass --toolchain-root"
            )
        toolchain_root = Path(configured)
    toolchain_lib = toolchain_root.expanduser().resolve() / "python_libs"
    if not toolchain_lib.is_dir():
        raise FileNotFoundError(f"DS toolchain libraries not found: {toolchain_lib}")
    if not input_path.is_file():
        raise FileNotFoundError(f"ONNX input not found: {input_path}")

    if str(toolchain_lib) not in sys.path:
        sys.path.insert(0, str(toolchain_lib))
    sys.argv = ["quant_onnx", input_path.stem]
    from ACModelHelper.util import quant_onnx as quant_module

    original = quant_module.optimize_slopes_scale

    def optimize(input_tensor, op_type):
        if op_type == "Erf":
            return erf_pwl(input_tensor)
        return original(input_tensor, op_type)

    quant_module.optimize_slopes_scale = optimize
    input_dir = input_path.resolve().parent
    stem = input_path.stem
    old_cwd = Path.cwd()
    try:
        # The shared quantizer stores external-data locations relative to cwd.
        os.chdir(input_dir)
        quant_module.quant_onnx(stem, data_range=data_range)
    finally:
        os.chdir(old_cwd)

    output = input_dir / f"{stem}_sc.onnx"
    if not output.is_file():
        raise RuntimeError(f"DS quantizer did not produce {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/depth_anything_v2_vits.onnx"))
    parser.add_argument("--data-range", type=int, default=2)
    parser.add_argument(
        "--toolchain-root", type=Path,
        help="DS toolchain root; defaults to DS_TOOLCHAIN_ROOT",
    )
    args = parser.parse_args()
    output = quantize(args.input, args.data_range, args.toolchain_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
