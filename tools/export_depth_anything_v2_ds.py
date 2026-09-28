#!/usr/bin/env python3
"""Export the fixed-shape Depth Anything V2-Small graph for DS-Compiler."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def detach_numpy(value: torch.Tensor) -> np.ndarray:
    """Materialize a contiguous NumPy copy independent of Torch storage."""

    return np.ascontiguousarray(value.detach().cpu().numpy()).copy()


def _tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def audit_graph(path: str | Path) -> dict[str, Any]:
    """Return deterministic shape and operator metadata for an ONNX graph."""

    model = onnx.load(str(path), load_external_data=False)
    tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    dynamic = sorted(
        value.name
        for value in tensors
        if any(not isinstance(dim, int) for dim in _tensor_shape(value))
    )
    return {
        "inputs": {value.name: _tensor_shape(value) for value in model.graph.input},
        "outputs": {value.name: _tensor_shape(value) for value in model.graph.output},
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "dynamic_tensors": dynamic,
        "opset": max((entry.version for entry in model.opset_import if not entry.domain), default=0),
    }


def export_model(
    output: Path,
    seed: int,
    verify_ort: bool = True,
    static_attention_profile: Path | None = None,
    adapter_module: str = "ds_models.depth_anything_v2_vits",
) -> dict[str, Any]:
    adapter = importlib.import_module(adapter_module)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = adapter.Model().cpu().eval()
    if static_attention_profile is not None:
        model.enable_static_int8_attention_export(static_attention_profile)
    input_tensor = torch.randn(1, *adapter.ifmap_sz, dtype=torch.float32)

    with torch.no_grad():
        torch_output = detach_numpy(model(input_tensor))

    torch.onnx.export(
        model,
        input_tensor,
        str(output),
        input_names=list(adapter.input_names),
        output_names=["depth"],
        opset_version=adapter.op_version,
        do_constant_folding=True,
        dynamo=False,
    )

    input_path = output.with_suffix(".input.npy")
    torch_path = output.with_suffix(".pytorch.npy")
    np.save(input_path, input_tensor.numpy())
    np.save(torch_path, torch_output)

    audit = audit_graph(output)
    audit["checkpoint"] = str(adapter.CHECKPOINT)
    audit["checkpoint_bytes"] = Path(adapter.CHECKPOINT).stat().st_size
    audit["seed"] = seed
    audit["static_attention_profile"] = (
        str(static_attention_profile.resolve())
        if static_attention_profile is not None else None
    )
    audit["adapter_module"] = adapter_module

    if verify_ort:
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_output = np.ascontiguousarray(
            session.run(["depth"], {adapter.input_names[0]: input_tensor.numpy()})[0]
        ).copy()
        np.save(output.with_suffix(".onnxruntime.npy"), ort_output)
        abs_error = np.abs(torch_output - ort_output)
        audit["onnxruntime"] = {
            "output_shape": list(ort_output.shape),
            "finite": bool(np.isfinite(ort_output).all()),
            "max_abs_error": float(abs_error.max()),
            "mean_abs_error": float(abs_error.mean()),
        }

    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/depth_anything_v2_vits.onnx"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--skip-ort", action="store_true")
    parser.add_argument("--static-attention-profile", type=Path)
    parser.add_argument(
        "--adapter-module", default="ds_models.depth_anything_v2_vits",
        help="DS model adapter module; use depth_anything_v2_vits_280 for 280x280",
    )
    args = parser.parse_args()

    audit = export_model(
        args.output,
        args.seed,
        verify_ort=not args.skip_ort,
        static_attention_profile=args.static_attention_profile,
        adapter_module=args.adapter_module,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
