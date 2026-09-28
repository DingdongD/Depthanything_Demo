#!/usr/bin/env python3
"""Capture FP32 encoder/decoder boundaries for causal U250 replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The board's lean inference environment intentionally omits torchvision.  The
# model's tensor-only forward does not use Compose, but dpt.py imports it for
# infer_image(), so provide the smallest import-compatible fallback.
try:
    from torchvision.transforms import Compose as _Compose  # noqa: F401
except ModuleNotFoundError:
    transforms = types.ModuleType("torchvision.transforms")

    class Compose:
        def __init__(self, transforms_list):
            self.transforms = transforms_list

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    transforms.Compose = Compose
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms

from depth_anything_v2.dpt import DepthAnythingV2
try:
    from tools.u250_calibration_manifest import load_manifest, samples_for_shape
except ModuleNotFoundError:  # Direct execution keeps tools/ on sys.path.
    from u250_calibration_manifest import load_manifest, samples_for_shape


ENCODER_INTERNAL_KEYS = (
    "norm1", "q", "k", "v", "attention", "attention_branch", "post",
    "norm2", "fc1", "gelu", "fc2",
)


def sampled_flat(value: np.ndarray, count: int, identity: str) -> np.ndarray:
    """Return a deterministic, bounded calibration view of a tensor."""
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size <= count:
        return np.ascontiguousarray(flat)
    stride = max(flat.size // count, 1)
    phase = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    offset = phase % stride
    return np.ascontiguousarray(flat[offset::stride][:count])


def compact_calibration_trace(captured: dict[str, np.ndarray], sample_id: str,
                              values_per_boundary: int,
                              attention_query_rows: int) -> dict[str, np.ndarray]:
    """Keep all 1-6 boundaries without writing multi-GB full tensor traces."""
    result = {"depth": captured["depth"]}
    for layer in range(12):
        prefix = f"encoder_l{layer:02d}_"
        q = captured[prefix + "q"]
        tokens = q.shape[-2]
        rows = np.linspace(
            0, tokens - 1, min(attention_query_rows, tokens), dtype=np.int64
        )
        result[prefix + "query_rows"] = rows
        result[prefix + "q_rows"] = np.ascontiguousarray(q[..., rows, :])
        result[prefix + "k"] = captured[prefix + "k"]
        result[prefix + "v"] = captured[prefix + "v"]
        result[prefix + "attention_rows"] = np.ascontiguousarray(
            captured[prefix + "attention"][..., rows, :]
        )
        for key in ("norm1", "attention", "post", "norm2", "fc1", "gelu", "fc2"):
            name = prefix + key
            result[name + "_values"] = sampled_flat(
                captured[name], values_per_boundary, f"{sample_id}:{name}"
            )
        block = f"block_l{layer:02d}"
        result[block + "_values"] = sampled_flat(
            captured[block], values_per_boundary, f"{sample_id}:{block}"
        )
    for index in range(32):
        for kind in ("input", "conv"):
            name = f"decoder_{kind}_{index:02d}"
            result[name + "_values"] = sampled_flat(
                captured[name], values_per_boundary, f"{sample_id}:{name}"
            )
    return result


def qkv_runtime_boundaries(
    output: torch.Tensor, num_heads: int, q_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the three BF16 QKV kernel outputs, including fused Q scaling."""
    batch, tokens, three_channels = output.shape
    if three_channels % (3 * num_heads):
        raise ValueError(f"invalid QKV shape {tuple(output.shape)}")
    head_dim = three_channels // (3 * num_heads)
    qkv = output.reshape(
        batch, tokens, 3, num_heads, head_dim
    ).permute(2, 0, 1, 3, 4).reshape(3, batch, tokens, num_heads * head_dim)
    return qkv[0] * q_scale, qkv[1], qkv[2]


def decoder_module_name(node_name: str) -> str:
    """Map one exported Decoder Conv node to its source PyTorch module."""
    prefix = "/depth_head/"
    suffix = "/Conv"
    if not node_name.startswith(prefix) or not node_name.endswith(suffix):
        raise ValueError(f"not a DepthAnything decoder Conv: {node_name}")
    relative = node_name[len(prefix):-len(suffix)].replace("/", ".")
    if relative in ("resize_layers.0.conv", "resize_layers.1.conv"):
        relative = relative.removesuffix(".conv")
    if relative.startswith("output_conv2.output_conv2."):
        relative = relative.removeprefix("output_conv2.")
    if relative.startswith(("layer", "refinenet", "output_conv")):
        relative = "scratch." + relative
    return "depth_head." + relative


def decoder_capture_modules(model: nn.Module, host_plan: dict) -> dict[int, nn.Module]:
    modules = dict(model.named_modules())
    result = {}
    for step in host_plan["decoder_steps"]:
        if step["backend"] != "npu":
            continue
        index = int(step["index"])
        name = decoder_module_name(step["name"])
        module = modules.get(name)
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            raise ValueError(f"decoder Conv {index} does not map to {name}")
        result[index] = module
    expected = set(range(32))
    if set(result) != expected:
        raise ValueError(f"decoder Conv indices mismatch: {sorted(set(result) ^ expected)}")
    return result


def inverse_depth_to_space_crd(value: torch.Tensor, block: int) -> torch.Tensor:
    """Return the pre-DepthToSpace tensor for ONNX CRD channel ordering."""
    n, channels, height, width = value.shape
    if height % block or width % block:
        raise ValueError(f"cannot invert CRD block {block} for {tuple(value.shape)}")
    return value.reshape(
        n, channels, height // block, block, width // block, block
    ).permute(0, 1, 3, 5, 2, 4).reshape(
        n, channels * block * block, height // block, width // block
    )


def decoder_boundary_output(index: int, output: torch.Tensor) -> torch.Tensor:
    """Translate PyTorch ConvTranspose outputs to exported pre-D2S boundaries."""
    if index == 1:
        return inverse_depth_to_space_crd(
            inverse_depth_to_space_crd(output, 2), 2
        )
    if index == 3:
        return inverse_depth_to_space_crd(output, 2)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--sample-manifest", type=Path,
        help="immutable full-graph calibration manifest; replaces file selection",
    )
    parser.add_argument(
        "--manifest-shape", type=int,
        help="input side selected from --sample-manifest (for example 280 or 518)",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample", action="append",
                        help="relative sample stem; default captures every input")
    parser.add_argument(
        "--encoder-internal-layer", type=int, action="append", default=[],
        help="capture internal boundaries for this layer (repeatable)",
    )
    parser.add_argument(
        "--encoder-only", action="store_true",
        help=("omit decoder boundaries from saved traces; the model still runs "
              "end to end so encoder hooks observe the normal execution path"),
    )
    parser.add_argument(
        "--compact-calibration", action="store_true",
        help="save bounded all-layer calibration views plus full final depth",
    )
    parser.add_argument("--values-per-boundary", type=int, default=16384)
    parser.add_argument("--attention-query-rows", type=int, default=16)
    args = parser.parse_args()
    internal_layers = sorted(set(args.encoder_internal_layer))
    if args.sample_manifest is not None:
        if args.input_root is not None or args.sample:
            parser.error("--sample-manifest is mutually exclusive with --input-root/--sample")
        if args.manifest_shape is None:
            parser.error("--sample-manifest requires --manifest-shape")
        if args.encoder_only:
            parser.error("a full-graph calibration manifest cannot use --encoder-only")
        # One pass captures every boundary.  Callers cannot accidentally omit
        # a later block and recreate the historical layer-by-layer workflow.
        internal_layers = list(range(12))
    elif args.input_root is None:
        parser.error("one of --input-root or --sample-manifest is required")
    if args.compact_calibration and args.sample_manifest is None:
        parser.error("--compact-calibration requires --sample-manifest")
    if args.values_per_boundary <= 0 or args.attention_query_rows <= 0:
        parser.error("compact calibration sample counts must be positive")
    if any(not 0 <= layer <= 11 for layer in internal_layers):
        parser.error("--encoder-internal-layer must be within [0, 11]")

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    host_plan = json.loads(args.host_plan.read_text())
    expected_input_shape = tuple(
        host_plan.get("frontend", {}).get("input_shape", [1, 3, 518, 518])
    )
    decoder_modules = ({}
                       if args.encoder_only
                       else decoder_capture_modules(model, host_plan))
    captured: dict[str, np.ndarray] = {}
    handles = []

    def capture(key: str, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{key} produced non-tensor output")
        # A forward hook runs before the next module.  Some DepthAnything
        # decoder activations use ReLU(inplace=True), so a CPU NumPy view can
        # otherwise be mutated after this hook returns (notably conv30/31).
        # Keep an owning snapshot of the exact pre-activation boundary.
        captured[key] = np.ascontiguousarray(
            output.detach().float().cpu().numpy(), dtype=np.float32
        ).copy()

    for layer, block in enumerate(model.pretrained.blocks):
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"block_l{layer_index:02d}", output)
        ))
        if layer not in internal_layers:
            continue
        handles.append(block.norm1.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_norm1", output)
        ))

        def capture_qkv(
            _module, _inputs, output, *, layer_index=layer,
            heads=block.attn.num_heads, q_scale=block.attn.scale,
        ):
            q, k, v = qkv_runtime_boundaries(output, heads, q_scale)
            capture(f"encoder_l{layer_index:02d}_q", q[:, None])
            capture(f"encoder_l{layer_index:02d}_k", k[:, None])
            capture(f"encoder_l{layer_index:02d}_v", v[:, None])

        handles.append(block.attn.qkv.register_forward_hook(capture_qkv))
        handles.append(block.attn.proj.register_forward_pre_hook(
            lambda _module, inputs, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_attention", inputs[0][:, None])
        ))
        handles.append(block.ls1.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_attention_branch", output)
        ))
        handles.append(block.norm2.register_forward_pre_hook(
            lambda _module, inputs, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_post", inputs[0])
        ))
        handles.append(block.norm2.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_norm2", output)
        ))
        handles.append(block.mlp.fc1.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_fc1", output[:, None])
        ))
        handles.append(block.mlp.act.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_gelu", output[:, None])
        ))
        handles.append(block.ls2.register_forward_hook(
            lambda _module, _inputs, output, layer_index=layer:
            capture(f"encoder_l{layer_index:02d}_fc2", output)
        ))
    for index, module in decoder_modules.items():
        handles.append(module.register_forward_pre_hook(
            lambda _module, inputs, conv_index=index:
            capture(f"decoder_input_{conv_index:02d}", inputs[0])
        ))
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, conv_index=index:
            capture(
                f"decoder_conv_{conv_index:02d}",
                decoder_boundary_output(conv_index, output),
            )
        ))

    calibration_manifest = None
    selected_records = None
    if args.sample_manifest is not None:
        calibration_manifest = load_manifest(args.sample_manifest)
        selected_records = samples_for_shape(
            calibration_manifest, args.manifest_shape, verify_files=True
        )
        inputs = [item["tensor_path"] for item in selected_records]
    elif args.sample:
        inputs = [args.input_root / f"{sample}.npy" for sample in args.sample]
    else:
        inputs = sorted(args.input_root.rglob("*.npy"))
    if not inputs or any(not path.is_file() for path in inputs):
        raise ValueError("no valid input tensors selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for position, path in enumerate(inputs, 1):
            captured.clear()
            value = np.load(path, allow_pickle=False).astype(np.float32)
            if value.shape != expected_input_shape:
                raise ValueError(f"unexpected input shape {value.shape}: {path}")
            depth = model(torch.from_numpy(value).to(device))
            capture("depth", depth)
            for layer in (2, 5, 8, 11):
                captured[f"capture_l{layer:02d}"] = captured[f"block_l{layer:02d}"]
            expected = {
                *(f"block_l{layer:02d}" for layer in range(12)),
                *(f"capture_l{layer:02d}" for layer in (2, 5, 8, 11)),
                "depth",
            }
            if not args.encoder_only:
                expected.update(
                    f"decoder_{kind}_{index:02d}"
                    for kind in ("input", "conv") for index in range(32)
                )
            expected.update(
                f"encoder_l{layer:02d}_{key}"
                for layer in internal_layers for key in ENCODER_INTERNAL_KEYS
            )
            if set(captured) != expected:
                raise RuntimeError(f"capture mismatch: {sorted(set(captured) ^ expected)}")
            if selected_records is not None:
                sample_record = selected_records[position - 1]
                sample_id = sample_record["sample_id"]
                relative = Path(sample_id).with_suffix(".npz")
            else:
                sample_record = None
                sample_id = path.relative_to(args.input_root).with_suffix("").as_posix()
                relative = path.relative_to(args.input_root).with_suffix(".npz")
            output = args.output_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            saved = (
                compact_calibration_trace(
                    captured, sample_id, args.values_per_boundary,
                    args.attention_query_rows,
                )
                if args.compact_calibration else captured
            )
            np.savez(output, **saved)
            records.append({
                "sample": sample_id,
                "domain": None if sample_record is None else sample_record["domain"],
                "split": None if sample_record is None else sample_record["split"],
                "bytes": output.stat().st_size,
            })
            print(json.dumps({
                "captured": position, "total": len(inputs),
                "sample": records[-1]["sample"],
                "elapsed_seconds": time.perf_counter() - started,
            }), flush=True)

    for handle in handles:
        handle.remove()
    report = {
        "schema": ("depthanything-fp32-full-graph-calibration-traces-v1"
                   if args.compact_calibration
                   else "depthanything-fp32-replacement-traces-v1"),
        "device": str(device), "samples": records,
        "encoder_blocks": list(range(12)),
        "decoder_convs": [] if args.encoder_only else list(range(32)),
        "encoder_only": args.encoder_only,
        "encoder_internal_layers": internal_layers,
        "compact_calibration": bool(args.compact_calibration),
        "calibration_manifest": (
            None if calibration_manifest is None else str(args.sample_manifest.resolve())
        ),
        "calibration_manifest_sha256": (
            None if calibration_manifest is None
            else calibration_manifest["manifest_sha256"]
        ),
        "manifest_shape": args.manifest_shape,
        "values_per_boundary": args.values_per_boundary,
        "attention_query_rows": args.attention_query_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
