#!/usr/bin/env python3
"""Capture one real model attention head and build its six U250 invocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dinov2_layers.attention import Attention  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from ds_models.static_int8_attention import (  # noqa: E402
    StaticAttentionProfile, StaticInt8AttentionRuntime, plan_query_chunks,
    quantize_symmetric,
)


MODEL_CONFIG = {"encoder": "vits", "features": 64,
                "out_channels": [48, 96, 192, 384]}


def preprocess(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((image - mean) / std).transpose(2, 0, 1)).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--image", type=Path,
                        default=REPO_ROOT / "assets/examples/demo17.jpg")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--compiled-prefix", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = StaticAttentionProfile.load(args.profile)
    if not 0 <= args.layer < len(profile.layers) or not 0 <= args.head < profile.heads:
        raise ValueError("layer/head is outside the profile")
    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    attention = [m for m in model.modules() if isinstance(m, Attention)][args.layer]
    captured = []
    hook = attention.qkv.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
    with torch.inference_mode():
        model(preprocess(args.image))
    hook.remove()
    qkv = captured[0]
    b, tokens, channels3 = qkv.shape
    channels = channels3 // 3
    qkv = qkv.reshape(b, tokens, 3, attention.num_heads, channels // attention.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q = qkv[0] * attention.scale
    k, v = qkv[1], qkv[2]
    scale = profile.scale(args.layer, args.head)
    qi = quantize_symmetric(q[:, args.head], scale.q)
    ki = quantize_symmetric(k[:, args.head], scale.k)
    vi = quantize_symmetric(v[:, args.head], scale.v)
    kt = ki.transpose(-2, -1).contiguous()
    runtime = StaticInt8AttentionRuntime(profile)

    source_ddr = source_cfg = None
    if args.compiled_prefix:
        source_ddr = args.compiled_prefix.with_name(args.compiled_prefix.name + "_ddr.bin")
        source_cfg = args.compiled_prefix.with_name(args.compiled_prefix.name + "_cfg.txt")
        if not source_ddr.is_file() or not source_cfg.is_file():
            raise FileNotFoundError("compiled DDR binary/cfg missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    combined = []
    for chunk in plan_query_chunks(tokens):
        case_name = f"attention_l{args.layer:02d}_h{args.head:02d}_chunk{chunk.index}"
        case_dir = args.output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        query = torch.zeros(1, 256, profile.head_dimension, dtype=torch.int8)
        query[:, :chunk.valid_rows] = qi[:, chunk.start:chunk.stop]
        golden = runtime._surrogate(query, kt, vi, scale)
        combined.append(golden[:, :chunk.valid_rows])
        # The compiler cfg describes matrix tensors in physical NDWC form as
        # [N, D=1, L, C].  Preserve that singleton dimension for the vendor
        # NPZ codec; the model/runtime API intentionally remains [N, L, C].
        np.savez(case_dir / "logical_input0.npz", input=query.unsqueeze(1).numpy())
        np.savez(case_dir / "logical_input1.npz", input=kt.unsqueeze(1).numpy())
        np.savez(case_dir / "logical_input2.npz", input=vi.unsqueeze(1).numpy())
        np.savez(
            case_dir / "golden_output0.npz",
            output_bf16=golden.unsqueeze(1).numpy(),
        )
        if source_ddr and source_cfg:
            shutil.copy2(source_ddr, case_dir / f"{case_name}_ddr.bin")
            shutil.copy2(source_cfg, case_dir / f"{case_name}_cfg.txt")
        rows.append({"index": chunk.index, "start": chunk.start, "stop": chunk.stop,
                     "valid_rows": chunk.valid_rows, "case_name": case_name})
    quantized_output = torch.cat(combined, dim=1)
    ideal_output = torch.softmax(
        q[:, args.head] @ k[:, args.head].transpose(-2, -1), dim=-1
    ) @ v[:, args.head]
    np.savez(
        args.output_dir / "software_reference.npz",
        q=q[:, args.head].numpy(), k=k[:, args.head].numpy(), v=v[:, args.head].numpy(),
        query_i8=qi.numpy(), key_transposed_i8=kt.numpy(), value_i8=vi.numpy(),
        quantized_output_bf16=quantized_output.numpy(), ideal_output=ideal_output.numpy(),
    )
    rel_l2 = float(torch.linalg.vector_norm(quantized_output - ideal_output)
                   / torch.linalg.vector_norm(ideal_output).clamp_min(1e-30))
    manifest = {
        "profile": str(args.profile.resolve()), "image": str(args.image.resolve()),
        "layer": args.layer, "head": args.head, "scales": scale.__dict__,
        "chunks": rows, "quantized_vs_ideal_relative_l2": rel_l2,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
