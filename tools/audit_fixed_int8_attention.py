#!/usr/bin/env python3
"""Numerically audit fixed-scale INT8 attention for Depth Anything V2 ViT-S.

This is a software surrogate for the proposed U250 path:

    BF16 Q,K -> fixed-scale INT8 QK^T -> BF16 Softmax (SPU surrogate)
    -> fixed-scale unsigned-range-in-signed-INT8 probabilities
    -> fixed-scale INT8 V -> INT32 Attention@V -> BF16

It deliberately isolates the two attention matrix multiplications.  The QKV and
output projections, MLP, layer norms, and DPT head remain in PyTorch FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dinov2_layers.attention import Attention  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=path, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def bf16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def quantize_symmetric(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return signed INT8 values represented as INT32 for exact accumulation."""
    return torch.round(value / scale).clamp(-127, 127).to(torch.int32)


def quantize_probability(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Use the nonnegative half (0..127) of signed INT8."""
    return torch.round(value / scale).clamp(0, 127).to(torch.int32)


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.detach().float().reshape(-1)
    got = candidate.detach().float().reshape(-1)
    diff = got - ref
    ref_norm = torch.linalg.vector_norm(ref).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    centered_ref = ref - ref.mean()
    centered_got = got - got.mean()
    pearson_den = (
        torch.linalg.vector_norm(centered_ref) *
        torch.linalg.vector_norm(centered_got)
    ).item()
    return {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(diff * diff)).item(),
        "max_abs": diff.abs().max().item(),
        "relative_l2": diff_norm / max(ref_norm, 1e-12),
        "cosine": torch.dot(ref, got).item() /
                  max(ref_norm * torch.linalg.vector_norm(got).item(), 1e-12),
        "pearson": torch.dot(centered_ref, centered_got).item() /
                   max(pearson_den, 1e-12),
    }


def aggregate_metrics(items: List[dict]) -> dict:
    keys = items[0].keys()
    return {
        key: {
            "mean": float(np.mean([item[key] for item in items])),
            "max": float(np.max([item[key] for item in items])),
        }
        for key in keys
    }


def square_preprocess(image_path: Path, size: int) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


@dataclass
class LayerCalibration:
    q_head_max: torch.Tensor
    k_head_max: torch.Tensor
    v_head_max: torch.Tensor
    p_head_max: torch.Tensor
    calls: int = 0


@dataclass
class ErrorAccumulator:
    squared_error: float = 0.0
    squared_reference: float = 0.0
    absolute_error: float = 0.0
    count: int = 0
    max_abs: float = 0.0

    def add(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        diff = (candidate.float() - reference.float()).detach()
        self.squared_error += torch.sum(diff * diff).item()
        self.squared_reference += torch.sum(reference.float() ** 2).item()
        self.absolute_error += torch.sum(diff.abs()).item()
        self.count += diff.numel()
        self.max_abs = max(self.max_abs, diff.abs().max().item())

    def result(self) -> dict:
        return {
            "relative_l2": math.sqrt(self.squared_error /
                                     max(self.squared_reference, 1e-30)),
            "rmse": math.sqrt(self.squared_error / max(self.count, 1)),
            "mae": self.absolute_error / max(self.count, 1),
            "max_abs": self.max_abs,
        }


@dataclass
class RuntimeStats:
    logits: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    probability: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    attention_output: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    q_saturated: int = 0
    q_count: int = 0
    k_saturated: int = 0
    k_count: int = 0
    v_saturated: int = 0
    v_count: int = 0
    p_saturated: int = 0
    p_zero: int = 0
    p_count: int = 0
    row_sum_abs_error: float = 0.0
    row_count: int = 0

    def result(self) -> dict:
        return {
            "logits": self.logits.result(),
            "probability": self.probability.result(),
            "attention_output": self.attention_output.result(),
            "saturation_fraction": {
                "q": self.q_saturated / max(self.q_count, 1),
                "k": self.k_saturated / max(self.k_count, 1),
                "v": self.v_saturated / max(self.v_count, 1),
                "probability": self.p_saturated / max(self.p_count, 1),
            },
            "probability_zero_fraction": self.p_zero / max(self.p_count, 1),
            "probability_row_sum_mae": self.row_sum_abs_error /
                                       max(self.row_count, 1),
        }


class AttentionAudit:
    def __init__(self, model: torch.nn.Module):
        self.layers: List[Attention] = [
            module for module in model.modules() if isinstance(module, Attention)
        ]
        self.calibration: Dict[int, LayerCalibration] = {}
        self.runtime: Dict[int, RuntimeStats] = {}
        self.tuning: Dict[int, Dict[int, List[ErrorAccumulator]]] = {}
        self.tuning_denominators = [127, 255, 511, 1023, 2047]
        self.tuned_denominators: Dict[int, List[int]] = {}
        self.mode = "reference"
        self.granularity = "head"
        self.probability_denominator = 127
        for index, layer in enumerate(self.layers):
            layer.forward = types.MethodType(self._make_forward(index), layer)

    def _make_forward(self, index: int):
        audit = self

        def forward(layer: Attention, x: torch.Tensor) -> torch.Tensor:
            batch, tokens, channels = x.shape
            qkv = layer.qkv(x).reshape(
                batch, tokens, 3, layer.num_heads, channels // layer.num_heads
            ).permute(2, 0, 3, 1, 4)
            q = qkv[0] * layer.scale
            k = qkv[1]
            v = qkv[2]

            if audit.mode == "calibrate":
                logits = q @ k.transpose(-2, -1)
                probability = logits.softmax(dim=-1)
                audit._update_calibration(index, q, k, v, probability)
                output = probability @ v
            elif audit.mode == "tune":
                output = audit._tune_probability_scale(index, q, k, v)
            elif audit.mode == "quantized":
                output = audit._quantized_attention(index, q, k, v)
            else:
                probability = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                output = probability @ v

            output = output.transpose(1, 2).reshape(batch, tokens, channels)
            output = layer.proj(output)
            return layer.proj_drop(output)

        return forward

    def _update_calibration(
        self,
        index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        probability: torch.Tensor,
    ) -> None:
        def head_max(value: torch.Tensor) -> torch.Tensor:
            return value.detach().abs().amax(dim=(0, 2, 3)).cpu()

        values = [head_max(item) for item in (q, k, v, probability)]
        if index not in self.calibration:
            self.calibration[index] = LayerCalibration(*values, calls=1)
            return
        current = self.calibration[index]
        current.q_head_max = torch.maximum(current.q_head_max, values[0])
        current.k_head_max = torch.maximum(current.k_head_max, values[1])
        current.v_head_max = torch.maximum(current.v_head_max, values[2])
        current.p_head_max = torch.maximum(current.p_head_max, values[3])
        current.calls += 1

    def _tune_probability_scale(
        self, index: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Score offline scale candidates without perturbing later layers."""
        calibration = self.calibration[index]
        q_bf16, k_bf16, v_bf16 = (bf16_round(item) for item in (q, k, v))
        sq = self._scale(calibration.q_head_max, q)
        sk = self._scale(calibration.k_head_max, k)
        sv = self._scale(calibration.v_head_max, v)
        qi = quantize_symmetric(q_bf16, sq)
        ki = quantize_symmetric(k_bf16, sk)
        vi = quantize_symmetric(v_bf16, sv)
        logits = bf16_round((qi @ ki.transpose(-2, -1)).float() * (sq * sk))
        probability = bf16_round(torch.softmax(logits, dim=-1))
        reference_output = (torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v)

        layer_stats = self.tuning.setdefault(index, {})
        for denominator in self.tuning_denominators:
            sp = bf16_round(torch.tensor(
                1.0 / denominator, dtype=torch.float32, device=q.device
            )).reshape(1, 1, 1, 1)
            pi = quantize_probability(probability, sp)
            candidate = bf16_round((pi @ vi).float() * (sp * sv))
            head_stats = layer_stats.setdefault(
                denominator, [ErrorAccumulator() for _ in range(q.shape[1])]
            )
            for head in range(q.shape[1]):
                head_stats[head].add(
                    reference_output[:, head], candidate[:, head]
                )
        return reference_output

    def select_tuned_denominators(self) -> Dict[int, List[int]]:
        selected = {}
        for index, candidates in sorted(self.tuning.items()):
            heads = len(next(iter(candidates.values())))
            selected[index] = []
            for head in range(heads):
                best = min(
                    self.tuning_denominators,
                    key=lambda denominator: candidates[denominator][head].squared_error,
                )
                selected[index].append(best)
        self.tuned_denominators = selected
        return selected

    def _scale(self, maximum: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if self.granularity == "tensor":
            maximum = maximum.max().reshape(1)
        scale = bf16_round(torch.clamp(maximum / 127.0, min=2 ** -24))
        if self.granularity == "tensor":
            return scale.reshape(1, 1, 1, 1).to(value.device)
        return scale.reshape(1, -1, 1, 1).to(value.device)

    def _quantized_attention(
        self, index: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        calibration = self.calibration[index]
        stats = self.runtime.setdefault(index, RuntimeStats())

        # NPU boundary surrogate: BF16 operands and BF16-stored fixed scales.
        q_bf16, k_bf16, v_bf16 = (bf16_round(item) for item in (q, k, v))
        sq = self._scale(calibration.q_head_max, q)
        sk = self._scale(calibration.k_head_max, k)
        sv = self._scale(calibration.v_head_max, v)
        qi = quantize_symmetric(q_bf16, sq)
        ki = quantize_symmetric(k_bf16, sk)
        vi = quantize_symmetric(v_bf16, sv)

        reference_logits = q @ k.transpose(-2, -1)
        logits_i32 = qi @ ki.transpose(-2, -1)
        logits = bf16_round(logits_i32.float() * (sq * sk))
        probability = bf16_round(torch.softmax(logits, dim=-1))

        # This is static for the whole graph variant; it is never computed from
        # the current input.  BF16 rounding models the encoded scale operand.
        if self.probability_denominator == "tuned":
            denominators = torch.tensor(
                self.tuned_denominators[index], dtype=torch.float32, device=q.device
            ).reshape(1, -1, 1, 1)
            sp = bf16_round(1.0 / denominators)
        else:
            sp = bf16_round(torch.tensor(
                1.0 / self.probability_denominator,
                dtype=torch.float32,
                device=q.device,
            )).reshape(1, 1, 1, 1)
        pi = quantize_probability(probability, sp)
        probability_dequant = bf16_round(pi.float() * sp)

        output_i32 = pi @ vi
        output = bf16_round(output_i32.float() * (sp * sv))

        reference_probability = torch.softmax(reference_logits, dim=-1)
        reference_output = reference_probability @ v
        stats.logits.add(reference_logits, logits)
        stats.probability.add(reference_probability, probability_dequant)
        stats.attention_output.add(reference_output, output)
        for name, integer in (("q", qi), ("k", ki), ("v", vi)):
            setattr(stats, f"{name}_saturated",
                    getattr(stats, f"{name}_saturated") +
                    torch.count_nonzero(integer.abs() == 127).item())
            setattr(stats, f"{name}_count",
                    getattr(stats, f"{name}_count") + integer.numel())
        stats.p_saturated += torch.count_nonzero(pi == 127).item()
        stats.p_zero += torch.count_nonzero(pi == 0).item()
        stats.p_count += pi.numel()
        row_error = (probability_dequant.sum(dim=-1) - 1.0).abs()
        stats.row_sum_abs_error += row_error.sum().item()
        stats.row_count += row_error.numel()
        return output

    def calibration_json(self) -> dict:
        result = {}
        for index, item in sorted(self.calibration.items()):
            result[str(index)] = {
                "calls": item.calls,
                "q_head_max": item.q_head_max.tolist(),
                "k_head_max": item.k_head_max.tolist(),
                "v_head_max": item.v_head_max.tolist(),
                "probability_head_max": item.p_head_max.tolist(),
                "q_head_scale_bf16": bf16_round(item.q_head_max / 127).tolist(),
                "k_head_scale_bf16": bf16_round(item.k_head_max / 127).tolist(),
                "v_head_scale_bf16": bf16_round(item.v_head_max / 127).tolist(),
            }
        return result

    def runtime_json(self) -> dict:
        per_layer = {str(i): stats.result() for i, stats in sorted(self.runtime.items())}
        combined = RuntimeStats()
        for stats in self.runtime.values():
            for metric_name in ("logits", "probability", "attention_output"):
                source = getattr(stats, metric_name)
                target = getattr(combined, metric_name)
                target.squared_error += source.squared_error
                target.squared_reference += source.squared_reference
                target.absolute_error += source.absolute_error
                target.count += source.count
                target.max_abs = max(target.max_abs, source.max_abs)
            for name in (
                "q_saturated", "q_count", "k_saturated", "k_count",
                "v_saturated", "v_count", "p_saturated", "p_zero", "p_count",
                "row_sum_abs_error", "row_count",
            ):
                setattr(combined, name, getattr(combined, name) + getattr(stats, name))
        return {"combined": combined.result(), "per_layer": per_layer}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=
                        REPO_ROOT / "checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "assets/examples")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    images = sorted(args.images.glob("*.jpg"))
    required = args.calibration_count + args.validation_count
    if len(images) < required:
        raise RuntimeError(f"need {required} images, found {len(images)}")
    calibration_images = images[:args.calibration_count]
    validation_images = images[-args.validation_count:]

    model = DepthAnythingV2(**MODEL_CONFIG)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    audit = AttentionAudit(model)

    audit.mode = "calibrate"
    with torch.inference_mode():
        for path in calibration_images:
            model(square_preprocess(path, args.input_size))

    # A second calibration-only pass chooses one constant from a small hardware-
    # friendly table for each already-separated layer/head MatMul.
    audit.mode = "tune"
    audit.granularity = "head"
    with torch.inference_mode():
        for path in calibration_images:
            model(square_preprocess(path, args.input_size))
    tuned_denominators = audit.select_tuned_denominators()

    inputs = {path.name: square_preprocess(path, args.input_size)
              for path in validation_images}
    references = {}
    audit.mode = "reference"
    with torch.inference_mode():
        for name, value in inputs.items():
            references[name] = model(value).detach().clone()

    strategies = [
        ("tensor_p127", "tensor", 127),
        ("head_p127", "head", 127),
        ("head_p255", "head", 255),
        ("head_p511", "head", 511),
        ("head_p1023", "head", 1023),
        ("head_p2047", "head", 2047),
        ("head_ptuned", "head", "tuned"),
    ]
    strategy_results = {}
    audit.mode = "quantized"
    for strategy_name, granularity, probability_denominator in strategies:
        audit.granularity = granularity
        audit.probability_denominator = probability_denominator
        audit.runtime = {}
        per_image = {}
        begin = time.time()
        with torch.inference_mode():
            for name, value in inputs.items():
                candidate = model(value)
                per_image[name] = tensor_metrics(references[name], candidate)
        if probability_denominator == "tuned":
            probability_scale = {
                str(index): bf16_round(1.0 / torch.tensor(values).float()).tolist()
                for index, values in tuned_denominators.items()
            }
            probability_max = {
                str(index): bf16_round(127.0 / torch.tensor(values).float()).tolist()
                for index, values in tuned_denominators.items()
            }
        else:
            probability_scale = float(bf16_round(torch.tensor(
                1.0 / probability_denominator)).item())
            probability_max = float(bf16_round(torch.tensor(
                127.0 / probability_denominator)).item())
        strategy_results[strategy_name] = {
            "fixed_probability_scale": probability_scale,
            "probability_representable_max": probability_max,
            "depth_output": {
                "aggregate": aggregate_metrics(list(per_image.values())),
                "per_image": per_image,
            },
            "attention": audit.runtime_json(),
            "runtime_seconds": time.time() - begin,
        }
        print(strategy_name, json.dumps(
            strategy_results[strategy_name]["depth_output"]["aggregate"],
            sort_keys=True,
        ), flush=True)

    output = {
        "contract": {
            "decision": "numerical feasibility of fixed-scale INT8 attention with SPU Softmax",
            "surrogate": (
                "real Depth Anything V2 ViT-S checkpoint; BF16-rounded Q/K/V and "
                "logits; signed symmetric INT8 Q/K/V; INT32 QK^T and P@V; "
                "PyTorch Softmax on BF16-rounded logits as an idealized SPU surrogate"
            ),
            "non_claims": [
                "Not a bit-exact model of the U250 SPU Softmax implementation.",
                "Does not test compiler scheduling, DMA, instruction encoding, or bitstream behavior.",
                "Does not quantize QKV/output projections, MLP, layer norms, or DPT head.",
                "Finite calibration and validation images do not establish dataset-wide depth accuracy.",
            ],
        },
        "configuration": {
            "input_shape": [1, 3, args.input_size, args.input_size],
            "tokens": (args.input_size // 14) ** 2 + 1,
            "heads": 6,
            "head_dimension": 64,
            "q_scale_in_model": 0.125,
            "calibration_images": [path.name for path in calibration_images],
            "validation_images": [path.name for path in validation_images],
            "seed": args.seed,
            "rounding": "torch.round (nearest, ties-to-even)",
            "int8_range_qkv": [-127, 127],
            "int8_range_probability": [0, 127],
        },
        "calibration": audit.calibration_json(),
        "tuned_probability_denominators_per_layer_head": {
            str(index): values for index, values in tuned_denominators.items()
        },
        "strategies": strategy_results,
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "image_sha256": {path.name: sha256_file(path)
                             for path in calibration_images + validation_images},
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    commit, dirty = git_state(REPO_ROOT)
    command = " ".join(sys.argv)
    manifest = {
        "schema_version": 1,
        "claim_id": "depth-anything-v2-fixed-int8-attention-feasibility",
        "repository": {"commit": commit, "dirty": dirty},
        "command": command,
        "environment": {
            "software": [
                f"Python {platform.python_version()}",
                f"PyTorch {torch.__version__}",
                f"OpenCV {cv2.__version__}",
            ],
            "hardware": platform.platform(),
        },
        "mathematics": {
            "assertion_tested": (
                "Fixed offline Q/K/V scales and a fixed Softmax-output INT8 scale "
                "can preserve Depth Anything V2 ViT-S outputs on held-out examples."
            ),
            "coefficient_domain": "BF16, signed INT8, and INT32 accumulation surrogate",
            "conventions": (
                "symmetric q=clip(round(x/s),-127,127); probabilities use 0..127; "
                "Q includes the model's 1/sqrt(64) factor; scales are BF16-rounded"
            ),
            "inputs": [str(path.resolve()) for path in calibration_images + validation_images],
            "bounds": {
                "calibration_images": args.calibration_count,
                "validation_images": args.validation_count,
                "input_size": args.input_size,
                "attention_layers": len(audit.layers),
                "probability_scale_denominators": [127, 255, 511, 1023, 2047],
            },
            "non_claims": output["contract"]["non_claims"],
        },
        "randomness": {
            "used": False,
            "generator": "PyTorch and NumPy seeds recorded defensively; image split deterministic",
            "seed": args.seed,
        },
        "run": {
            "started_at": started_at,
            "runtime_seconds": time.time() - started,
            "exit_status": 0,
        },
        "outputs": [{"path": str(args.output.resolve()), "sha256": sha256_file(args.output)}],
        "checks": [
            "Calibration and validation image sets are disjoint.",
            "All twelve ViT-S attention layers were instrumented.",
            "INT8 products use INT32 PyTorch matrix multiplication.",
            "Per-image and attention-internal errors are recorded.",
        ],
        "result": "See the strategy comparison in the raw JSON output.",
        "residual_risks": output["contract"]["non_claims"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
