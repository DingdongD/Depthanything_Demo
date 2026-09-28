#!/usr/bin/env python3
"""Calibrate every Depth Anything V2-S attention head for the U250 path.

The generated schema-v2 profile is consumed by both DS kernel codegen and the
host runtime.  Scales are static, per layer and per head.  Q already contains
the model's 1/sqrt(head_dim) factor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from depth_anything_v2.dinov2_layers.attention import Attention  # noqa: E402
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from ds_models.static_int8_attention import (  # noqa: E402
    HeadScale,
    StaticAttentionProfile,
    StaticInt8AttentionRuntime,
    bf16_round,
    plan_query_chunks,
    quantize_probability,
    quantize_symmetric,
)


MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}
ABS_PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.95, 99.99, 100.0)
SCALE_PERCENTILES = (99.0, 99.5, 99.9, 99.95, 99.99, 100.0)
PROBABILITY_DENOMINATORS = (127, 255, 511, 1023, 2047, 4095, 8191, 16383)


def square_preprocess(path: Path, size: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(((image - mean) / std).transpose(2, 0, 1)).unsqueeze(0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bf16_scalar(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).float())


@dataclass
class SampledDistribution:
    sample_limit: int
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    abs_maximum: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0

    def add(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1).cpu()
        self.count += flat.numel()
        self.total += float(flat.double().sum())
        self.total_sq += float((flat.double() ** 2).sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.abs_maximum = max(self.abs_maximum, float(flat.abs().max()))
        # Feed every image into the percentile sample.  When the bounded
        # reservoir is full, deterministic even re-sampling retains coverage
        # of both earlier and later calibration inputs.
        take = min(max(1, self.sample_limit // 8), flat.numel())
        indices = torch.linspace(0, flat.numel() - 1, take).long()
        sample = flat[indices].numpy().copy()
        self.samples.append(sample)
        values = np.concatenate(self.samples)
        if values.size > self.sample_limit:
            keep = np.linspace(0, values.size - 1, self.sample_limit).astype(np.int64)
            values = values[keep]
            self.samples = [values]
        self.sample_count = values.size

    def values(self) -> np.ndarray:
        if not self.samples:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self.samples)

    def result(self) -> dict:
        values = self.values()
        absolute = np.abs(values)
        mean = self.total / max(self.count, 1)
        variance = max(self.total_sq / max(self.count, 1) - mean * mean, 0.0)
        return {
            "count": self.count,
            "sample_count": int(values.size),
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            "abs_max": self.abs_maximum,
            "abs_percentiles": {
                f"p{p:g}": float(np.percentile(absolute, p)) for p in ABS_PERCENTILES
            },
        }


@dataclass
class ErrorSum:
    squared_error: float = 0.0
    squared_reference: float = 0.0
    absolute_error: float = 0.0
    count: int = 0
    max_abs: float = 0.0
    reference_candidate: float = 0.0
    candidate_squared: float = 0.0

    def add(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        diff = candidate.float() - reference.float()
        self.squared_error += float(torch.sum(diff * diff))
        self.squared_reference += float(torch.sum(reference.float() ** 2))
        self.absolute_error += float(torch.sum(diff.abs()))
        self.count += diff.numel()
        self.max_abs = max(self.max_abs, float(diff.abs().max()))
        self.reference_candidate += float(torch.sum(reference.float() * candidate.float()))
        self.candidate_squared += float(torch.sum(candidate.float() ** 2))

    def optimal_gain(self) -> float:
        return self.reference_candidate / max(self.candidate_squared, 1e-30)

    def compensated_squared_error(self) -> float:
        gain = self.optimal_gain()
        return max(
            self.squared_reference - 2 * gain * self.reference_candidate
            + gain * gain * self.candidate_squared,
            0.0,
        )

    def result(self) -> dict:
        gain = self.optimal_gain()
        return {
            "relative_l2": math.sqrt(self.squared_error / max(self.squared_reference, 1e-30)),
            "rmse": math.sqrt(self.squared_error / max(self.count, 1)),
            "mae": self.absolute_error / max(self.count, 1),
            "max_abs": self.max_abs,
            "optimal_output_gain": gain,
            "compensated_relative_l2": math.sqrt(
                self.compensated_squared_error() / max(self.squared_reference, 1e-30)
            ),
        }


def choose_scale(distribution: SampledDistribution) -> tuple[float, list[dict]]:
    values = distribution.values().astype(np.float32)
    absolute = np.abs(values)
    reference_norm = max(float(np.linalg.norm(values)), 1e-30)
    candidates = []
    for percentile in SCALE_PERCENTILES:
        threshold = (distribution.abs_maximum if percentile == 100.0
                     else float(np.percentile(absolute, percentile)))
        scale = bf16_scalar(max(threshold / 127.0, 2 ** -24))
        integer = np.clip(np.rint(values / scale), -127, 127)
        restored = integer * scale
        candidates.append({
            "percentile": percentile,
            "threshold": threshold,
            "scale_bf16": scale,
            "saturation_fraction": float(np.mean(np.abs(values) >= 126.5 * scale)),
            "relative_l2": float(np.linalg.norm(restored - values) / reference_norm),
            "rmse": float(np.sqrt(np.mean((restored - values) ** 2))),
        })
    selected = min(candidates, key=lambda item: item["relative_l2"])
    return float(selected["scale_bf16"]), candidates


class Calibrator:
    def __init__(self, model: torch.nn.Module, sample_limit: int):
        self.layers = [module for module in model.modules() if isinstance(module, Attention)]
        self.mode = "collect"
        self.distributions = {
            layer: {
                name: [SampledDistribution(sample_limit) for _ in range(module.num_heads)]
                for name in ("q", "k", "v", "logits", "probability")
            }
            for layer, module in enumerate(self.layers)
        }
        self.selected: dict[int, list[dict[str, float]]] = {}
        self.scale_candidates: dict[int, list[dict[str, list[dict]]]] = {}
        self.probability_errors = {
            layer: {
                denominator: [ErrorSum() for _ in range(module.num_heads)]
                for denominator in PROBABILITY_DENOMINATORS
            }
            for layer, module in enumerate(self.layers)
        }
        self.qk_errors = {
            layer: {
                (q_index, k_index): [ErrorSum() for _ in range(module.num_heads)]
                for q_index in range(len(SCALE_PERCENTILES))
                for k_index in range(len(SCALE_PERCENTILES))
            }
            for layer, module in enumerate(self.layers)
        }
        self.runtime_errors = {layer: [ErrorSum() for _ in range(module.num_heads)]
                               for layer, module in enumerate(self.layers)}
        for index, layer in enumerate(self.layers):
            layer.forward = types.MethodType(self._forward(index), layer)

    def _forward(self, index: int):
        owner = self

        def forward(layer: Attention, x: torch.Tensor) -> torch.Tensor:
            b, n, c = x.shape
            qkv = layer.qkv(x).reshape(b, n, 3, layer.num_heads, c // layer.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * layer.scale, qkv[1], qkv[2]
            if owner.mode == "collect":
                for head in range(layer.num_heads):
                    for name, value in (("q", q), ("k", k), ("v", v)):
                        owner.distributions[index][name][head].add(value[:, head])
                output = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
            elif owner.mode == "tune":
                output = owner._tune(index, q, k, v)
            elif owner.mode == "tune_qk":
                output = owner._tune_qk(index, q, k, v)
            elif owner.mode == "quantized":
                output = owner._quantized(index, q, k, v)
            else:
                output = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
            output = output.transpose(1, 2).reshape(b, n, c)
            return layer.proj_drop(layer.proj(output))

        return forward

    def select_qkv_scales(self) -> None:
        for layer, module in enumerate(self.layers):
            selected_heads = []
            candidate_heads = []
            for head in range(module.num_heads):
                selected = {}
                candidates = {}
                for name in ("q", "k", "v"):
                    selected[name], candidates[name] = choose_scale(
                        self.distributions[layer][name][head]
                    )
                selected_heads.append(selected)
                candidate_heads.append(candidates)
            self.selected[layer] = selected_heads
            self.scale_candidates[layer] = candidate_heads

    def _tune_qk(self, layer: int, q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor) -> torch.Tensor:
        reference_chunks = []
        for chunk in plan_query_chunks(q.shape[-2]):
            q_part = q[:, :, chunk.start:chunk.stop]
            reference_probability = torch.softmax(q_part @ k.transpose(-2, -1), dim=-1)
            reference_chunks.append(reference_probability @ v)
            tune_rows = min(32, q_part.shape[-2])
            row_index = torch.linspace(0, q_part.shape[-2] - 1, tune_rows).long()
            for head in range(q.shape[1]):
                q_sample = q_part[:, head, row_index]
                key = k[:, head]
                value = v[:, head]
                reference = reference_probability[:, head, row_index] @ value
                q_candidates = self.scale_candidates[layer][head]["q"]
                k_candidates = self.scale_candidates[layer][head]["k"]
                for q_index, q_item in enumerate(q_candidates):
                    qi = quantize_symmetric(q_sample, q_item["scale_bf16"]).to(torch.int32)
                    for k_index, k_item in enumerate(k_candidates):
                        ki = quantize_symmetric(key, k_item["scale_bf16"]).to(torch.int32)
                        logits = bf16_round(
                            (qi @ ki.transpose(-2, -1)).float()
                            * (q_item["scale_bf16"] * k_item["scale_bf16"])
                        )
                        candidate = torch.softmax(logits, dim=-1) @ value
                        self.qk_errors[layer][(q_index, k_index)][head].add(
                            reference, candidate
                        )
        return torch.cat(reference_chunks, dim=-2)

    def select_qk_scales(self) -> None:
        for layer, module in enumerate(self.layers):
            for head in range(module.num_heads):
                q_index, k_index = min(
                    self.qk_errors[layer],
                    key=lambda pair: self.qk_errors[layer][pair][head].squared_error,
                )
                self.selected[layer][head]["q"] = self.scale_candidates[layer][head]["q"][q_index]["scale_bf16"]
                self.selected[layer][head]["k"] = self.scale_candidates[layer][head]["k"][k_index]["scale_bf16"]
                self.selected[layer][head]["q_percentile"] = SCALE_PERCENTILES[q_index]
                self.selected[layer][head]["k_percentile"] = SCALE_PERCENTILES[k_index]

    def _tune(self, layer: int, q: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor) -> torch.Tensor:
        reference_chunks = []
        for chunk in plan_query_chunks(q.shape[-2]):
            q_part = q[:, :, chunk.start:chunk.stop]
            reference_logits = q_part @ k.transpose(-2, -1)
            reference_probability = torch.softmax(reference_logits, dim=-1)
            reference_chunks.append(reference_probability @ v)
            for head in range(q.shape[1]):
                self.distributions[layer]["logits"][head].add(reference_logits[:, head])
                self.distributions[layer]["probability"][head].add(reference_probability[:, head])

            # Probability-scale search uses 32 evenly distributed rows from
            # every hardware chunk.  Q/K/P distributions still cover all rows,
            # while this bounded objective keeps 12-layer calibration practical.
            tune_rows = min(32, q_part.shape[-2])
            row_index = torch.linspace(0, q_part.shape[-2] - 1, tune_rows).long()
            q_sample = q_part[:, :, row_index]
            ref_sample = reference_probability[:, :, row_index] @ v
            sq = torch.tensor([h["q"] for h in self.selected[layer]]).reshape(1, -1, 1, 1)
            sk = torch.tensor([h["k"] for h in self.selected[layer]]).reshape(1, -1, 1, 1)
            sv = torch.tensor([h["v"] for h in self.selected[layer]]).reshape(1, -1, 1, 1)
            qi = torch.round(bf16_round(q_sample) / sq).clamp(-127, 127).to(torch.int32)
            ki = torch.round(bf16_round(k) / sk).clamp(-127, 127).to(torch.int32)
            vi = torch.round(bf16_round(v) / sv).clamp(-127, 127).to(torch.int32)
            logits = bf16_round((qi @ ki.transpose(-2, -1)).float() * (sq * sk))
            probability = bf16_round(torch.softmax(logits, dim=-1))
            for denominator in PROBABILITY_DENOMINATORS:
                sp = bf16_scalar(1.0 / denominator)
                pi = quantize_probability(probability, sp).to(torch.int32)
                got = bf16_round((pi @ vi).float() * (sp * sv))
                for head in range(q.shape[1]):
                    self.probability_errors[layer][denominator][head].add(
                        ref_sample[:, head], got[:, head]
                    )
        return torch.cat(reference_chunks, dim=-2)

    def select_probability_scales(self) -> None:
        for layer, heads in self.selected.items():
            for head, values in enumerate(heads):
                denominator = min(
                    PROBABILITY_DENOMINATORS,
                    key=lambda d: self.probability_errors[layer][d][head].compensated_squared_error(),
                )
                values["probability_denominator"] = denominator
                values["probability"] = bf16_scalar(1.0 / denominator)
                values["av_output_gain"] = bf16_scalar(
                    self.probability_errors[layer][denominator][head].optimal_gain()
                )
                values["av_value_accumulator_scale"] = bf16_scalar(
                    values["v"] * values["av_output_gain"]
                )

    def _profile_document(self, tokens: int) -> dict:
        return {
            "schema_version": 2,
            "model": {
                "name": "depth_anything_v2_vits",
                "tokens": tokens,
                "heads": 6,
                "head_dimension": 64,
                "attention_layers": len(self.layers),
            },
            "layers": {
                str(layer): {
                    "heads": [
                        {"selected_scales_bf16": self.selected[layer][head]}
                        for head in range(module.num_heads)
                    ]
                }
                for layer, module in enumerate(self.layers)
            },
        }

    def _quantized(self, layer: int, q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
        runtime = StaticInt8AttentionRuntime(
            StaticAttentionProfile(self._profile_document(int(q.shape[-2])))
        )
        got = runtime.execute(layer, q, k, v)
        reference = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
        for head in range(q.shape[1]):
            self.runtime_errors[layer][head].add(reference[:, head], got[:, head])
        return got

    def layer_json(self) -> dict:
        result = {}
        for layer, module in enumerate(self.layers):
            heads = []
            for head in range(module.num_heads):
                selected = self.selected[layer][head]
                denominator = int(selected["probability_denominator"])
                heads.append({
                    "distributions": {
                        name: self.distributions[layer][name][head].result()
                        for name in ("q", "k", "v", "logits", "probability")
                    },
                    "scale_candidates": self.scale_candidates[layer][head],
                    "qk_attention_scale_candidates": {
                        f"q{SCALE_PERCENTILES[qi]:g}_k{SCALE_PERCENTILES[ki]:g}":
                            self.qk_errors[layer][(qi, ki)][head].result()
                        for qi in range(len(SCALE_PERCENTILES))
                        for ki in range(len(SCALE_PERCENTILES))
                    },
                    "probability_scale_candidates": {
                        str(d): self.probability_errors[layer][d][head].result()
                        for d in PROBABILITY_DENOMINATORS
                    },
                    "selected_scales_bf16": selected,
                    "validation_attention_output": self.runtime_errors[layer][head].result(),
                })
            result[str(layer)] = {"heads": heads}
        return result


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    error = ErrorSum()
    error.add(reference, candidate)
    result = error.result()
    flat_ref = reference.float().reshape(-1)
    flat_got = candidate.float().reshape(-1)
    result["cosine"] = float(torch.dot(flat_ref, flat_got) /
                             (torch.linalg.vector_norm(flat_ref)
                              * torch.linalg.vector_norm(flat_got)).clamp_min(1e-30))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "assets/examples")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--calibration-count", type=int, default=12)
    parser.add_argument("--tuning-count", type=int, default=4)
    parser.add_argument("--qk-tuning-count", type=int, default=2)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=131072)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    images = sorted(args.images.glob("*.jpg"))
    required = args.calibration_count + args.validation_count
    if len(images) < required:
        raise RuntimeError(f"need {required} disjoint images, found {len(images)}")
    calibration_images = images[:args.calibration_count]
    tuning_images = calibration_images[-min(args.tuning_count, len(calibration_images)):]
    qk_tuning_images = calibration_images[-min(args.qk_tuning_count, len(calibration_images)):]
    validation_images = images[-args.validation_count:]

    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    calibrator = Calibrator(model, args.sample_limit)

    with torch.inference_mode():
        for index, path in enumerate(calibration_images, 1):
            calibrator.mode = "collect"
            model(square_preprocess(path, args.input_size))
            print(f"collect {index}/{len(calibration_images)} {path.name}", flush=True)
    calibrator.select_qkv_scales()

    with torch.inference_mode():
        for index, path in enumerate(qk_tuning_images, 1):
            calibrator.mode = "tune_qk"
            model(square_preprocess(path, args.input_size))
            print(f"tune-qk {index}/{len(qk_tuning_images)} {path.name}", flush=True)
    calibrator.select_qk_scales()

    with torch.inference_mode():
        for index, path in enumerate(tuning_images, 1):
            calibrator.mode = "tune"
            model(square_preprocess(path, args.input_size))
            print(f"tune {index}/{len(tuning_images)} {path.name}", flush=True)
    calibrator.select_probability_scales()

    validation = {}
    with torch.inference_mode():
        for index, path in enumerate(validation_images, 1):
            value = square_preprocess(path, args.input_size)
            calibrator.mode = "reference"
            reference = model(value)
            calibrator.mode = "quantized"
            candidate = model(value)
            validation[path.name] = output_metrics(reference, candidate)
            print(f"validate {index}/{len(validation_images)} {path.name} "
                  f"rel_l2={validation[path.name]['relative_l2']:.6g}", flush=True)

    tokens = (args.input_size // 14) ** 2 + 1
    chunks = plan_query_chunks(tokens)
    output = {
        "schema_version": 2,
        "model": {
            "name": "depth_anything_v2_vits",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "input_shape": [1, 3, args.input_size, args.input_size],
            "tokens": tokens,
            "heads": 6,
            "head_dimension": 64,
            "attention_layers": len(calibrator.layers),
        },
        "calibration": {
            "images": [p.name for p in calibration_images],
            "tuning_images": [p.name for p in tuning_images],
            "qk_tuning_images": [p.name for p in qk_tuning_images],
            "validation_images": [p.name for p in validation_images],
            "sample_limit_per_layer_head_tensor": args.sample_limit,
            "scale_percentiles": list(SCALE_PERCENTILES),
            "probability_denominators": list(PROBABILITY_DENOMINATORS),
            "selection": (
                "V minimizes sampled tensor relative-L2; Q/K jointly minimize real "
                "attention-output SSE; probability scale then minimizes real attention-output "
                "SSE after Q/K/V quantization"
            ),
        },
        "runtime_contract": {
            "query_rows_limit": 256,
            "chunks": [c.__dict__ | {"valid_rows": c.valid_rows} for c in chunks],
            "calls_per_attention_layer": len(chunks) * 6,
            "calls_per_model": len(chunks) * 6 * len(calibrator.layers),
            "last_chunk_policy": "pad query rows 90->256 with zeros, crop output 256->90",
            "key_value_policy": f"reuse full {tokens}-token K^T and V for every query chunk",
        },
        "layers": calibrator.layer_json(),
        "validation_depth_output": validation,
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    # Re-open through the production loader so malformed profiles fail calibration.
    StaticAttentionProfile.load(args.output)
    print(f"profile={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
