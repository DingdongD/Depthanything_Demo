#!/usr/bin/env python3
"""Calibrate decoder A8 scales on balanced NYU-train and DA-2K inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import cv2
import h5py
import numpy as np
import torch

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


def decoder_module_name(node_name: str) -> str:
    prefix = "/depth_head/"
    suffix = "/Conv"
    relative = node_name[len(prefix):-len(suffix)].replace("/", ".")
    if relative in ("resize_layers.0.conv", "resize_layers.1.conv"):
        relative = relative.removesuffix(".conv")
    if relative.startswith("output_conv2.output_conv2."):
        relative = relative.removeprefix("output_conv2.")
    if relative.startswith(("layer", "refinenet", "output_conv")):
        relative = "scratch." + relative
    return "depth_head." + relative


def preprocess_nyu(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as source:
        rgb = source["rgb"][:]
    image = cv2.resize(
        rgb.transpose(1, 2, 0), (518, 518), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32) / np.float32(255.0)
    image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count > len(paths):
        raise ValueError(f"requested {count} samples from only {len(paths)} paths")
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def tensor_sample(value: torch.Tensor, count: int, phase: int) -> np.ndarray:
    flat = value.detach().float().cpu().numpy().reshape(-1)
    if flat.size <= count:
        return flat.copy()
    stride = max(flat.size // count, 1)
    offset = phase % stride
    return flat[offset::stride][:count].copy()


def metrics(values: np.ndarray, scale: float) -> dict[str, float]:
    code = np.clip(np.rint(values / np.float32(scale)), -128, 127)
    error = code * np.float32(scale) - values
    return {
        "relative_l2": float(
            np.linalg.norm(error.astype(np.float64))
            / max(np.linalg.norm(values.astype(np.float64)), 1.0e-30)
        ),
        "clipping_fraction": float(np.mean(
            (values < -128.0 * scale) | (values > 127.0 * scale)
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyu-root", type=Path, required=True)
    parser.add_argument("--da2k-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-convs", default="10,11,12,13,18,21,22,23,26,27,28,29")
    parser.add_argument("--samples-per-domain", type=int, default=32)
    parser.add_argument("--values-per-layer-sample", type=int, default=16384)
    parser.add_argument("--candidate-count", type=int, default=49)
    parser.add_argument("--minimum-factor", type=float, default=0.5)
    parser.add_argument("--maximum-factor", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    targets = sorted({int(value) for value in args.target_convs.split(",")})
    if not targets or any(not 0 <= value <= 31 for value in targets):
        parser.error("--target-convs must contain decoder Conv indices in [0,31]")
    plan = json.loads(args.host_plan.read_text())
    steps = {
        int(step["index"]): step for step in plan["decoder_steps"]
        if step["backend"] == "npu"
    }
    modules = dict()
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    ))
    model.eval().to(args.device)
    named_modules = dict(model.named_modules())
    for index in targets:
        modules[index] = named_modules[decoder_module_name(steps[index]["name"])]

    nyu_paths = evenly_spaced(sorted(args.nyu_root.glob("*.h5")), args.samples_per_domain)
    da2k_paths = evenly_spaced(sorted(args.da2k_root.glob("*/*.npy")), args.samples_per_domain)
    samples: dict[str, dict[int, list[np.ndarray]]] = {
        domain: {index: [] for index in targets} for domain in ("nyu", "da2k")
    }
    active_domain = ""
    active_phase = 0
    handles = []
    for index, module in modules.items():
        def capture(_module, inputs, *, conv_index=index):
            samples[active_domain][conv_index].append(tensor_sample(
                inputs[0], args.values_per_layer_sample,
                active_phase * 131 + conv_index * 17,
            ))
        handles.append(module.register_forward_pre_hook(capture))

    records = {"nyu": [], "da2k": []}
    with torch.inference_mode():
        for active_domain, paths in (("nyu", nyu_paths), ("da2k", da2k_paths)):
            for active_phase, path in enumerate(paths):
                value = (preprocess_nyu(path) if active_domain == "nyu" else
                         np.load(path, allow_pickle=False).astype(np.float32))
                if value.shape != (1, 3, 518, 518):
                    raise ValueError(f"unexpected input shape {value.shape}: {path}")
                model(torch.from_numpy(value).to(args.device))
                records[active_domain].append(str(path.resolve()))
                print(json.dumps({
                    "domain": active_domain, "position": active_phase + 1,
                    "total": len(paths), "sample": path.stem,
                }), flush=True)
    for handle in handles:
        handle.remove()

    layers = []
    for index in targets:
        current = float(steps[index]["input_scale"])
        scales = sorted(set(np.geomspace(
            current * args.minimum_factor,
            current * args.maximum_factor,
            args.candidate_count,
        ).tolist() + [current]))
        split_values = {}
        for domain in ("nyu", "da2k"):
            domain_values = samples[domain][index]
            split_values[domain] = {
                "training": np.concatenate([
                    value for position, value in enumerate(domain_values)
                    if position % 4 != 3
                ]),
                "validation": np.concatenate([
                    value for position, value in enumerate(domain_values)
                    if position % 4 == 3
                ]),
            }
        candidates = []
        for scale in scales:
            result = {
                domain: {
                    split: metrics(values, scale)
                    for split, values in split_values[domain].items()
                } for domain in ("nyu", "da2k")
            }
            result["balanced_training_objective"] = float(
                0.5 * (result["nyu"]["training"]["relative_l2"]
                       + result["da2k"]["training"]["relative_l2"])
            )
            candidates.append({"scale": float(scale), **result})
        selected = min(candidates, key=lambda item: item["balanced_training_objective"])
        current_result = next(item for item in candidates if item["scale"] == current)
        layers.append({
            "index": index, "node": steps[index]["name"],
            "current_scale": current, "selected": selected,
            "current": current_result, "candidates": candidates,
        })
        print(json.dumps({
            "index": index, "current_scale": current,
            "selected_scale": selected["scale"],
            "objective": selected["balanced_training_objective"],
            "nyu_validation": selected["nyu"]["validation"],
            "da2k_validation": selected["da2k"]["validation"],
        }, sort_keys=True), flush=True)

    report = {
        "schema": "depthanything-u250-mixed-decoder-calibration-v1",
        "selection": "minimum equal-domain mean training relative-L2",
        "split": "within each domain, every fourth selected sample is validation",
        "sampling": {
            "samples_per_domain": args.samples_per_domain,
            "values_per_layer_sample": args.values_per_layer_sample,
            "candidate_count": args.candidate_count,
            "minimum_factor": args.minimum_factor,
            "maximum_factor": args.maximum_factor,
        },
        "inputs": records, "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
