#!/usr/bin/env python3
"""Collect streaming FP32 activation histograms and teacher depths on DA-2K."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from depth_anything_v2.dpt import DepthAnythingV2


LOG_MIN = -24.0
LOG_MAX = 16.0
BINS = 2048


class ActivationHistogram:
    def __init__(self) -> None:
        self.count = 0
        self.zero_count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.absolute_maximum = 0.0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.histogram = np.zeros(BINS, dtype=np.int64)

    def update(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        finite = value[torch.isfinite(value)]
        if not finite.numel():
            return
        absolute = finite.abs()
        self.count += int(finite.numel())
        self.zero_count += int(torch.count_nonzero(absolute == 0).item())
        self.minimum = min(self.minimum, float(finite.min().item()))
        self.maximum = max(self.maximum, float(finite.max().item()))
        self.absolute_maximum = max(self.absolute_maximum, float(absolute.max().item()))
        self.sum += float(finite.double().sum().item())
        self.sum_squares += float((finite.double() * finite.double()).sum().item())
        nonzero = absolute[absolute > 0]
        if nonzero.numel():
            logs = torch.log2(nonzero).clamp(LOG_MIN, LOG_MAX)
            hist = torch.histc(logs, bins=BINS, min=LOG_MIN, max=LOG_MAX)
            self.histogram += hist.to(device="cpu", dtype=torch.int64).numpy()

    def quantile(self, probability: float) -> float:
        if not 0.0 <= probability <= 1.0 or not self.count:
            return 0.0
        rank = int(np.ceil(probability * self.count))
        if rank <= self.zero_count:
            return 0.0
        cumulative = np.cumsum(self.histogram)
        index = int(np.searchsorted(cumulative, rank - self.zero_count, side="left"))
        index = min(index, BINS - 1)
        return float(2.0 ** (LOG_MIN + (index + 0.5) * (LOG_MAX - LOG_MIN) / BINS))

    def summary(self) -> dict:
        mean = self.sum / self.count if self.count else 0.0
        variance = max(0.0, self.sum_squares / self.count - mean * mean) if self.count else 0.0
        quantiles = {key: self.quantile(value) for key, value in (
            ("p99", 0.99), ("p999", 0.999), ("p9999", 0.9999),
            ("p99999", 0.99999),
        )}
        return {
            "count": self.count, "zero_count": self.zero_count,
            "min": self.minimum, "max": self.maximum,
            "abs_max": self.absolute_maximum, "mean": mean,
            "std": variance ** 0.5, **quantiles,
            "symmetric_int8_scales": {
                key: value / 127.0 for key, value in {
                    **quantiles, "max": self.absolute_maximum,
                }.items()
            },
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--splits", default="calibration,tuning,holdout")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--teacher-only", action="store_true",
        help="save FP32 teacher depths without collecting activation histograms",
    )
    parser.add_argument("--save-inputs", action="store_true",
                        help="save normalized square inputs for board validation")
    parser.add_argument("--teacher-dtype", choices=("float16", "float32"),
                        default="float16")
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    requested = [item for item in args.splits.split(",") if item]
    entries = [(split, item) for split in requested
               for item in manifest["splits"][split]]
    if args.max_samples is not None:
        entries = entries[:args.max_samples]
    if not entries:
        raise ValueError("no DA-2K samples selected")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    statistics: dict[str, ActivationHistogram] = {}

    def record(key: str, value: torch.Tensor) -> None:
        statistics.setdefault(key, ActivationHistogram()).update(value)

    handles = []
    observed = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.LayerNorm, nn.GELU)
    for name, module in model.named_modules():
        if not args.teacher_only and isinstance(module, observed):
            def hook(_module, inputs, output, key=name):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    record(key + ".input", inputs[0])
                if isinstance(output, torch.Tensor):
                    record(key + ".output", output)
            handles.append(module.register_forward_hook(hook))

    for layer, block in enumerate(model.pretrained.blocks):
        if args.teacher_only:
            break
        def qkv_hook(_module, _inputs, output, layer_index=layer):
            batch, tokens, _ = output.shape
            values = output.reshape(batch, tokens, 3, 6, 64).permute(2, 0, 3, 1, 4)
            for kind, tensor in zip(("q", "k", "v"), values):
                if kind == "q":
                    tensor = tensor * (64.0 ** -0.5)
                for head in range(6):
                    record(f"encoder.block{layer_index:02d}.{kind}.head{head}", tensor[:, head])
        handles.append(block.attn.qkv.register_forward_hook(qkv_hook))

        def attention_input_hook(_module, inputs, layer_index=layer):
            value = inputs[0]
            for head in range(6):
                record(f"encoder.block{layer_index:02d}.attention.head{head}",
                       value[..., head * 64:(head + 1) * 64])
        handles.append(block.attn.proj.register_forward_pre_hook(attention_input_hook))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher_root = args.output_dir / "teacher_depth"
    started = time.time()
    processed = 0
    with torch.inference_mode():
        for begin in range(0, len(entries), args.batch_size):
            batch_entries = entries[begin:begin + args.batch_size]
            arrays = [preprocess(args.dataset_root / item[1]["path"])
                      for item in batch_entries]
            tensor = torch.from_numpy(np.stack(arrays)).to(device)
            if not args.teacher_only:
                record("model.input", tensor)
            depth = model(tensor).cpu().numpy()
            for (split, item), value, normalized in zip(
                    batch_entries, depth, arrays):
                target = teacher_root / split / f"{item['sample_id']}.npy"
                target.parent.mkdir(parents=True, exist_ok=True)
                teacher_dtype = np.float16 if args.teacher_dtype == "float16" else np.float32
                np.save(target, value.astype(teacher_dtype), allow_pickle=False)
                if args.save_inputs:
                    input_path = (args.output_dir / "inputs" / split
                                  / f"{item['sample_id']}.npy")
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(input_path, normalized[None], allow_pickle=False)
            processed += len(batch_entries)
            if processed % 32 == 0 or processed == len(entries):
                print(json.dumps({"processed": processed, "total": len(entries),
                                  "seconds": time.time() - started}), flush=True)
    for handle in handles:
        handle.remove()

    keys = sorted(statistics)
    histogram_path = args.output_dir / "activation_histograms.npz"
    np.savez_compressed(histogram_path, **{
        f"histogram_{index:04d}": statistics[key].histogram
        for index, key in enumerate(keys)
    })
    report = {
        "schema": "depthanything-da2k-fp32-calibration-v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "device": str(device), "batch_size": args.batch_size,
        "samples": processed, "splits": requested,
        "teacher_only": args.teacher_only,
        "teacher_dtype": args.teacher_dtype,
        "saved_inputs": args.save_inputs,
        "elapsed_seconds": time.time() - started,
        "histogram_contract": {"domain": "log2(abs(x))", "minimum": LOG_MIN,
                               "maximum": LOG_MAX, "bins": BINS,
                               "file": histogram_path.name},
        "activations": {
            key: {"histogram_key": f"histogram_{index:04d}",
                  **statistics[key].summary()}
            for index, key in enumerate(keys)
        },
    }
    (args.output_dir / "activation_scales.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"samples": processed, "activations": len(keys),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "output": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
