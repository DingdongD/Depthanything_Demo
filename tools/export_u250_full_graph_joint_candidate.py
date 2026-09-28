#!/usr/bin/env python3
"""Export every kernel changed by one joint full-graph calibration proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import onnx
from onnx import external_data_helper, helper


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def export_scaled(source: Path, output: Path, op_type: str,
                  attribute: str, scale: float) -> list[str]:
    model = onnx.load(str(source), load_external_data=True)
    external_data_helper.convert_model_from_external_data(model)
    nodes = [node for node in model.graph.node if node.op_type == op_type]
    if not nodes:
        raise ValueError(f"{source}: no {op_type} nodes")
    for node in nodes:
        replace_attribute(node, attribute, [float(scale)])
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return [value.name for value in model.graph.input]


def attention_signature(block: dict) -> tuple[tuple[float, ...], ...]:
    """Return the effective per-head kernel parameters, not JSON encoding details."""
    attention = block["attention"]
    shared = attention.get("dual_range_probability")
    result = []
    for head, specification in enumerate(attention["heads"]):
        scales = specification["scales_bf16"]
        if shared is not None:
            probability = shared["heads"][str(head)]
            fine = float(probability["fine_step"])
            threshold = float(probability["threshold"])
            residual = float(probability["residual_step"])
        else:
            probability = scales["probability"]
            if not isinstance(probability, dict):
                raise ValueError(
                    "joint candidate requires complete per-head probability parameters"
                )
            fine = float(probability["fine"])
            threshold = float(probability["threshold"])
            residual = float(
                probability.get("residual", probability.get("residual_step"))
            )
        result.append((
            float(scales["q"]), float(scales["k"]), float(scales["v"]),
            fine, threshold, residual,
        ))
    return tuple(result)


def attention_changed(before: dict, after: dict) -> bool:
    old, new = attention_signature(before), attention_signature(after)
    if len(old) != len(new):
        return True
    return any(
        not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)
        for old_head, new_head in zip(old, new)
        for left, right in zip(old_head, new_head)
    )


def direct_source(roots: list[Path], name: str) -> Path:
    for root in roots:
        path = root / f"{name}.onnx"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{name}.onnx not found below: " + ", ".join(map(str, roots))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--proposed-contract", type=Path, required=True)
    parser.add_argument("--base-host-plan", type=Path, required=True)
    parser.add_argument("--proposed-host-plan", type=Path, required=True)
    parser.add_argument("--qkv-root", type=Path, action="append", required=True)
    parser.add_argument("--encoder-tail-root", type=Path, action="append", required=True)
    parser.add_argument("--fc1-root", type=Path, action="append", required=True)
    parser.add_argument("--decoder-root", type=Path, action="append", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"non-empty output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tools = Path(__file__).resolve().parent
    original = json.loads(args.base_contract.read_text())
    proposed = json.loads(args.proposed_contract.read_text())
    original_plan = json.loads(args.base_host_plan.read_text())
    proposed_plan = json.loads(args.proposed_host_plan.read_text())
    records = []

    def add_scaled(name: str, family: str, source: Path, scale: float,
                   op_type: str, attribute: str, codegen: int) -> None:
        output = args.output_root / "models" / f"{name}.onnx"
        inputs = export_scaled(source, output, op_type, attribute, scale)
        records.append({
            "kernel": name, "family": family, "source": str(source.resolve()),
            "source_sha256": sha256(source), "onnx": str(output.resolve()),
            "onnx_sha256": sha256(output), "input_scale": float(scale),
            "layouts": ",".join(f"{value}=BWC" for value in inputs),
            "codegen": codegen,
        })

    attention_layers = []
    for layer, (before, after) in enumerate(zip(
        original["encoder"], proposed["encoder"]
    )):
        if attention_changed(before, after):
            attention_layers.append(layer)
            directory = args.output_root / "attention" / f"l{layer:02d}"
            subprocess.run([
                sys.executable, str(tools / "export_u250_dual_range_attention_layer.py"),
                "--contract", str(args.proposed_contract), "--layer", str(layer),
                "--tokens", str(args.tokens), "--output-dir", str(directory),
            ], check=True)
            subprocess.run([
                sys.executable, str(tools / "fuse_u250_attention_heads.py"),
                "--attention-dir", str(directory), "--contract",
                str(args.proposed_contract), "--layer", str(layer),
                "--output-dir", str(directory),
            ], check=True)
            fused = directory / f"attention6_l{layer:02d}_a8_to_12xbf16.onnx"
            records.append({
                "kernel": f"attention6_l{layer:02d}", "family": "attention6",
                "onnx": str(fused.resolve()), "onnx_sha256": sha256(fused),
                "input_scale": None,
                "layouts": ",".join(f"input{index}=BWC" for index in range(24)),
                "codegen": 3,
            })

        if (before["qkv"]["input_quantization"]["scale"]
                != after["qkv"]["input_quantization"]["scale"]):
            name = after["qkv"]["kernel"]
            add_scaled(
                name, "qkv", direct_source(args.qkv_root, name),
                after["qkv"]["input_quantization"]["scale"],
                "MatMul", "A_scales", 3,
            )
        if (before["post_attention"]["input_quantization"]["scale"]
                != after["post_attention"]["input_quantization"]["scale"]):
            name = after["post_attention"]["kernel"]
            add_scaled(
                name, "post_attention", direct_source(args.encoder_tail_root, name),
                after["post_attention"]["input_quantization"]["scale"],
                "MatMul", "A_scales", 2,
            )
        before_mlp, after_mlp = before["mlp"], after["mlp"]
        if (before_mlp["fc1_input_quantization"]["scale"]
                != after_mlp["fc1_input_quantization"]["scale"]):
            for name in after_mlp["fc1_kernels"]:
                add_scaled(
                    name, "mlp_fc1", direct_source(args.fc1_root, name),
                    after_mlp["fc1_input_quantization"]["scale"],
                    "MatMul", "A_scales", 2,
                )
        if (before_mlp["fc2_input_quantization"]["scale"]
                != after_mlp["fc2_input_quantization"]["scale"]):
            name = after_mlp["fc2_kernel"]
            add_scaled(
                name, "mlp_fc2", direct_source(args.encoder_tail_root, name),
                after_mlp["fc2_input_quantization"]["scale"],
                "MatMul", "A_scales", 2,
            )

    for before, after in zip(
        original_plan["decoder_steps"], proposed_plan["decoder_steps"]
    ):
        if before.get("backend") != "npu" or before["input_scale"] == after["input_scale"]:
            continue
        for kernel in after["kernels"]:
            name = kernel["name"]
            source = direct_source(args.decoder_root, name)
            output = args.output_root / "models" / f"{name}.onnx"
            export_scaled(source, output, "Conv", "input_scales",
                          after["input_scale"])
            records.append({
                "kernel": name, "family": "decoder", "decoder_index": after["index"],
                "source": str(source.resolve()), "source_sha256": sha256(source),
                "onnx": str(output.resolve()), "onnx_sha256": sha256(output),
                "input_scale": float(after["input_scale"]),
                "layouts": "input0=BCHW", "codegen": 2,
            })

    names = [item["kernel"] for item in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate replacement kernel names")
    manifest = {
        "schema": "depthanything-u250-full-graph-joint-export-v1",
        "calibration_manifest_sha256": proposed["calibration"]["manifest_sha256"],
        "tokens": args.tokens,
        "attention_layers": attention_layers,
        "kernels": records,
        "kernel_count": len(records),
        "base_contract": str(args.base_contract.resolve()),
        "proposed_contract": str(args.proposed_contract.resolve()),
        "base_host_plan": str(args.base_host_plan.resolve()),
        "proposed_host_plan": str(args.proposed_host_plan.resolve()),
    }
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(output.resolve()), "kernels": len(records),
        "attention_layers": attention_layers,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
