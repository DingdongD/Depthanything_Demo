#!/usr/bin/env python3
"""Generate and validate the host/NPU schedule for the resident U250 bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper


def attr(node: onnx.NodeProto, name: str) -> object:
    for value in node.attribute:
        if value.name == name:
            return helper.get_attribute_value(value)
    raise KeyError(f"{node.name}: {name}")


def qkv_input_scale(model_dir: Path, name: str) -> float:
    model = onnx.load(str((model_dir / (name + ".onnx")).resolve()),
                      load_external_data=False)
    scales = attr(model.graph.node[0], "A_scales")
    return float(scales[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--qkv-manifest", type=Path, required=True)
    parser.add_argument("--qkv-model-dir", type=Path, required=True)
    parser.add_argument("--attention-manifest", type=Path, required=True)
    parser.add_argument("--encoder-tail-manifest", type=Path, required=True)
    parser.add_argument("--gelu-manifest", type=Path,
                        help="replace host GELU and 256-wide FC1 slices with fused 64-wide Conv2DPWL")
    parser.add_argument("--decoder-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument(
        "--layernorm-kernel",
        help="resident BF16 SPU kernel for the normalization core; affine stays on host",
    )
    parser.add_argument(
        "--layernorm-affine-folded", action="store_true",
        help="gamma/beta are folded into QKV and FC1 weights",
    )
    parser.add_argument(
        "--decoder-layernorm-kernel",
        help="reuse a resident BF16 LayerNorm core for the four decoder captures",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layernorm_affine_folded and not args.layernorm_kernel:
        parser.error("--layernorm-affine-folded requires --layernorm-kernel")

    bank = json.loads(args.bank_manifest.read_text())
    resident = {item["name"] for item in bank["cases"]}
    qkv = json.loads(args.qkv_manifest.read_text())
    attention = json.loads(args.attention_manifest.read_text())
    tail = json.loads(args.encoder_tail_manifest.read_text())
    gelu = (json.loads(args.gelu_manifest.read_text())
            if args.gelu_manifest is not None else None)
    decoder = json.loads(args.decoder_manifest.read_text())
    source = onnx.load(str(args.model.resolve()), load_external_data=False)
    input_dims = source.graph.input[0].type.tensor_type.shape.dim
    image_shape = [int(dim.dim_value) for dim in input_dims]
    if len(image_shape) != 4 or any(value <= 0 for value in image_shape):
        raise ValueError(f"model must have static NCHW input, got {image_shape}")
    image_size = image_shape[2]
    if image_shape[3] != image_size or image_size % 14:
        raise ValueError(f"expected square /14 input, got {image_shape}")
    patch_grid = image_size // 14
    tokens = patch_grid * patch_grid + 1
    qkv_by_layer = {item["layer"]: item for item in qkv["kernels"]}
    attn_by_layer_head = {
        (item["layer"], item["head"]): item for item in attention["kernels"]
    }
    tail_by_layer = {item["layer"]: item for item in tail["layers"]}
    gelu_by_layer = ({layer: sorted(
        (item for item in gelu["kernels"] if item["layer"] == layer),
        key=lambda item: item["chunk"],
    ) for layer in range(12)} if gelu is not None else {})

    encoder_schedule = []
    for layer in range(12):
        qkv_item = qkv_by_layer[layer]
        invalid_qkv_scales = {
            branch: scale
            for branch, scale in qkv_item["output_scales"].items()
            if scale is not None and float(scale) <= 0.0
        }
        if invalid_qkv_scales:
            raise ValueError(
                f"layer {layer}: QKV must expose positive INT8 output scales; "
                f"got {invalid_qkv_scales}. This usually means BF16 QKV outputs "
                "were connected to the INT8 attention path."
            )
        tail_item = tail_by_layer[layer]
        heads = []
        for head in range(6):
            item = attn_by_layer_head[layer, head]
            calls = []
            for start in range(0, tokens, 512):
                q0_stop = min(start + 256, tokens)
                q1_start = min(start + 256, tokens)
                q1_stop = min(start + 512, tokens)
                call = {
                    "q0_rows": [start, q0_stop],
                    "q1_rows": [q1_start, q1_stop],
                }
                if q0_stop - start < 256:
                    call["q0_pad_to_rows"] = 256
                if q1_stop - q1_start < 256:
                    call["q1_pad_to_rows"] = 256
                calls.append(call)
            if len(calls) != int(item.get("calls_per_head", len(calls))):
                raise ValueError(f"layer {layer} head {head}: attention call count mismatch")
            heads.append({
                "head": head,
                "kernel": item["name"],
                "calls": calls,
                "scales_bf16": item["scales_bf16"],
            })
        encoder_schedule.append({
            "layer": layer,
            "input_shape": [1, tokens, 384],
            "host_norm1": {
                "op": "LayerNormalizationAffine", "precision": "FP32",
                **({"npu_core": args.layernorm_kernel,
                    "core_precision": "BF16",
                    "affine": ("folded" if args.layernorm_affine_folded else "host")}
                   if args.layernorm_kernel else {}),
            },
            "qkv": {
                "kernel": qkv_item["name"],
                "input_quantization": {"dtype": "INT8", "symmetric": True,
                                       "scale": qkv_input_scale(
                                           args.qkv_model_dir, qkv_item["name"])},
                "output_scales": qkv_item["output_scales"],
            },
            "attention": {
                "implementation": "fixed-scale INT8 QK + SPU softmax + INT8 AV",
                "heads": heads,
                "npu_calls": 6 * len(heads[0]["calls"]),
            },
            "post_attention": {
                "kernel": tail_item["post_attention"]["name"],
                "input_quantization": {
                    "dtype": "INT8", "symmetric": True,
                    "scale": tail_item["post_attention"]["attention_input_scale"],
                },
                "host_residual_input": True,
            },
            "host_norm2": {
                "op": "LayerNormalizationAffine", "precision": "FP32",
                **({"npu_core": args.layernorm_kernel,
                    "core_precision": "BF16",
                    "affine": ("folded" if args.layernorm_affine_folded else "host")}
                   if args.layernorm_kernel else {}),
            },
            "mlp": {
                "fc1_input_quantization": {
                    "dtype": "INT8", "symmetric": True,
                    "scale": tail_item["mlp"]["fc1_input_scale"],
                },
                "fc1_kernels": [item["name"] for item in
                                tail_item["mlp"]["safe_host_gelu"]["fc1_slices"]],
                "host_activation": {"op": "GELU", "precision": "FP32",
                                    "approximation": "none"},
                "fc2_input_quantization": {
                    "dtype": "INT8", "symmetric": True,
                    "scale": tail_item["mlp"]["fc2_input_scale"],
                },
                "fc2_kernel": tail_item["mlp"]["safe_host_gelu"]["fc2_name"],
                "host_residual_add": True,
            },
            "capture_for_decoder": layer in (2, 5, 8, 11),
        })
        if gelu is not None:
            native = gelu_by_layer[layer]
            if len(native) != 24:
                raise ValueError(f"layer {layer}: expected 24 FC1+GELU PWL kernels")
            input_steps = {float(item["input_step"]) for item in native}
            output_steps = {float(item["output_step"]) for item in native}
            if input_steps != {float(tail_item["mlp"]["fc1_input_scale"])}:
                raise ValueError(
                    f"layer {layer}: fused GELU input step does not match FC1 contract"
                )
            if output_steps != {float(tail_item["mlp"]["fc2_input_scale"])}:
                raise ValueError(
                    f"layer {layer}: fused GELU output step does not match FC2 contract"
                )
            encoder_schedule[-1]["mlp"].update({
                "fc1_kernels": [item["name"] for item in native],
                "fc1_input_layout": f"BCHW [1,384,1,{tokens}]",
                "host_activation": None,
                "npu_activation": {
                    "op": "GELU_PWL8", "backend": "CTC Conv2DPWL",
                    "precision": "INT8", "channel_slice": 64,
                    "output_step": native[0]["output_step"],
                },
            })

    conv_records = decoder["kernels"]
    conv_by_name = {item["source_node"]: item for item in conv_records}
    model_conv_names = [
        node.name for node in source.graph.node
        if node.op_type == "Conv" and node.name in conv_by_name
    ]
    if model_conv_names != [item["source_node"] for item in conv_records]:
        raise ValueError("decoder Conv manifest does not match source graph order")
    start = next(index for index, node in enumerate(source.graph.node)
                 if node.name == "/norm/LayerNormalization")
    decoder_schedule = []
    host_nodes = []

    def flush_host() -> None:
        nonlocal host_nodes
        if host_nodes:
            decoder_schedule.append({"backend": "host", "nodes": host_nodes})
            host_nodes = []

    decoder_calls = 0
    for node in source.graph.node[start:]:
        record = conv_by_name.get(node.name)
        if record is None:
            if (args.decoder_layernorm_kernel
                    and node.op_type == "LayerNormalization"):
                flush_host()
                decoder_schedule.append({
                    "backend": "npu_layernorm",
                    "source_node": node.name,
                    "input_tensor": node.input[0],
                    "scale_tensor": node.input[1],
                    "bias_tensor": node.input[2],
                    "output_tensor": node.output[0],
                    "kernel": args.decoder_layernorm_kernel,
                    "core_precision": "BF16",
                    "affine": "host_fp32",
                })
                decoder_calls += 1
                continue
            host_nodes.append({"name": node.name, "op_type": node.op_type,
                               "inputs": list(node.input), "outputs": list(node.output)})
            continue
        flush_host()
        index = int(record["index"])
        if "safe_channel_slices" in record:
            kernels = record["safe_channel_slices"]
            calls = len(kernels)
            channel_sliced = True
        elif "safe_tile_variants" not in record:
            kernels = [{"name": record["name"], "position": "full"}]
            calls = 1
            channel_sliced = False
        else:
            kernels = record["safe_tile_variants"]
            calls = int(record["row_tiles"])
            channel_sliced = False
        decoder_calls += calls
        decoder_schedule.append({
            "backend": "npu",
            "source_node": node.name,
            "input_tensor": node.input[0],
            "output_tensor": node.output[0],
            "input_quantization": {"dtype": "INT8", "symmetric": True,
                                   "scale": record["input_scale"]},
            "output_dtype": "BF16",
            "input_shape": record["input_shape"],
            "output_shape": record["output_shape"],
            "kernels": kernels,
            "row_tiles": int(record.get("row_tiles", 1)) if not channel_sliced else 1,
            "channel_sliced": channel_sliced,
            "tile_output_rows": record.get("tile_output_rows"),
            "halo_rows": 1 if not channel_sliced and len(kernels) == 3 else 0,
        })
    flush_host()

    referenced = set()
    frontend = None
    frontend_calls = 0
    frontend_patch_precision = None
    if args.patch_manifest is not None:
        patch = json.loads(args.patch_manifest.read_text())
        frontend_patch_precision = patch["input_precision"]
        patch_kernels = [item["name"] for item in patch["kernels"]]
        referenced.update(patch_kernels)
        frontend_calls = len(patch_kernels)
        frontend = {
            "input_shape": image_shape,
            "patchify": {
                "backend": "host", "op": "exact pixel rearrangement",
                "output_shape": [1, 588, patch_grid, patch_grid],
            },
            "patch_projection": {
                "backend": "npu", "input_dtype": patch["input_precision"],
                "weight_dtype": "INT8", "output_dtype": "BF16",
                "input_quantization": ({
                    "dtype": "INT8", "symmetric": True,
                    "scale": patch["input_scale"],
                } if patch["input_precision"] == "INT8" else None),
                "kernels": patch_kernels,
            },
            "token_assembly": {
                "backend": "host", "ops": ["Transpose", "Reshape", "Concat", "Add"],
                "output_shape": [1, tokens, 384],
            },
        }
    for block in encoder_schedule:
        for key in ("host_norm1", "host_norm2"):
            if "npu_core" in block[key]:
                referenced.add(block[key]["npu_core"])
        referenced.add(block["qkv"]["kernel"])
        referenced.update(head["kernel"] for head in block["attention"]["heads"])
        referenced.add(block["post_attention"]["kernel"])
        referenced.update(block["mlp"]["fc1_kernels"])
        referenced.add(block["mlp"]["fc2_kernel"])
    for step in decoder_schedule:
        if step["backend"] == "npu":
            referenced.update(item["name"] for item in step["kernels"])
        elif step["backend"] == "npu_layernorm":
            referenced.add(step["kernel"])
    missing = sorted(referenced - resident)
    unused = sorted(resident - referenced)
    if missing or unused:
        raise ValueError(f"resident coverage mismatch: missing={missing}, unused={unused}")

    attention_calls = sum(
        block["attention"]["npu_calls"] for block in encoder_schedule
    )
    encoder_other_calls = 12 + 12 + (288 if gelu is not None else 72) + 12 + (24 if args.layernorm_kernel else 0)
    contract = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "bank": {
            "file": bank["bank_file"], "bytes": bank["bank_size_bytes"],
            "sha256": bank["bank_sha256"], "resident_kernels": len(resident),
            "static_h2c_writes": bank["static_h2c_writes"],
            "shared_fm_workspace_bytes": bank["shared_fm_workspace_bytes"],
            "required_fm_io_bytes": bank["required_fm_io_bytes"],
        },
        "precision_policy": {
            "npu": ((frontend_patch_precision + "xB8 patch Conv; ")
                    if frontend_patch_precision else "") +
                   "A8xB8 MatMul/Conv; BF16 outputs; SPU softmax",
            "host": (("FP32 ReLU/Add/Resize/DepthToSpace and glue"
                       if args.layernorm_affine_folded else
                       "FP32 LayerNorm affine/ReLU/Add/Resize/DepthToSpace and glue"
                       if args.layernorm_kernel else
                       "FP32 LayerNorm/ReLU/Add/Resize/DepthToSpace and glue")
                      if gelu is not None else
                      ("FP32 GELU/ReLU/Add/Resize/DepthToSpace and glue"
                       if args.layernorm_affine_folded else
                       "FP32 LayerNorm affine/GELU/ReLU/Add/Resize/DepthToSpace and glue"
                       if args.layernorm_kernel else
                       "FP32 LayerNorm/GELU/ReLU/Add/Resize/DepthToSpace and glue")),
            "reason": (("SPU runs the BF16 LayerNorm normalization core; gamma/beta are folded into Linear weights; "
                        if args.layernorm_affine_folded else
                        "SPU runs the BF16 LayerNorm normalization core; learned affine remains FP32; ")
                       if args.layernorm_kernel else "") +
                      ("GELU uses board-qualified INT8 CTC PWL8; HighResLut BF16 remains disabled"
                       if gelu is not None else
                       "BF16 HighResLut and BF16 MatMul remain unsafe on the current bitstream"),
        },
        "encoder": encoder_schedule,
        "decoder": decoder_schedule,
        "execution_totals": {
            "resident_kernel_variants": len(resident),
            "encoder_npu_calls": encoder_other_calls + attention_calls,
            "decoder_npu_calls": decoder_calls,
            "frontend_npu_calls": frontend_calls,
            "npu_calls_per_inference": frontend_calls + encoder_other_calls + attention_calls + decoder_calls,
            "weight_bank_loads_per_process": 1,
            "static_h2c_writes_per_process": 2,
        },
    }
    if frontend is not None:
        contract["frontend"] = frontend
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "resident_kernels": len(resident),
        "decoder_steps": len(decoder_schedule), "decoder_npu_calls": decoder_calls,
        "npu_calls_per_inference": contract["execution_totals"]["npu_calls_per_inference"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
