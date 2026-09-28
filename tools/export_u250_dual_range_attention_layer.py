#!/usr/bin/env python3
"""Export two-chunk U250 attention with a fine plus residual A8 carrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DEFAULT_FINE_STEP = 1.0 / 16384.0
DEFAULT_RESIDUAL_STEP = 1.0 / 128.0
QUERY_ROWS = 256
HEAD_WIDTH = 64


def make_model(
    q_scale: float,
    k_scale: float,
    v_scale: float,
    fine_step: float = DEFAULT_FINE_STEP,
    residual_step: float = DEFAULT_RESIDUAL_STEP,
    tokens: int = 1370,
) -> onnx.ModelProto:
    """Build the CompletionFormer-v383 dual-range topology around QK."""
    if not all(value > 0.0 for value in (
        q_scale, k_scale, v_scale, fine_step, residual_step
    )):
        raise ValueError("Q/K/V and probability scales must be positive")
    threshold = 127.0 * fine_step
    if threshold >= 1.0:
        raise ValueError("fine carrier threshold must be below one")

    inputs = [
        helper.make_tensor_value_info(
            "input0", TensorProto.FLOAT, [1, QUERY_ROWS, HEAD_WIDTH]
        ),
        helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [1, QUERY_ROWS, HEAD_WIDTH]
        ),
        helper.make_tensor_value_info(
            "input2", TensorProto.FLOAT, [1, HEAD_WIDTH, tokens]
        ),
        helper.make_tensor_value_info(
            "input3", TensorProto.FLOAT, [1, tokens, HEAD_WIDTH]
        ),
    ]
    nodes = []
    initializers = [numpy_helper.from_array(
        np.asarray(-threshold, dtype=np.float32),
        name="dual_range_negative_threshold",
    )]
    outputs = []
    for index in range(2):
        prefix = f"/chunk{index}"
        logits = prefix + "/logits"
        fine_probability = prefix + "/fine_probability"
        fine_context = prefix + "/fine_context"
        residual_bf16 = prefix + "/residual_probability_bf16"
        residual_shifted = prefix + "/residual_probability_shifted"
        residual_probability = prefix + "/residual_probability"
        residual_context = prefix + "/residual_context"
        context = prefix + "/context"
        output = f"output{index}"
        nodes.extend([
            helper.make_node(
                "MatMul", [f"input{index}", "input2"], [logits],
                name=prefix + "/QK",
                A_bitdepth=8, A_scales=[q_scale],
                B_bitdepth=8, B_scales=[k_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
            helper.make_node(
                "Softmax", [logits], [fine_probability],
                name=prefix + "/DualRangeFineSoftmax", axis=-1,
                input_bitdepth=16, input_scale=-1.0, input_scales=[-1.0],
                output_bitdepth=8, output_scale=fine_step,
                output_scales=[fine_step],
            ),
            helper.make_node(
                "MatMul", [fine_probability, "input3"], [fine_context],
                name=prefix + "/DualRangeFineAV",
                A_bitdepth=8, A_scales=[fine_step],
                B_bitdepth=8, B_scales=[v_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
            helper.make_node(
                "Softmax", [logits], [residual_bf16],
                name=prefix + "/DualRangeResidualSoftmax", axis=-1,
                input_bitdepth=16, input_scale=-1.0, input_scales=[-1.0],
                output_bitdepth=16, output_scale=-1.0, output_scales=[-1.0],
            ),
            helper.make_node(
                "Add", [residual_bf16, "dual_range_negative_threshold"],
                [residual_shifted], name=prefix + "/DualRangeResidualSubtract",
                left_scale=-1.0, left_scales=[-1.0], left_bitdepth=16,
                input_scale=-1.0, input_scales=[-1.0], input_bitdepth=16,
                const_scale=1.0 / residual_step, const_bitdepth=16,
                output_scale=residual_step, output_scales=[residual_step],
                output_bitdepth=8,
            ),
            helper.make_node(
                "Relu", [residual_shifted], [residual_probability],
                name=prefix + "/DualRangeResidualRelu",
                input_scale=residual_step, input_scales=[residual_step],
                input_bitdepth=8,
                output_scale=residual_step, output_scales=[residual_step],
                output_bitdepth=8,
            ),
            helper.make_node(
                "MatMul", [residual_probability, "input3"],
                [residual_context], name=prefix + "/DualRangeResidualAV",
                A_bitdepth=8, A_scales=[residual_step],
                B_bitdepth=8, B_scales=[v_scale],
                output_bitdepth=16, output_scale=-1.0,
            ),
            helper.make_node(
                "Add", [fine_context, residual_context], [context],
                name=prefix + "/DualRangeContextAdd",
                left_scale=-1.0, left_scales=[-1.0], left_bitdepth=16,
                right_scale=-1.0, right_scales=[-1.0], right_bitdepth=16,
                input_scale=-1.0, input_scales=[-1.0], input_bitdepth=16,
                output_scale=-1.0, output_scales=[-1.0], output_bitdepth=16,
            ),
        ])
        shape = prefix + "/output_shape"
        initializers.append(numpy_helper.from_array(
            np.asarray([1, QUERY_ROWS, HEAD_WIDTH], dtype=np.int64), name=shape
        ))
        nodes.append(helper.make_node(
            "Reshape", [context, shape], [output],
            name=prefix + "/OutputReshape",
        ))
        outputs.append(helper.make_tensor_value_info(
            output, TensorProto.FLOAT, [1, QUERY_ROWS, HEAD_WIDTH]
        ))

    graph = helper.make_graph(
        nodes, "u250_attention_dual_range_2chunk", inputs, outputs, initializers
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--contract", type=Path)
    source.add_argument("--static-profile", type=Path,
                        help="schema-v2 static attention calibration profile")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--fine-step", type=float, default=DEFAULT_FINE_STEP)
    parser.add_argument("--tokens", type=int, default=1370,
                        help="number of ViT tokens including CLS")
    parser.add_argument(
        "--residual-step", type=float, default=DEFAULT_RESIDUAL_STEP
    )
    parser.add_argument(
        "--calibration", type=Path,
        help=("optional per-head calibration report from "
              "calibrate_u250_dual_range_attention.py"),
    )
    parser.add_argument(
        "--qkv-scale-overrides", type=Path,
        help="optional per-head Q/K/V scale mapping; AV gain must remain one",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 1:
        raise ValueError("tokens must include CLS and be greater than one")
    if args.contract is not None:
        contract = json.loads(args.contract.read_text())
        layer = contract["encoder"][args.layer]
    else:
        profile = json.loads(args.static_profile.read_text())
        if int(profile["model"]["tokens"]) != args.tokens:
            raise ValueError("static profile token count does not match --tokens")
        calibrated_heads = profile["layers"][str(args.layer)]["heads"]
        layer = {
            "attention": {
                "heads": [
                    {
                        "head": head,
                        "scales_bf16": {
                            "q": float(item["selected_scales_bf16"]["q"]),
                            "k": float(item["selected_scales_bf16"]["k"]),
                            "v": float(item["selected_scales_bf16"]["v"]),
                            "av_v": float(item["selected_scales_bf16"]["v"]),
                            "av_output_gain": 1.0,
                        },
                    }
                    for head, item in enumerate(calibrated_heads)
                ]
            }
        }
    calibrated = None
    if args.calibration is not None:
        calibrated = json.loads(args.calibration.read_text())
        if int(calibrated["layer"]) != args.layer:
            raise ValueError("calibration layer does not match --layer")
        if not calibrated.get("v_unchanged") or float(
            calibrated.get("av_output_gain", 0.0)
        ) != 1.0:
            raise ValueError("calibration violates unit-amplitude AV policy")
        calibrated = {
            int(item["head"]): item["selected"] for item in calibrated["heads"]
        }
    qkv_overrides = {}
    if args.qkv_scale_overrides is not None:
        override_report = json.loads(args.qkv_scale_overrides.read_text())
        if float(override_report.get("av_output_gain", 0.0)) != 1.0:
            raise ValueError("QKV scale overrides violate unit-amplitude AV policy")
        qkv_overrides = {
            int(head): values for head, values in override_report["heads"].items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = []
    host_heads = set(layer.get("host_attention_heads", []))
    for head, specification in enumerate(layer["attention"]["heads"]):
        scales = dict(specification["scales_bf16"])
        if head in qkv_overrides:
            scales.update({
                name: float(qkv_overrides[head][name])
                for name in ("q", "k", "v")
            })
            scales["av_v"] = scales["v"]
            scales["av_output_gain"] = 1.0
        if float(scales.get("av_output_gain", 1.0)) != 1.0:
            raise ValueError(f"head {head}: forbidden AV gain")
        if float(scales.get("av_v", scales["v"])) != float(scales["v"]):
            raise ValueError(f"head {head}: AV scale differs from V")
        if calibrated is not None and head in calibrated:
            parameters = calibrated[head]
        elif calibrated is not None and head in host_heads:
            probability = scales.get("probability", {})
            parameters = ({
                "fine_step": probability["fine"],
                "threshold": probability["threshold"],
                "residual_step": probability["residual"],
            } if isinstance(probability, dict) else {
                # This resident entry is ABI-only: the runtime executes host
                # FP32 attention for host heads and never dispatches its BIN.
                "fine_step": args.fine_step,
                "threshold": 127.0 * args.fine_step,
                "residual_step": args.residual_step,
            })
        elif calibrated is not None:
            raise ValueError(f"calibration is missing NPU head {head}")
        elif args.contract is not None:
            probability_contract = layer["attention"].get(
                "dual_range_probability"
            )
            if probability_contract is not None:
                item = probability_contract["heads"][str(head)]
                parameters = {
                    "fine_step": float(item["fine_step"]),
                    "threshold": float(item["threshold"]),
                    "residual_step": float(item["residual_step"]),
                }
            else:
                item = scales.get("probability")
                if not isinstance(item, dict):
                    # Legacy contracts expose only the old single-carrier
                    # probability scale.  Keep the explicit CLI defaults for
                    # backward-compatible probe generation; production joint
                    # contracts always provide the complete three parameters.
                    parameters = {
                        "fine_step": args.fine_step,
                        "threshold": 127.0 * args.fine_step,
                        "residual_step": args.residual_step,
                    }
                else:
                    parameters = {
                        "fine_step": float(item["fine"]),
                        "threshold": float(item["threshold"]),
                        "residual_step": float(
                            item.get("residual", item.get("residual_step"))
                        ),
                    }
        else:
            parameters = {
                "fine_step": args.fine_step,
                "threshold": 127.0 * args.fine_step,
                "residual_step": args.residual_step,
            }
        fine_step = float(parameters["fine_step"])
        residual_step = float(parameters["residual_step"])
        threshold = float(parameters["threshold"])
        if not np.isclose(threshold, 127.0 * fine_step, rtol=1e-6):
            raise ValueError(f"head {head}: threshold does not equal 127*fine_step")
        name = f"attention2_l{args.layer:02d}_h{head:02d}"
        output = args.output_dir / f"{name}.onnx"
        onnx.save(make_model(
            float(scales["q"]), float(scales["k"]), float(scales["v"]),
            fine_step, residual_step,
            tokens=args.tokens,
        ), output)
        kernels.append({
            "layer": args.layer,
            "head": head,
            "name": name,
            "onnx": output.name,
            "probability_quantization": "dual-range-a8",
            "probability_rounding": (
                "static-SPU fine plus v383 EPU residual quantization"
            ),
            "fine_step": fine_step,
            "threshold": threshold,
            "residual_step": residual_step,
            "calls_per_head": (args.tokens + 2 * QUERY_ROWS - 1) // (2 * QUERY_ROWS),
            "query_groups": [
                [2 * call, 2 * call + 1]
                for call in range((args.tokens + 2 * QUERY_ROWS - 1) // (2 * QUERY_ROWS))
            ],
            "scales_bf16": {
                "q": float(scales["q"]),
                "k": float(scales["k"]),
                "v": float(scales["v"]),
                "av_v": float(scales["v"]),
                "av_output_gain": 1.0,
                "probability": {
                    "fine": fine_step,
                    "threshold": threshold,
                    "residual": residual_step,
                },
            },
        })
        print(output)
    manifest = {
        "schema_version": 1,
        "strategy": (
            "CompletionFormer-v383 dual-range attention; two query chunks "
            "per launch; no host intermediate"
        ),
        "source_qualification": "completionformer-u250-v383",
        "layer": args.layer,
        "fine_step": kernels[0]["fine_step"] if len({item["fine_step"] for item in kernels}) == 1 else None,
        "threshold": kernels[0]["threshold"] if len({item["threshold"] for item in kernels}) == 1 else None,
        "residual_step": kernels[0]["residual_step"] if len({item["residual_step"] for item in kernels}) == 1 else None,
        "per_head_probability": {
            str(item["head"]): {
                "fine_step": item["fine_step"],
                "threshold": item["threshold"],
                "residual_step": item["residual_step"],
            }
            for item in kernels
        },
        "kernels": kernels,
        "kernels_total": len(kernels),
        "tokens": args.tokens,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
