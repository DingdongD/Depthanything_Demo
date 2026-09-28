#!/usr/bin/env python3
"""Export QK, Softmax and AV probes using real traced Attention tensors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def bf16_round(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def save_single_node(source: onnx.ModelProto, node: onnx.NodeProto, name: str,
                     inputs: list[tuple[str, list[int]]],
                     output: tuple[str, list[int]], path: Path) -> None:
    probe = copy.deepcopy(node)
    probe.name = name
    del probe.input[:]
    probe.input.extend(item[0] for item in inputs)
    del probe.output[:]
    probe.output.extend([output[0]])
    graph = helper.make_graph(
        [probe], "depthanything_" + name,
        [helper.make_tensor_value_info(key, TensorProto.FLOAT, shape)
         for key, shape in inputs],
        [helper.make_tensor_value_info(output[0], TensorProto.FLOAT, output[1])],
    )
    model = helper.make_model(
        graph, opset_imports=copy.deepcopy(source.opset_import),
        producer_name=source.producer_name, producer_version=source.producer_version,
    )
    model.ir_version = source.ir_version
    onnx.save(model, str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-model-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", action="append", required=True,
                        metavar="LAYER:HEAD:CHUNK")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with np.load(args.trace, allow_pickle=False) as trace:
        for specification in args.probe:
            layer, head, chunk = map(int, specification.split(":"))
            start = chunk * 256
            stop = min(start + 256, 1370)
            if start >= stop:
                raise ValueError(f"invalid query chunk {chunk}")
            source_path = args.attention_model_dir / f"attention2_l{layer:02d}_h{head:02d}.onnx"
            source = onnx.load(str(source_path), load_external_data=True)
            qk_node = next(node for node in source.graph.node
                           if node.name == "/chunk0/QK")
            softmax_node = next(node for node in source.graph.node
                                if node.name == "/chunk0/Softmax")
            av_node = next(node for node in source.graph.node
                           if node.name == "/chunk0/AV")
            prefix = f"attention_probe_l{layer:02d}_h{head:02d}_c{chunk}"
            save_single_node(
                source, qk_node, prefix + "_qk",
                [("input0", [1, 256, 64]), ("input1", [1, 64, 1370])],
                ("output0", [1, 256, 1370]),
                args.output_dir / (prefix + "_qk.onnx"),
            )
            save_single_node(
                source, softmax_node, prefix + "_softmax",
                [("input0", [1, 256, 1370])],
                ("output0", [1, 256, 1370]),
                args.output_dir / (prefix + "_softmax.onnx"),
            )
            save_single_node(
                source, av_node, prefix + "_av",
                [("input0", [1, 256, 1370]), ("input1", [1, 1370, 64])],
                ("output0", [1, 256, 64]),
                args.output_dir / (prefix + "_av.onnx"),
            )

            begin = head * 64
            end = begin + 64
            q = trace[f"q_l{layer:02d}"][0, 0, start:stop, begin:end]
            k = trace[f"k_l{layer:02d}"][0, 0, :, begin:end]
            v = trace[f"v_l{layer:02d}"][0, 0, :, begin:end]
            padded_q = np.zeros((1, 256, 64), np.int8)
            padded_q[0, :stop - start] = q
            kt = np.ascontiguousarray(k.T[None])
            value = np.ascontiguousarray(v[None])
            block = contract["encoder"][layer]
            q_scale = float(block["qkv"]["output_scales"]["q"])
            k_scale = float(block["qkv"]["output_scales"]["k"])
            v_scale = float(block["qkv"]["output_scales"]["v"])
            probability_scale = float(
                block["attention"]["heads"][head]["scales_bf16"]["probability"]
            )
            logits = bf16_round(
                (padded_q.astype(np.int32) @ kt.astype(np.int32)).astype(np.float32)
                * np.float32(q_scale * k_scale)
            )
            probability = bf16_round(softmax(logits))
            probability_code = np.clip(
                np.rint(probability / probability_scale), 0, 127
            ).astype(np.int8)
            context = bf16_round(
                (probability_code.astype(np.int32) @ value.astype(np.int32)).astype(np.float32)
                * np.float32(probability_scale * v_scale)
            )
            data_path = args.output_dir / (prefix + "_data.npz")
            np.savez_compressed(
                data_path, q=padded_q, kt=kt, v=value,
                logits_bf16=logits, probability_bf16=probability,
                probability_i8=probability_code, context_bf16=context,
            )
            records.append({
                "name": prefix, "layer": layer, "head": head, "chunk": chunk,
                "valid_rows": stop - start, "q_scale": q_scale,
                "k_scale": k_scale, "v_scale": v_scale,
                "probability_scale": probability_scale,
                "data": data_path.name,
                "kernels": {
                    stage: prefix + "_" + stage for stage in ("qk", "softmax", "av")
                },
            })
    manifest = {
        "schema_version": 1, "strategy": "instruction-level Attention stage probes",
        "probes": records,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "probes": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
