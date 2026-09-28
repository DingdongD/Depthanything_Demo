#!/usr/bin/env python3
"""Prepare and calibrate the U250 token-tail A8xB8 model.

The script has three deliberately separate stages:

* ``extract`` cuts the full tail at the real post-position-embedding token
  tensor.  Patch embedding and token preparation remain on the host.
* ``profile`` exposes every constant-RHS encoder MatMul input and every
  decoder Conv input, then records their real activation distributions from
  one or more representative inferences.  Mixed-domain input lists can select
  scales by equal-domain quantization error instead of allowing the larger
  dataset to dominate.
* ``apply`` attaches symmetric INT8 activation scales and per-output-channel
  INT8 weight scales to the corresponding nodes in a DS-quantized model.

Dynamic QK/AV MatMuls are intentionally excluded; their scales are bound by
``apply_static_attention_profile_to_onnx.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


QUANTIZATION_ATTRIBUTES = {
    "A_bitdepth",
    "A_scale",
    "A_scales",
    "B_bitdepth",
    "B_scale",
    "B_scales",
    "B_quant_dim",
    "input_bitdepth",
    "input_scale",
    "input_scales",
    "weight_bitdepth",
    "weight_ch_scales",
    "output_bitdepth",
    "output_scale",
    "output_scales",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_external(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        raise FileExistsError(f"refusing to overwrite {data_path}")
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=1024,
    )


def extract_tail(
    source: Path,
    output: Path,
    boundary: str,
    output_name: str,
    input_shape: list[int],
    output_shape: list[int] | None = None,
) -> dict:
    model = onnx.load(str(source.resolve()), load_external_data=True)
    producers = {
        tensor: node
        for node in model.graph.node
        for tensor in node.output
        if tensor
    }
    initializers = {value.name: value for value in model.graph.initializer}
    source_graph_inputs = {value.name for value in model.graph.input}
    if boundary not in producers and boundary not in source_graph_inputs:
        raise ValueError(f"boundary tensor has no producer: {boundary!r}")
    source_graph_outputs = {value.name for value in model.graph.output}
    if output_name not in producers and output_name not in source_graph_outputs:
        raise ValueError(f"output tensor has no producer: {output_name!r}")

    needed_nodes: set[int] = set()
    pending = [output_name]
    while pending:
        tensor = pending.pop()
        if tensor == boundary or tensor in initializers:
            continue
        node = producers.get(tensor)
        if node is None:
            raise ValueError(f"cannot resolve producer for tensor {tensor!r}")
        node_id = id(node)
        if node_id in needed_nodes:
            continue
        needed_nodes.add(node_id)
        pending.extend(name for name in node.input if name)

    nodes = [copy.deepcopy(node) for node in model.graph.node if id(node) in needed_nodes]
    for node in nodes:
        for index, name in enumerate(node.input):
            if name == boundary:
                node.input[index] = "input0"
    used_initializers = {name for node in nodes for name in node.input}
    kept_initializers = [
        copy.deepcopy(value)
        for value in model.graph.initializer
        if value.name in used_initializers
    ]
    value_info_by_name = {value.name: value for value in model.graph.value_info}
    kept_value_info = [
        copy.deepcopy(value_info_by_name[name])
        for name in {name for node in nodes for name in (*node.input, *node.output)}
        if name in value_info_by_name and name not in {boundary, "input0", output_name}
    ]
    if output_name in source_graph_outputs:
        graph_output = copy.deepcopy(
            next(value for value in model.graph.output if value.name == output_name)
        )
    else:
        graph_output = helper.make_tensor_value_info(
            output_name, TensorProto.FLOAT, output_shape
        )
    graph = helper.make_graph(
        nodes,
        model.graph.name + "_from_tokens",
        [helper.make_tensor_value_info("input0", TensorProto.FLOAT, input_shape)],
        [graph_output],
        initializer=kept_initializers,
        value_info=kept_value_info,
    )
    extracted = helper.make_model(
        graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        opset_imports=copy.deepcopy(model.opset_import),
        ir_version=model.ir_version,
    )
    extracted.metadata_props.extend(copy.deepcopy(model.metadata_props))
    save_external(extracted, output)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "boundary": boundary,
        "graph_input": "input0",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "nodes": len(nodes),
        "initializers": len(kept_initializers),
    }


def selected_operators(model: onnx.ModelProto) -> list[dict]:
    initializers = {value.name for value in model.graph.initializer}
    selected = []
    for node in model.graph.node:
        if (
            node.op_type == "MatMul"
            and node.name.startswith("/blocks.")
            and len(node.input) == 2
            and node.input[1] in initializers
        ):
            selected.append({
                "kind": "encoder_linear",
                "node": node.name,
                "activation": node.input[0],
                "weight": node.input[1],
            })
        elif (
            node.op_type == "Conv"
            and node.name.startswith("/depth_head/")
            and len(node.input) >= 2
            and node.input[1] in initializers
        ):
            selected.append({
                "kind": "decoder_conv",
                "node": node.name,
                "activation": node.input[0],
                "weight": node.input[1],
            })
    return selected


def tensor_statistics(array: np.ndarray, scale_percentile: float = 100.0) -> dict:
    if not 0.0 < scale_percentile <= 100.0:
        raise ValueError("scale_percentile must be in (0, 100]")
    values = np.asarray(array, dtype=np.float32)
    absolute = np.abs(values).reshape(-1)
    max_abs = float(np.max(absolute))
    selected_abs = float(np.percentile(absolute, scale_percentile))
    scale = selected_abs / 127.0 if selected_abs > 0.0 else 1.0 / 127.0
    quantized = np.clip(np.rint(values / scale), -127, 127)
    restored = quantized * scale
    return {
        "shape": list(values.shape),
        "elements": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
        "max_abs": max_abs,
        "abs_p99": float(np.percentile(absolute, 99.0)),
        "abs_p999": float(np.percentile(absolute, 99.9)),
        "abs_p9999": float(np.percentile(absolute, 99.99)),
        "a8_scale": scale,
        "a8_dequant_mae": float(np.mean(np.abs(values - restored), dtype=np.float64)),
        "a8_saturation_count": int(np.count_nonzero(np.abs(quantized) >= 127)),
    }


def load_input(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    with np.load(path, allow_pickle=False) as archive:
        if "input" in archive:
            return np.asarray(archive["input"], dtype=np.float32)
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]], dtype=np.float32)
        raise ValueError(f"cannot choose input from keys {archive.files}: {path}")


def load_input_records(input_npzs: list[Path], input_list: Path | None) -> list[dict]:
    records = [
        {"path": path.resolve(), "domain": "unspecified", "sample_id": path.stem}
        for path in input_npzs
    ]
    if input_list is not None:
        document = json.loads(input_list.read_text())
        values = document.get("samples") if isinstance(document, dict) else document
        if not isinstance(values, list):
            raise ValueError("input list must be a list or contain a 'samples' list")
        for index, value in enumerate(values):
            if isinstance(value, str):
                value = {"path": value}
            if not isinstance(value, dict) or "path" not in value:
                raise ValueError(f"invalid input-list record {index}: {value!r}")
            path = Path(value["path"])
            if not path.is_absolute():
                path = input_list.parent / path
            records.append({
                "path": path.resolve(),
                "domain": str(value.get("domain", "unspecified")),
                "sample_id": str(value.get("sample_id", path.stem)),
            })
    if not records:
        raise ValueError("at least one --input-npz or --input-list sample is required")
    for record in records:
        if not record["path"].is_file():
            raise FileNotFoundError(record["path"])
    return records


def deterministic_tensor_sample(array: np.ndarray, count: int, phase: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    if values.size <= count:
        return values.copy()
    stride = max(values.size // count, 1)
    offset = phase % stride
    return values[offset::stride][:count].copy()


def quantization_metrics(values: np.ndarray, scale: float) -> dict[str, float]:
    restored = np.clip(np.rint(values / np.float32(scale)), -127, 127) * np.float32(scale)
    error = restored - values
    return {
        "relative_l2": float(
            np.linalg.norm(error.astype(np.float64))
            / max(np.linalg.norm(values.astype(np.float64)), 1.0e-30)
        ),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "saturation_fraction": float(np.mean(np.abs(values) >= 127.0 * scale)),
    }


def balanced_mse_statistics(
    samples_by_domain: dict[str, list[np.ndarray]],
    shapes: set[tuple[int, ...]],
    candidate_count: int,
    minimum_percentile: float,
) -> dict:
    if not 0.0 < minimum_percentile < 100.0:
        raise ValueError("minimum_percentile must be in (0, 100)")
    splits: dict[str, dict[str, np.ndarray]] = {}
    for domain, samples in samples_by_domain.items():
        if not samples:
            continue
        training = [value for index, value in enumerate(samples) if index % 4 != 3]
        validation = [value for index, value in enumerate(samples) if index % 4 == 3]
        # A one-to-three sample domain would otherwise have an empty validation set.
        if not validation:
            validation = training
        splits[domain] = {
            "training": np.concatenate(training),
            "validation": np.concatenate(validation),
            "all": np.concatenate(samples),
        }
    training_pool = np.concatenate([value["training"] for value in splits.values()])
    absolute = np.abs(training_pool)
    lower = float(np.percentile(absolute, minimum_percentile))
    upper = float(np.max(absolute))
    lower = max(lower, upper / 127.0, np.finfo(np.float32).tiny)
    candidate_abs = np.geomspace(lower, max(upper, lower), candidate_count)
    candidate_scales = sorted(set((candidate_abs / 127.0).tolist() + [upper / 127.0]))
    candidates = []
    for scale in candidate_scales:
        by_domain = {
            domain: {
                split: quantization_metrics(values[split], scale)
                for split in ("training", "validation")
            }
            for domain, values in splits.items()
        }
        objective = float(np.mean([
            result["training"]["relative_l2"] for result in by_domain.values()
        ]))
        candidates.append({
            "scale": float(scale),
            "balanced_training_relative_l2": objective,
            "domains": by_domain,
        })
    selected = min(candidates, key=lambda item: item["balanced_training_relative_l2"])
    all_values = np.concatenate([value["all"] for value in splits.values()])
    stats = tensor_statistics(all_values, 100.0)
    stats.update({
        "shape": list(next(iter(shapes))) if len(shapes) == 1 else [list(item) for item in sorted(shapes)],
        "sampled_elements": int(all_values.size),
        "domains": {domain: len(values) for domain, values in samples_by_domain.items()},
        "a8_scale": selected["scale"],
        "a8_scale_selection": {
            "policy": "minimum equal-domain mean training relative-L2",
            "split": "every fourth sample per domain is validation",
            "minimum_percentile": minimum_percentile,
            "candidate_count": len(candidates),
            "selected": selected,
            "best_validation": {
                domain: selected["domains"][domain]["validation"]
                for domain in selected["domains"]
            },
        },
    })
    selected_metrics = quantization_metrics(all_values, selected["scale"])
    stats["a8_dequant_mae"] = selected_metrics["mae"]
    stats["a8_saturation_count"] = int(
        np.count_nonzero(np.abs(all_values) >= 127.0 * selected["scale"])
    )
    return stats


def profile_model(
    model_path: Path,
    input_records: list[dict],
    manifest_path: Path,
    golden: Path | None,
    scale_percentile: float = 100.0,
    scale_objective: str = "percentile",
    values_per_tensor_per_input: int = 16384,
    candidate_count: int = 65,
    minimum_percentile: float = 99.0,
) -> dict:
    model = onnx.load(str(model_path.resolve()), load_external_data=True)
    operators = selected_operators(model)
    kind_counts = {
        kind: sum(item["kind"] == kind for item in operators)
        for kind in ("encoder_linear", "decoder_conv")
    }
    if kind_counts != {"encoder_linear": 72, "decoder_conv": 32}:
        raise ValueError(f"unexpected operator coverage: {kind_counts}")
    activation_names = list(dict.fromkeys(item["activation"] for item in operators))

    profile_model_proto = copy.deepcopy(model)
    existing_outputs = {value.name for value in profile_model_proto.graph.output}
    for name in activation_names:
        if name not in existing_outputs:
            profile_model_proto.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
    output_names = [value.name for value in profile_model_proto.graph.output]
    with tempfile.TemporaryDirectory(prefix="depthanything_a8b8_profile_") as directory:
        temporary_model = Path(directory) / "profile.onnx"
        onnx.save_model(
            profile_model_proto,
            str(temporary_model),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="profile.onnx.data",
            size_threshold=1024,
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(temporary_model), options, providers=["CPUExecutionProvider"]
        )
        runtime_input = session.get_inputs()[0]
        sampled: dict[str, dict[str, list[np.ndarray]]] = {
            name: {} for name in activation_names
        }
        shapes: dict[str, set[tuple[int, ...]]] = {name: set() for name in activation_names}
        input_shape: list[int] | None = None
        for input_index, record in enumerate(input_records):
            input_value = load_input(record["path"])
            if list(input_value.shape) != list(runtime_input.shape):
                raise ValueError(
                    f"input shape {list(input_value.shape)} != model shape "
                    f"{runtime_input.shape}: {record['path']}"
                )
            if input_shape is None:
                input_shape = list(input_value.shape)
            outputs = session.run(output_names, {runtime_input.name: input_value})
            values_by_name = dict(zip(output_names, outputs))
            if golden is not None and input_index == 0:
                golden.parent.mkdir(parents=True, exist_ok=True)
                np.save(golden, values_by_name[model.graph.output[0].name])
            for name_index, name in enumerate(activation_names):
                value = values_by_name[name]
                shapes[name].add(tuple(value.shape))
                sampled[name].setdefault(record["domain"], []).append(
                    deterministic_tensor_sample(
                        value, values_per_tensor_per_input,
                        input_index * 131 + name_index * 17,
                    )
                )
            print(json.dumps({
                "profiled": input_index + 1,
                "total": len(input_records),
                "domain": record["domain"],
                "sample_id": record["sample_id"],
            }), flush=True)

    if scale_objective == "balanced-mse":
        tensors = {
            name: balanced_mse_statistics(
                sampled[name], shapes[name], candidate_count, minimum_percentile
            )
            for name in activation_names
        }
        scale_policy = "balanced per-domain sampled activation relative-L2 search"
    else:
        tensors = {
            name: tensor_statistics(
                np.concatenate([
                    value for values in sampled[name].values() for value in values
                ]),
                scale_percentile,
            )
            for name in activation_names
        }
        scale_policy = f"symmetric signed INT8 abs percentile {scale_percentile:g} / 127"
    for item in operators:
        item["a8_scale"] = tensors[item["activation"]]["a8_scale"]
    manifest = {
        "schema_version": 1,
        "scale_policy": scale_policy,
        "scale_objective": scale_objective,
        "scale_percentile": scale_percentile,
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "inputs": [
            {
                "path": str(record["path"]),
                "sha256": sha256(record["path"]),
                "domain": record["domain"],
                "sample_id": record["sample_id"],
            }
            for record in input_records
        ],
        "input_shape": input_shape,
        "values_per_tensor_per_input": values_per_tensor_per_input,
        "operator_counts": kind_counts,
        "unique_activation_tensors": len(activation_names),
        "tensors": tensors,
        "operators": operators,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def replace_quant_attributes(node: onnx.NodeProto, **values: object) -> None:
    retained = [attribute for attribute in node.attribute if attribute.name not in QUANTIZATION_ATTRIBUTES]
    del node.attribute[:]
    node.attribute.extend(retained)
    for name, value in values.items():
        node.attribute.append(helper.make_attribute(name, value))


def set_attributes(node: onnx.NodeProto, **values: object) -> None:
    """Set named attributes while preserving all unrelated annotations."""

    names = set(values)
    retained = [attribute for attribute in node.attribute if attribute.name not in names]
    del node.attribute[:]
    node.attribute.extend(retained)
    for name, value in values.items():
        node.attribute.append(helper.make_attribute(name, value))


def set_a8_output_contract(node: onnx.NodeProto, scale: float) -> None:
    """Set the DS INT8 output contract (vector scale, not BF16 scalar scale)."""

    retained = [
        attribute
        for attribute in node.attribute
        if attribute.name not in {"output_bitdepth", "output_scale", "output_scales"}
    ]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute("output_bitdepth", 8))
    node.attribute.append(helper.make_attribute("output_scales", [float(scale)]))


def bind_graph_input_bitdepth(model: onnx.ModelProto) -> int:
    """Annotate every direct consumer of a newly extracted BF16 graph input.

    DS quantization normally adds these attributes before graph extraction.
    Cutting at an intermediate residual tensor creates a new graph input, so
    the residual Add also needs an explicit ``left/right_bitdepth``.  Without
    it ACMOSA assigns bitDepth=0 to that bypass edge and ACPC rejects it.
    """

    graph_inputs = {value.name for value in model.graph.input}
    updated = 0
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name not in graph_inputs:
                continue
            if node.op_type in {"Add", "Mul"}:
                prefix = "left" if index == 0 else "right"
                set_attributes(
                    node,
                    **{f"{prefix}_bitdepth": 16, f"{prefix}_scale": -1.0},
                )
            else:
                set_attributes(node, input_bitdepth=16, input_scale=-1.0)
            updated += 1
    return updated


def align_a8_activation_producers(
    model: onnx.ModelProto, profile: dict
) -> dict:
    """Emit INT8 at each producer feeding an A8 operator.

    The U250 CTC path accepts an A8 tensor directly.  Leaving the producer at
    BF16 asks ACPC to insert a BF16-to-A8 layout bridge; that sequence compiles
    but does not complete on the current bitstream.  Every profiled activation
    is exclusively consumed by its Linear/Conv target(s), so it is safe to put
    the measured scale on the producer output itself.
    """

    producers = {name: node for node in model.graph.node for name in node.output}
    scales_by_activation: dict[str, set[float]] = {}
    for record in profile["operators"]:
        scales_by_activation.setdefault(record["activation"], set()).add(
            float(record["a8_scale"])
        )

    aligned = {"encoder_linear": 0, "decoder_conv": 0}
    kind_by_activation = {
        record["activation"]: record["kind"] for record in profile["operators"]
    }
    for activation, scales in scales_by_activation.items():
        if len(scales) != 1:
            raise ValueError(f"inconsistent A8 scales for {activation!r}: {scales}")
        producer = producers.get(activation)
        if producer is None:
            raise ValueError(f"cannot align graph input/no-producer activation {activation!r}")
        set_a8_output_contract(producer, next(iter(scales)))
        aligned[kind_by_activation[activation]] += 1
    return aligned


def insert_a8_mul_bridges(model: onnx.ModelProto, profile: dict) -> dict:
    """Insert an explicit EPU identity Mul that materializes each A8 tensor."""

    nodes_by_name = {node.name: node for node in model.graph.node}
    producer_index = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    grouped: dict[str, dict] = {}
    for record in profile["operators"]:
        activation = record["activation"]
        entry = grouped.setdefault(
            activation,
            {
                "scale": float(record["a8_scale"]),
                "kind": record["kind"],
                "targets": [],
            },
        )
        if not np.isclose(entry["scale"], float(record["a8_scale"]), rtol=0, atol=0):
            raise ValueError(f"inconsistent A8 scales for {activation!r}")
        entry["targets"].append(record["node"])

    insert_after: dict[int, list[onnx.NodeProto]] = {}
    new_initializers: list[onnx.TensorProto] = []
    inserted = {"encoder_linear": 0, "decoder_conv": 0}
    for bridge_index, (activation, entry) in enumerate(grouped.items()):
        if activation not in producer_index:
            raise ValueError(f"cannot bridge graph input/no-producer activation {activation!r}")
        bridge_base = f"/DSA8Bridge/{bridge_index:03d}"
        constant_name = bridge_base + "/one"
        output_name = activation + "/DSA8"
        bridge = helper.make_node(
            "Mul",
            [activation, constant_name],
            [output_name],
            name=bridge_base + "/Mul",
            weight_bitdepth=16,
            weight_scale=-1.0,
            output_bitdepth=8,
            output_scales=[float(entry["scale"])],
        )
        insert_after.setdefault(producer_index[activation], []).append(bridge)
        new_initializers.append(
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), constant_name)
        )
        for target_name in entry["targets"]:
            target = nodes_by_name[target_name]
            if target.input[0] != activation:
                raise ValueError(
                    f"{target.name}: expected activation {activation!r}, got {target.input[0]!r}"
                )
            target.input[0] = output_name
        inserted[entry["kind"]] += 1

    rewritten = []
    for index, node in enumerate(model.graph.node):
        rewritten.append(node)
        rewritten.extend(insert_after.get(index, []))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.initializer.extend(new_initializers)
    return inserted


def apply_profile(
    quantized: Path,
    profile_path: Path,
    output: Path,
    align_producer_output: bool = False,
    insert_mul_bridge: bool = False,
) -> dict:
    if align_producer_output and insert_mul_bridge:
        raise ValueError("choose either producer alignment or explicit Mul bridges")
    model = onnx.load(str(quantized.resolve()), load_external_data=True)
    profile = json.loads(profile_path.read_text())
    nodes = {node.name: node for node in model.graph.node}
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for candidate in model.graph.node:
        for input_name in candidate.input:
            consumers.setdefault(input_name, []).append(candidate)
    applied = {"encoder_linear": 0, "decoder_conv": 0}
    linear_bias_adds_annotated = 0
    for record in profile["operators"]:
        node = nodes.get(record["node"])
        if node is None:
            raise ValueError(f"quantized model is missing node {record['node']!r}")
        if node.input[0] != record["activation"] or node.input[1] != record["weight"]:
            raise ValueError(f"node inputs changed for {node.name!r}")
        weight = np.asarray(initializers[node.input[1]], dtype=np.float32)
        activation_scale = float(record["a8_scale"])
        if record["kind"] == "encoder_linear":
            if node.op_type != "MatMul" or weight.ndim != 2:
                raise ValueError(f"invalid encoder linear {node.name}: {node.op_type}, {weight.shape}")
            weight_scales = np.max(np.abs(weight), axis=0) / 127.0
            weight_scales = np.where(weight_scales > 0.0, weight_scales, 1.0 / 127.0)
            replace_quant_attributes(
                node,
                A_bitdepth=8,
                A_scales=[activation_scale],
                B_bitdepth=8,
                B_scales=weight_scales.astype(np.float32).tolist(),
                B_quant_dim=[1],
            )
            bias_users = [
                user
                for user in consumers.get(node.output[0], [])
                if user.op_type == "Add"
                and any(name in initializers for name in user.input)
            ]
            if len(bias_users) != 1:
                raise ValueError(
                    f"{node.name}: expected exactly one constant bias Add, got "
                    f"{[user.name for user in bias_users]}"
                )
            set_attributes(
                bias_users[0],
                const_bitdepth=16,
                const_scale=-1.0,
                output_bitdepth=16,
                output_scale=-1.0,
            )
            linear_bias_adds_annotated += 1
        elif record["kind"] == "decoder_conv":
            if node.op_type != "Conv" or weight.ndim != 4:
                raise ValueError(f"invalid decoder Conv {node.name}: {node.op_type}, {weight.shape}")
            weight_scales = np.max(np.abs(weight).reshape(weight.shape[0], -1), axis=1) / 127.0
            weight_scales = np.where(weight_scales > 0.0, weight_scales, 1.0 / 127.0)
            replace_quant_attributes(
                node,
                input_bitdepth=8,
                input_scales=[activation_scale],
                weight_bitdepth=8,
                weight_ch_scales=weight_scales.astype(np.float32).tolist(),
            )
        else:
            raise ValueError(f"unknown operator kind {record['kind']!r}")
        node.attribute.append(helper.make_attribute("output_bitdepth", 16))
        node.attribute.append(helper.make_attribute("output_scale", -1.0))
        applied[record["kind"]] += 1
    expected = profile["operator_counts"]
    if applied != expected:
        raise ValueError(f"applied counts {applied} != expected {expected}")
    graph_input_consumers = bind_graph_input_bitdepth(model)
    qkv_concats_annotated = 0
    for node in model.graph.node:
        if node.op_type == "Concat" and node.name.endswith("/QKVSplit"):
            set_attributes(node, output_bitdepth=16, output_scale=-1.0)
            qkv_concats_annotated += 1
    aligned_activation_producers = (
        align_a8_activation_producers(model, profile)
        if align_producer_output
        else {"encoder_linear": 0, "decoder_conv": 0}
    )
    explicit_a8_mul_bridges = (
        insert_a8_mul_bridges(model, profile)
        if insert_mul_bridge
        else {"encoder_linear": 0, "decoder_conv": 0}
    )
    save_external(model, output)
    return {
        "input": str(quantized.resolve()),
        "profile": str(profile_path.resolve()),
        "output": str(output.resolve()),
        "applied": applied,
        "graph_input_consumers_annotated": graph_input_consumers,
        "linear_bias_adds_annotated": linear_bias_adds_annotated,
        "qkv_concats_annotated": qkv_concats_annotated,
        "aligned_activation_producers": aligned_activation_producers,
        "explicit_a8_mul_bridges": explicit_a8_mul_bridges,
    }


def parse_shape(text: str) -> list[int]:
    result = [int(value) for value in text.split(",")]
    if not result or any(value <= 0 for value in result):
        raise argparse.ArgumentTypeError("shape must be comma-separated positive integers")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--boundary", default="/Add_1_output_0")
    extract_parser.add_argument("--graph-output", default="depth")
    extract_parser.add_argument("--input-shape", type=parse_shape, default=[1, 1370, 384])
    extract_parser.add_argument("--output-shape", type=parse_shape)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--model", type=Path, required=True)
    profile_parser.add_argument("--input-npz", type=Path, action="append", default=[])
    profile_parser.add_argument(
        "--input-list", type=Path,
        help="JSON list (or object with 'samples') containing path/domain/sample_id records",
    )
    profile_parser.add_argument("--manifest", type=Path, required=True)
    profile_parser.add_argument("--golden", type=Path)
    profile_parser.add_argument("--scale-percentile", type=float, default=100.0)
    profile_parser.add_argument(
        "--scale-objective", choices=("percentile", "balanced-mse"),
        default="percentile",
    )
    profile_parser.add_argument("--values-per-tensor-per-input", type=int, default=16384)
    profile_parser.add_argument("--candidate-count", type=int, default=65)
    profile_parser.add_argument("--minimum-percentile", type=float, default=99.0)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--input", type=Path, required=True)
    apply_parser.add_argument("--profile", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument(
        "--align-producer-output",
        action="store_true",
        help="emit measured INT8 directly from every profiled activation producer",
    )
    apply_parser.add_argument(
        "--insert-a8-mul-bridge",
        action="store_true",
        help="insert explicit BF16 identity-Mul to materialize every A8 activation",
    )

    args = parser.parse_args()
    if args.command == "extract":
        result = extract_tail(
            args.input,
            args.output,
            args.boundary,
            args.graph_output,
            args.input_shape,
            args.output_shape,
        )
    elif args.command == "profile":
        input_records = load_input_records(args.input_npz, args.input_list)
        result = profile_model(
            args.model, input_records, args.manifest, args.golden,
            args.scale_percentile, args.scale_objective,
            args.values_per_tensor_per_input, args.candidate_count,
            args.minimum_percentile,
        )
        result = {
            "manifest": str(args.manifest.resolve()),
            "operator_counts": result["operator_counts"],
            "unique_activation_tensors": result["unique_activation_tensors"],
            "input_shape": result["input_shape"],
        }
    else:
        result = apply_profile(
            args.input,
            args.profile,
            args.output,
            align_producer_output=args.align_producer_output,
            insert_mul_bridge=args.insert_a8_mul_bridge,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
