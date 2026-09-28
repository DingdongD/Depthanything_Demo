#!/usr/bin/env python3
"""Jointly calibrate DepthAnything U250 stages 1-6 on one frozen set.

Unlike the historical scripts, this tool never accepts a layer-specific sample
count and never teacher-forces the previous block.  All scale/threshold
proposals are emitted together, with one manifest fingerprint.  A proposal is
not deployable until ``gate`` compares complete free-running graph outputs on
the same training and validation identities.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from tools.u250_calibration_manifest import load_manifest, samples_for_shape
except ModuleNotFoundError:  # Direct execution keeps tools/ on sys.path.
    from u250_calibration_manifest import load_manifest, samples_for_shape


TRACE_SCHEMA = "depthanything-fp32-full-graph-calibration-traces-v1"
REPORT_SCHEMA = "depthanything-u250-full-graph-joint-calibration-v1"


def bf16(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).copy()
    bits = value.view(np.uint32)
    bits += np.uint32(0x7FFF) + ((bits >> 16) & 1)
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def quantize(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(value / np.float32(scale)), -128, 127).astype(np.int8)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class Metric:
    def __init__(self) -> None:
        self.sse = 0.0
        self.reference_sq = 0.0
        self.elements = 0
        self.clipped = 0

    def add(self, actual: np.ndarray, reference: np.ndarray,
            clipped: int = 0) -> None:
        actual64 = np.asarray(actual, dtype=np.float64)
        reference64 = np.asarray(reference, dtype=np.float64)
        delta = actual64 - reference64
        self.sse += float(np.sum(delta * delta))
        self.reference_sq += float(np.sum(reference64 * reference64))
        self.elements += int(reference64.size)
        self.clipped += int(clipped)

    def result(self) -> dict[str, float | int]:
        return {
            "relative_l2": float(np.sqrt(self.sse / max(self.reference_sq, 1e-30))),
            "mse": float(self.sse / max(self.elements, 1)),
            "clipping_fraction": float(self.clipped / max(self.elements, 1)),
            "elements": self.elements,
        }


def trace_index(trace_root: Path, manifest: dict, shape: int) -> list[dict]:
    trace_manifest_path = trace_root / "manifest.json"
    trace_manifest = json.loads(trace_manifest_path.read_text())
    if trace_manifest.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"not a compact full-graph trace: {trace_manifest_path}")
    if trace_manifest.get("calibration_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("trace and calibration manifest fingerprints differ")
    if int(trace_manifest.get("manifest_shape")) != shape:
        raise ValueError("trace input shape differs from requested calibration shape")
    expected = samples_for_shape(manifest, shape, verify_files=False)
    actual = {item["sample"]: item for item in trace_manifest.get("samples", [])}
    expected_ids = {item["sample_id"] for item in expected}
    if set(actual) != expected_ids:
        raise ValueError("trace sample IDs do not exactly match the unified manifest")
    result = []
    for item in expected:
        path = trace_root / Path(item["sample_id"]).with_suffix(".npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({**item, "trace_path": path})
    return result


def groups(records: list[dict]) -> tuple[str, ...]:
    return tuple(sorted({f"{item['split']}:{item['domain']}" for item in records}))


def candidate_scales(current: float, count: int) -> list[float]:
    return sorted(set(float(value) for value in np.geomspace(
        current * 0.5, current * 2.0, count
    )) | {float(current)})


def load_values(records: list[dict], key: str,
                channel_slice: slice | None = None) -> list[tuple[str, np.ndarray]]:
    values = []
    for item in records:
        with np.load(item["trace_path"], allow_pickle=False) as archive:
            value = np.asarray(archive[key], dtype=np.float32)
            if channel_slice is not None:
                value = value[..., channel_slice]
            value = np.ascontiguousarray(value)
        values.append((f"{item['split']}:{item['domain']}", value))
    return values


def scale_metrics(values: list[tuple[str, np.ndarray]], scale: float,
                  group_names: tuple[str, ...]) -> dict[str, dict]:
    metrics = {group: Metric() for group in group_names}
    for group, value in values:
        code = quantize(value, scale)
        restored = code.astype(np.float32) * np.float32(scale)
        clipped = np.count_nonzero(
            (value < -128.0 * scale) | (value > 127.0 * scale)
        )
        metrics[group].add(restored, value, int(clipped))
    return {name: metric.result() for name, metric in metrics.items()}


def balanced_objective(result: dict[str, dict], split: str) -> float:
    values = [value["relative_l2"] for name, value in result.items()
              if name.startswith(split + ":")]
    if not values:
        raise ValueError(f"no {split} samples in calibration manifest")
    return float(np.mean(values))


def choose_scale(records: list[dict], key: str, current: float,
                 count: int, validation_tolerance: float) -> dict:
    values = load_values(records, key)
    group_names = groups(records)
    candidates = []
    for scale in candidate_scales(current, count):
        result = scale_metrics(values, scale, group_names)
        candidates.append({
            "scale": scale,
            "metrics": result,
            "training_objective": balanced_objective(result, "training"),
            "validation_objective": balanced_objective(result, "validation"),
        })
    baseline = next(item for item in candidates if item["scale"] == current)
    proposed = min(candidates, key=lambda item: item["training_objective"])
    accepted = (
        proposed["validation_objective"]
        <= baseline["validation_objective"] * (1.0 + validation_tolerance)
    )
    selected = proposed if accepted else baseline
    return {
        "key": key,
        "current": baseline,
        "proposed": proposed,
        "selected": selected,
        "local_validation_gate": accepted,
    }


def probability_parameters(specification: dict, attention: dict,
                           head: int) -> tuple[float, float]:
    shared = attention.get("dual_range_probability")
    if shared is not None:
        params = shared["heads"][str(head)]
        return float(params["threshold"]), float(params["residual_step"])
    params = specification["scales_bf16"]["probability"]
    return float(params["threshold"]), float(params.get("residual", params.get("residual_step")))


def probability_candidates(current: tuple[float, float]) -> list[tuple[float, float]]:
    denominators = (64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024)
    result = {(1.0 / value, multiplier / value)
              for value in denominators for multiplier in (0.5, 1.0, 2.0)}
    result.add(current)
    return sorted(result)


def encode_probability(probability: np.ndarray, threshold: float,
                       residual_step: float) -> tuple[np.ndarray, dict[str, float]]:
    fine_step = threshold / 127.0
    fine = np.clip(np.floor(probability / fine_step), 0, 127).astype(np.int16)
    residual = np.maximum(np.clip(
        np.rint((probability - threshold) / residual_step), -128, 127
    ).astype(np.int16), 0)
    represented = (fine.astype(np.float32) * np.float32(fine_step)
                   + residual.astype(np.float32) * np.float32(residual_step))
    return represented, {
        "fine_zero_fraction": float(np.mean(fine == 0)),
        "fine_saturation_fraction": float(np.mean(fine == 127)),
        "residual_zero_fraction": float(np.mean(residual == 0)),
        "residual_saturation_fraction": float(np.mean(residual == 127)),
        "row_sum_mean": float(np.mean(np.sum(represented, axis=-1))),
    }


def attention_head(records: list[dict], layer: int, head: int,
                   specification: dict, attention: dict,
                   scale_count: int, validation_tolerance: float) -> dict:
    begin, end = head * 64, (head + 1) * 64
    old = {name: float(specification["scales_bf16"][name])
           for name in ("q", "k", "v")}
    selected_scales = {}
    scale_reports = {}
    for name, suffix in (("q", "q_rows"), ("k", "k"), ("v", "v")):
        key = f"encoder_l{layer:02d}_{suffix}"
        values = load_values(records, key, slice(begin, end))
        group_names = groups(records)
        # Per-head views are evaluated here rather than by the scalar helper.
        candidates = []
        for scale in candidate_scales(old[name], scale_count):
            metric = {group: Metric() for group in group_names}
            for group, value in values:
                code = quantize(value, scale)
                restored = code.astype(np.float32) * np.float32(scale)
                clipped = np.count_nonzero(
                    (value < -128.0 * scale) | (value > 127.0 * scale)
                )
                metric[group].add(restored, value, int(clipped))
            result = {group: value.result() for group, value in metric.items()}
            candidates.append({
                "scale": scale, "metrics": result,
                "training_objective": balanced_objective(result, "training"),
                "validation_objective": balanced_objective(result, "validation"),
            })
        baseline = next(item for item in candidates if item["scale"] == old[name])
        proposed = min(candidates, key=lambda item: item["training_objective"])
        accepted = proposed["validation_objective"] <= (
            baseline["validation_objective"] * (1.0 + validation_tolerance)
        )
        selected = proposed if accepted else baseline
        selected_scales[name] = float(selected["scale"])
        scale_reports[name] = {
            "current": baseline, "proposed": proposed, "selected": selected,
            "local_validation_gate": accepted,
        }

    old_probability = probability_parameters(specification, attention, head)
    probability_metrics = {
        candidate: {group: Metric() for group in groups(records)}
        for candidate in probability_candidates(old_probability)
    }
    contexts = {"current": {group: Metric() for group in groups(records)},
                "selected": {group: Metric() for group in groups(records)}}
    probability_stats = {candidate: [] for candidate in probability_metrics}
    cached = []
    for item in records:
        with np.load(item["trace_path"], allow_pickle=False) as archive:
            q = np.asarray(
                archive[f"encoder_l{layer:02d}_q_rows"][0, 0, :, begin:end],
                dtype=np.float32,
            )
            k = np.asarray(
                archive[f"encoder_l{layer:02d}_k"][0, 0, :, begin:end],
                dtype=np.float32,
            )
            v = np.asarray(
                archive[f"encoder_l{layer:02d}_v"][0, 0, :, begin:end],
                dtype=np.float32,
            )
            target = np.asarray(
                archive[f"encoder_l{layer:02d}_attention_rows"][0, 0, :, begin:end],
                dtype=np.float32,
            )
        def quantized_attention(scales: dict[str, float]):
            q_code = quantize(q, scales["q"]).astype(np.int32)
            k_code = quantize(k, scales["k"]).astype(np.int32)
            v_code = quantize(v, scales["v"]).astype(np.int32)
            logits = bf16(
                (q_code @ k_code.T).astype(np.float32)
                * np.float32(scales["q"] * scales["k"])
            )
            return bf16(softmax(logits)), v_code

        current_probability, current_v_code = quantized_attention(old)
        probability, v_code = quantized_attention(selected_scales)
        group = f"{item['split']}:{item['domain']}"
        cached.append((
            group, current_probability, current_v_code,
            probability, v_code, target,
        ))
        for candidate, metric in probability_metrics.items():
            represented, stats = encode_probability(probability, *candidate)
            metric[group].add(represented, probability)
            probability_stats[candidate].append(stats)

    candidate_reports = []
    for candidate, metric in probability_metrics.items():
        result = {group: value.result() for group, value in metric.items()}
        candidate_reports.append({
            "threshold": candidate[0], "residual_step": candidate[1],
            "fine_step": candidate[0] / 127.0,
            "metrics": result,
            "training_objective": balanced_objective(result, "training"),
            "validation_objective": balanced_objective(result, "validation"),
            "statistics": {
                key: float(np.mean([value[key] for value in probability_stats[candidate]]))
                for key in probability_stats[candidate][0]
            },
        })
    baseline_probability = next(
        item for item in candidate_reports
        if (item["threshold"], item["residual_step"]) == old_probability
    )
    proposed_probability = min(
        candidate_reports, key=lambda item: item["training_objective"]
    )
    probability_accepted = proposed_probability["validation_objective"] <= (
        baseline_probability["validation_objective"] * (1.0 + validation_tolerance)
    )
    selected_probability = (
        proposed_probability if probability_accepted else baseline_probability
    )

    for (group, current_probability, current_v_code,
         selected_probability_values, selected_v_code, target) in cached:
        for name, params, probability, v_code, v_scale in (
            ("current", baseline_probability, current_probability,
             current_v_code, old["v"]),
            ("selected", selected_probability, selected_probability_values,
             selected_v_code, selected_scales["v"]),
        ):
            represented, _ = encode_probability(
                probability, params["threshold"], params["residual_step"]
            )
            context = bf16(
                (represented @ v_code).astype(np.float32)
                * np.float32(v_scale)
            )
            contexts[name][group].add(context, target)
    context_results = {
        name: {group: metric.result() for group, metric in values.items()}
        for name, values in contexts.items()
    }
    current_context_validation = balanced_objective(
        context_results["current"], "validation"
    )
    proposed_context_validation = balanced_objective(
        context_results["selected"], "validation"
    )
    joint_attention_accepted = proposed_context_validation <= (
        current_context_validation * (1.0 + validation_tolerance)
    )
    if not joint_attention_accepted:
        # Q/K/V reconstruction and probability reconstruction are only
        # proposal surrogates.  The actual dual-AV context boundary has final
        # authority before the full-depth gate; reject the entire coupled head
        # update rather than leaking a locally harmful partial selection.
        selected_scales = dict(old)
        for name in ("q", "k", "v"):
            scale_reports[name]["selected"] = scale_reports[name]["current"]
        selected_probability = baseline_probability
        context_results["selected"] = copy.deepcopy(context_results["current"])
    return {
        "head": head,
        "qkv_scales": scale_reports,
        "selected_scales_bf16": selected_scales,
        "probability": {
            "current": baseline_probability,
            "proposed": proposed_probability,
            "selected": selected_probability,
            "probability_validation_gate": probability_accepted,
        },
        "dual_av_merge": context_results,
        "joint_attention_validation_gate": joint_attention_accepted,
        "current_context_validation_objective": current_context_validation,
        "proposed_context_validation_objective": proposed_context_validation,
        "av_output_gain": 1.0,
        "v_scale_shared_by_both_av_routes": True,
    }


def update_attention_contract(attention: dict, head_report: dict) -> None:
    head = int(head_report["head"])
    specification = attention["heads"][head]
    scales = head_report["selected_scales_bf16"]
    specification["scales_bf16"].update(scales)
    specification["scales_bf16"]["av_v"] = scales["v"]
    specification["scales_bf16"]["av_output_gain"] = 1.0
    selected = head_report["probability"]["selected"]
    shared = attention.get("dual_range_probability")
    if shared is not None:
        shared["av_output_gain"] = 1.0
        shared["v_unchanged"] = True
        shared["heads"][str(head)] = {
            "threshold": selected["threshold"],
            "fine_step": selected["fine_step"],
            "residual_step": selected["residual_step"],
        }
    else:
        specification["scales_bf16"]["probability"] = {
            "threshold": selected["threshold"],
            "fine": selected["fine_step"],
            "residual": selected["residual_step"],
        }


def update_decoder_contract(decoder: list[dict], step: dict,
                            scale: float) -> None:
    """Keep the executable contract and host plan on one decoder scale."""
    matches = [item for item in decoder
               if item.get("backend") == "npu"
               and item.get("source_node") == step["name"]]
    if len(matches) != 1:
        raise ValueError(
            f"expected one NPU decoder contract entry for {step['name']}, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if ([item["name"] for item in entry["kernels"]]
            != [item["name"] for item in step["kernels"]]):
        raise ValueError(f"decoder kernel mismatch for {step['name']}")
    entry["input_quantization"]["scale"] = float(scale)


def fit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    records = trace_index(args.trace_root, manifest, args.shape)
    contract = json.loads(args.contract.read_text())
    host_plan = json.loads(args.host_plan.read_text())
    proposed_contract = copy.deepcopy(contract)
    proposed_plan = copy.deepcopy(host_plan)
    encoder_reports = []
    local_objectives = []
    for layer, block in enumerate(contract["encoder"]):
        proposal = proposed_contract["encoder"][layer]
        stage1 = choose_scale(
            records, f"encoder_l{layer:02d}_norm1_values",
            float(block["qkv"]["input_quantization"]["scale"]),
            args.scale_candidates, args.validation_tolerance,
        )
        proposal["qkv"]["input_quantization"]["scale"] = stage1["selected"]["scale"]
        heads = []
        for head, specification in enumerate(block["attention"]["heads"]):
            report = attention_head(
                records, layer, head, specification, block["attention"],
                args.scale_candidates, args.validation_tolerance,
            )
            heads.append(report)
            update_attention_contract(proposal["attention"], report)
            local_objectives.append(
                balanced_objective(report["dual_av_merge"]["selected"], "validation")
            )
        stage5 = {
            "post_attention": choose_scale(
                records, f"encoder_l{layer:02d}_attention_values",
                float(block["post_attention"]["input_quantization"]["scale"]),
                args.scale_candidates, args.validation_tolerance,
            ),
            "fc1": choose_scale(
                records, f"encoder_l{layer:02d}_norm2_values",
                float(block["mlp"]["fc1_input_quantization"]["scale"]),
                args.scale_candidates, args.validation_tolerance,
            ),
            "fc2": choose_scale(
                records, f"encoder_l{layer:02d}_gelu_values",
                float(block["mlp"]["fc2_input_quantization"]["scale"]),
                args.scale_candidates, args.validation_tolerance,
            ),
        }
        proposal["post_attention"]["input_quantization"]["scale"] = (
            stage5["post_attention"]["selected"]["scale"]
        )
        proposal["mlp"]["fc1_input_quantization"]["scale"] = (
            stage5["fc1"]["selected"]["scale"]
        )
        proposal["mlp"]["fc2_input_quantization"]["scale"] = (
            stage5["fc2"]["selected"]["scale"]
        )
        encoder_reports.append({
            "layer": layer, "stage1_qkv_input": stage1,
            "stages2_3_4_attention_heads": heads, "stage5": stage5,
        })
        print(json.dumps({"calibrated_layer": layer, "total": 12}), flush=True)

    decoder_reports = []
    for step, proposed_step in zip(
        host_plan["decoder_steps"], proposed_plan["decoder_steps"]
    ):
        if step.get("backend") != "npu":
            continue
        index = int(step["index"])
        report = choose_scale(
            records, f"decoder_input_{index:02d}_values", float(step["input_scale"]),
            args.scale_candidates, args.validation_tolerance,
        )
        selected_scale = report["selected"]["scale"]
        proposed_step["input_scale"] = selected_scale
        update_decoder_contract(
            proposed_contract["decoder"], proposed_step, selected_scale
        )
        decoder_reports.append({"index": index, "node": step["name"], **report})

    calibration = proposed_contract.setdefault("calibration", {})
    for legacy_key in (
        "active_per_head_layers", "attention_profile", "control_layers"
    ):
        calibration.pop(legacy_key, None)
    calibration.update({
        "mode": "single-pass-full-graph-joint",
        "manifest_sha256": manifest["manifest_sha256"],
        "shape": args.shape,
        "teacher_forcing": False,
        "sequential_layer_freeze": False,
        "av_output_gain": 1.0,
        "deployment_status": "proposal_requires_full_graph_gate",
    })
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_output = output_dir / "proposed_runtime_contract.json"
    plan_output = output_dir / "proposed_host_plan.json"
    contract_output.write_text(json.dumps(proposed_contract, indent=2, sort_keys=True) + "\n")
    plan_output.write_text(json.dumps(proposed_plan, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": REPORT_SCHEMA,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "shape": args.shape,
        "sample_ids": manifest["sample_ids"],
        "policy": {
            "one_frozen_sample_set_for_stages_1_to_6": True,
            "teacher_forcing": False,
            "sequential_layer_freeze": False,
            "all_layers_proposed_before_full_graph_gate": True,
            "av_output_gain": 1.0,
            "scale_candidates": args.scale_candidates,
            "validation_tolerance": args.validation_tolerance,
        },
        "encoder": encoder_reports,
        "decoder": decoder_reports,
        "joint_local_validation_objective": float(np.mean(local_objectives)),
        "proposed_runtime_contract": str(contract_output.resolve()),
        "proposed_host_plan": str(plan_output.resolve()),
        "deployment_status": "proposal_requires_full_graph_gate",
    }
    report_output = output_dir / "joint_calibration_report.json"
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(report_output.resolve()),
        "deployment_status": report["deployment_status"],
        "manifest_sha256": manifest["manifest_sha256"],
    }, sort_keys=True))
    return 0


def synchronize(args: argparse.Namespace) -> int:
    """Repair a proposal whose decoder plan changed without its contract."""
    contract = json.loads(args.contract.read_text())
    plan = json.loads(args.host_plan.read_text())
    report = json.loads(args.report.read_text())
    for step in plan["decoder_steps"]:
        if step.get("backend") == "npu":
            update_decoder_contract(
                contract["decoder"], step, float(step["input_scale"])
            )
    status = "proposal_requires_full_graph_gate"
    contract.setdefault("calibration", {})["deployment_status"] = status
    report["deployment_status"] = status
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract_output = args.output_dir / "proposed_runtime_contract.json"
    plan_output = args.output_dir / "proposed_host_plan.json"
    report_output = args.output_dir / "joint_calibration_report.json"
    report["proposed_runtime_contract"] = str(contract_output.resolve())
    report["proposed_host_plan"] = str(plan_output.resolve())
    contract_output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "contract": str(contract_output.resolve()),
        "host_plan": str(plan_output.resolve()),
        "report": str(report_output.resolve()),
        "deployment_status": status,
    }, sort_keys=True))
    return 0


def output_path(root: Path, sample_id: str) -> Path:
    return root / Path(sample_id).with_suffix(".npz")


def depth_metrics(records: list[dict], root: Path,
                  reference_root: Path) -> dict[str, dict]:
    metrics = {group: Metric() for group in groups(records)}
    for item in records:
        with np.load(output_path(root, item["sample_id"]), allow_pickle=False) as output:
            actual = np.asarray(output["depth"], dtype=np.float32)
        with np.load(output_path(reference_root, item["sample_id"]), allow_pickle=False) as ref:
            target = np.asarray(ref["depth"], dtype=np.float32)
        metrics[f"{item['split']}:{item['domain']}"].add(actual, target)
    return {name: value.result() for name, value in metrics.items()}


def gate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    records = samples_for_shape(manifest, args.shape, verify_files=False)
    report = json.loads(args.calibration.read_text())
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("not a joint full-graph calibration report")
    if report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("calibration report and gate manifest fingerprints differ")
    for root in (args.reference_root, args.baseline_root, args.candidate_root):
        missing = [item["sample_id"] for item in records
                   if not output_path(root, item["sample_id"]).is_file()]
        if missing:
            raise ValueError(f"{root}: missing {len(missing)} unified samples")
    baseline = depth_metrics(records, args.baseline_root, args.reference_root)
    candidate = depth_metrics(records, args.candidate_root, args.reference_root)
    comparisons = {}
    accepted = True
    for group in groups(records):
        before = float(baseline[group]["relative_l2"])
        after = float(candidate[group]["relative_l2"])
        ratio = after / max(before, 1e-30)
        comparisons[group] = {
            "baseline_relative_l2": before,
            "candidate_relative_l2": after,
            "candidate_over_baseline": ratio,
        }
        accepted &= ratio <= 1.0 + args.regression_tolerance
    result = {
        "schema": "depthanything-u250-full-graph-joint-gate-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "shape": args.shape,
        "free_running": True,
        "teacher_forcing": False,
        "sample_ids": manifest["sample_ids"],
        "comparisons": comparisons,
        "regression_tolerance": args.regression_tolerance,
        "accepted": bool(accepted),
        "deployment_status": "qualified" if accepted else "rejected",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if accepted else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--manifest", type=Path, required=True)
    fit_parser.add_argument("--shape", type=int, required=True)
    fit_parser.add_argument("--trace-root", type=Path, required=True)
    fit_parser.add_argument("--contract", type=Path, required=True)
    fit_parser.add_argument("--host-plan", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--scale-candidates", type=int, default=25)
    fit_parser.add_argument("--validation-tolerance", type=float, default=0.0)
    fit_parser.set_defaults(function=fit)
    sync_parser = subparsers.add_parser("synchronize")
    sync_parser.add_argument("--contract", type=Path, required=True)
    sync_parser.add_argument("--host-plan", type=Path, required=True)
    sync_parser.add_argument("--report", type=Path, required=True)
    sync_parser.add_argument("--output-dir", type=Path, required=True)
    sync_parser.set_defaults(function=synchronize)
    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--manifest", type=Path, required=True)
    gate_parser.add_argument("--shape", type=int, required=True)
    gate_parser.add_argument("--calibration", type=Path, required=True)
    gate_parser.add_argument("--reference-root", type=Path, required=True)
    gate_parser.add_argument("--baseline-root", type=Path, required=True)
    gate_parser.add_argument("--candidate-root", type=Path, required=True)
    gate_parser.add_argument("--regression-tolerance", type=float, default=0.0)
    gate_parser.add_argument("--output", type=Path, required=True)
    gate_parser.set_defaults(function=gate)
    args = parser.parse_args()
    if getattr(args, "scale_candidates", 2) < 2:
        parser.error("--scale-candidates must be at least 2")
    if getattr(args, "validation_tolerance", 0.0) < 0.0:
        parser.error("--validation-tolerance must be non-negative")
    if getattr(args, "regression_tolerance", 0.0) < 0.0:
        parser.error("--regression-tolerance must be non-negative")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
