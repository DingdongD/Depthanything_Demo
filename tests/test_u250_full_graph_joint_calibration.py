from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from calibrate_u250_full_graph_joint import (  # noqa: E402
    gate,
    trace_index,
    update_decoder_contract,
)
from export_u250_full_graph_joint_candidate import (  # noqa: E402
    attention_signature,
    direct_source,
)
from build_u250_full_graph_family_candidate import (  # noqa: E402
    encoder_kernel_layer,
    parse_indices,
)
from select_u250_final_depth_multidomain_candidate import (  # noqa: E402
    group_metrics,
    package_provenance,
    rank_candidates,
)
from u250_calibration_manifest import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    load_manifest,
    samples_for_shape,
)


STAGES = [
    "1_norm1_to_qkv_input_a8",
    "2_per_head_qkv_a8",
    "3_softmax_dual_range_threshold",
    "4_dual_av_merge_boundary",
    "5_post_fc2_decoder_input_a8",
    "6_full_graph_final_depth_gate",
]


def make_manifest(tmp_path: Path) -> Path:
    samples = []
    identities = (
        ("nyu/a", "nyu", "training"),
        ("nyu/b", "nyu", "validation"),
        ("da2k/indoor/c", "da2k", "training"),
        ("da2k/outdoor/d", "da2k", "validation"),
    )
    for sample_id, domain, split in identities:
        tensor = tmp_path / "inputs" / Path(sample_id).with_suffix(".npy")
        tensor.parent.mkdir(parents=True, exist_ok=True)
        np.save(tensor, np.zeros((1, 3, 14, 14), np.float32))
        samples.append({
            "sample_id": sample_id,
            "domain": domain,
            "scene": None,
            "split": split,
            "source": str(tensor),
            "source_relative": tensor.name,
            "source_sha256": file_sha256(tensor),
            "tensors": {
                "14": {
                    "path": str(tensor),
                    "sha256": file_sha256(tensor),
                    "shape": [1, 3, 14, 14],
                }
            },
        })
    ids = {
        split: [item["sample_id"] for item in samples if item["split"] == split]
        for split in ("training", "validation")
    }
    manifest = {
        "schema": "depthanything-u250-full-graph-calibration-set-v1",
        "seed": "test",
        "selection": {},
        "preprocessing": {},
        "shapes": [14],
        "sample_ids": ids,
        "samples": samples,
        "calibration_policy": {
            "mode": "single-pass-full-graph-joint",
            "teacher_forcing": False,
            "sequential_layer_freeze": False,
            "av_output_gain": 1.0,
            "stages": [{"name": name, "sample_ids": ids} for name in STAGES],
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(manifest))
    return output


def test_manifest_enforces_one_sample_set_for_all_six_stages(tmp_path: Path):
    path = make_manifest(tmp_path)
    manifest = load_manifest(path)
    assert len(samples_for_shape(manifest, 14)) == 4
    expected = manifest["sample_ids"]
    assert len(manifest["calibration_policy"]["stages"]) == 6
    assert all(stage["sample_ids"] == expected
               for stage in manifest["calibration_policy"]["stages"])
    assert manifest["calibration_policy"]["teacher_forcing"] is False
    assert manifest["calibration_policy"]["sequential_layer_freeze"] is False


def test_manifest_rejects_a_layer_specific_sample_subset(tmp_path: Path):
    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["calibration_policy"]["stages"][2]["sample_ids"]["training"] = ["nyu/a"]
    manifest["manifest_sha256"] = canonical_sha256({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    })
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unified sample set"):
        load_manifest(path)


def test_trace_index_rejects_missing_unified_sample(tmp_path: Path):
    manifest_path = make_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    records = []
    for item in manifest["samples"][:-1]:
        path = trace_root / Path(item["sample_id"]).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, depth=np.zeros((1, 14, 14), np.float32))
        records.append({"sample": item["sample_id"]})
    (trace_root / "manifest.json").write_text(json.dumps({
        "schema": "depthanything-fp32-full-graph-calibration-traces-v1",
        "calibration_manifest_sha256": manifest["manifest_sha256"],
        "manifest_shape": 14,
        "samples": records,
    }))
    with pytest.raises(ValueError, match="exactly match"):
        trace_index(trace_root, manifest, 14)


def test_full_graph_gate_uses_every_domain_and_split(tmp_path: Path):
    manifest_path = make_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    reference = tmp_path / "reference"
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for item in manifest["samples"]:
        rel = Path(item["sample_id"]).with_suffix(".npz")
        for root in (reference, baseline, candidate):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
        target = np.ones((1, 14, 14), np.float32)
        np.savez(reference / rel, depth=target)
        np.savez(baseline / rel, depth=target * 1.1)
        np.savez(candidate / rel, depth=target * 1.05)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "depthanything-u250-full-graph-joint-calibration-v1",
        "manifest_sha256": manifest["manifest_sha256"],
    }))
    output = tmp_path / "gate.json"
    code = gate(argparse.Namespace(
        manifest=manifest_path, shape=14, calibration=calibration,
        reference_root=reference, baseline_root=baseline,
        candidate_root=candidate, regression_tolerance=0.0, output=output,
    ))
    assert code == 0
    result = json.loads(output.read_text())
    assert result["accepted"] is True
    assert set(result["comparisons"]) == {
        "training:nyu", "validation:nyu",
        "training:da2k", "validation:da2k",
    }


def test_attention_signature_compares_effective_parameters():
    heads = [{"scales_bf16": {
        "q": 0.1, "k": 0.2, "v": 0.3,
        "probability": {
            "fine": 0.001, "threshold": 0.127, "residual": 0.01,
        },
    }}]
    embedded = {"attention": {"heads": heads}}
    shared = {"attention": {
        "heads": heads,
        "dual_range_probability": {"heads": {"0": {
            "fine_step": 0.001, "threshold": 0.127,
            "residual_step": 0.01,
        }}},
    }}
    assert attention_signature(embedded) == attention_signature(shared)
    shared["attention"]["dual_range_probability"]["heads"]["0"][
        "threshold"
    ] = 0.0635
    assert attention_signature(embedded) != attention_signature(shared)


def test_direct_source_searches_ordered_roots(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    wanted = second / "kernel.onnx"
    wanted.write_bytes(b"onnx")
    assert direct_source([first, second], "kernel") == wanted


def test_decoder_scale_updates_runtime_contract_and_checks_kernel_abi():
    decoder = [{
        "backend": "npu", "source_node": "conv",
        "input_quantization": {"scale": 1.0},
        "kernels": [{"name": "conv_tile"}],
    }]
    step = {"name": "conv", "kernels": [{"name": "conv_tile"}]}
    update_decoder_contract(decoder, step, 0.25)
    assert decoder[0]["input_quantization"]["scale"] == 0.25
    step["kernels"][0]["name"] = "wrong"
    with pytest.raises(ValueError, match="decoder kernel mismatch"):
        update_decoder_contract(decoder, step, 0.5)


def test_family_candidate_parses_coupled_index_groups():
    assert parse_indices("1-3,8,10-11", 12) == {1, 2, 3, 8, 10, 11}
    assert parse_indices(None, 12) is None
    assert encoder_kernel_layer("attention6_l09") == 9
    assert encoder_kernel_layer("mlp_fc1_pair_l02_p01") == 2
    with pytest.raises(ValueError, match="descending"):
        parse_indices("5-2", 12)


def test_final_depth_selector_enforces_every_group_constraint():
    baseline = {
        "training:nyu": {"relative_l2": 0.10},
        "validation:nyu": {"relative_l2": 0.20},
        "training:da2k": {"relative_l2": 0.30},
        "validation:da2k": {"relative_l2": 0.40},
    }
    candidates = {
        "baseline": baseline,
        "domain_tradeoff": {
            key: {"relative_l2": value["relative_l2"] * ratio}
            for (key, value), ratio in zip(
                baseline.items(), (0.8, 0.8, 1.01, 0.8)
            )
        },
        "pareto_improvement": {
            key: {"relative_l2": value["relative_l2"] * ratio}
            for (key, value), ratio in zip(
                baseline.items(), (0.99, 1.0, 0.98, 1.0)
            )
        },
    }
    selected, evaluated = rank_candidates(baseline, candidates, 0.0)
    assert evaluated["domain_tradeoff"]["feasible"] is False
    assert evaluated["pareto_improvement"]["feasible"] is True
    assert selected == "pareto_improvement"


def test_final_depth_selector_rejects_nonfinite_reference(tmp_path: Path):
    records = [{
        "sample_id": "nyu/a", "split": "training", "domain": "nyu",
    }]
    output = tmp_path / "output" / "nyu" / "a.npz"
    reference = tmp_path / "reference" / "nyu" / "a.npz"
    output.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    np.savez(output, depth=np.ones((1, 2, 2), np.float32))
    target = np.ones((1, 2, 2), np.float32)
    target[0, 0, 0] = np.nan
    np.savez(reference, depth=target)
    with pytest.raises(ValueError, match="invalid final depth"):
        group_metrics(records, tmp_path / "output", tmp_path / "reference")


def test_final_depth_package_provenance_verifies_bank(tmp_path: Path):
    package = tmp_path / "package"
    package.mkdir()
    bank = package / "bank.bin"
    bank.write_bytes(b"resident")
    (package / "resident_kernel_bank_manifest.json").write_text(json.dumps({
        "bank_file": bank.name, "bank_sha256": file_sha256(bank),
    }))
    for name in (
        "depthanything_u250_runtime_contract.json",
        "depthanything_u250_host_plan.json",
    ):
        (package / name).write_text("{}")
    (package / "native_codec_report_active.json").write_text(json.dumps({
        "extension_sha256": "target-extension",
    }))
    result = package_provenance(package)
    assert result["bank_sha256"] == file_sha256(bank)
    assert result["codec_extension_sha256"] == "target-extension"
    bank.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        package_provenance(package)
