from __future__ import annotations

import onnx
import pytest
from onnx import numpy_helper
import json
import sys

from tools.export_u250_dual_range_attention_layer import main, make_model


def test_calibration_report_has_per_head_deployable_parameters(tmp_path) -> None:
    report = {
        "layer": 10,
        "v_unchanged": True,
        "av_output_gain": 1.0,
        "heads": [
            {"head": head, "selected": {
                "fine_step": 1.0 / (127.0 * (160 + head)),
                "threshold": 1.0 / (160 + head),
                "residual_step": 1.0 / (160 + head),
            }}
            for head in range(6)
        ],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    loaded = json.loads(path.read_text())
    parameters = {int(item["head"]): item["selected"] for item in loaded["heads"]}
    assert parameters[0]["threshold"] == pytest.approx(1.0 / 160.0)
    assert parameters[5]["fine_step"] == pytest.approx(1.0 / (127.0 * 165.0))


def test_export_accepts_calibration_without_unused_host_head(
    tmp_path, monkeypatch
) -> None:
    heads = [{
        "scales_bf16": {
            "q": 0.01, "k": 0.02, "v": 0.03,
            "probability": 1.0 / 128.0,
        }
    } for _ in range(6)]
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "encoder": [{
            "host_attention_heads": [3],
            "attention": {"heads": heads},
        }]
    }))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "layer": 0,
        "v_unchanged": True,
        "av_output_gain": 1.0,
        "heads": [{
            "head": head,
            "selected": {
                "fine_step": 1.0 / (224.0 * 127.0),
                "threshold": 1.0 / 224.0,
                "residual_step": 1.0 / 224.0,
            },
        } for head in (0, 1, 2, 4, 5)],
    }))
    output = tmp_path / "models"
    monkeypatch.setattr(sys, "argv", [
        "export_u250_dual_range_attention_layer.py",
        "--contract", str(contract),
        "--layer", "0",
        "--calibration", str(calibration),
        "--output-dir", str(output),
    ])
    assert main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["kernels"]) == 6
    assert manifest["kernels"][3]["fine_step"] == pytest.approx(1.0 / 16384.0)


def test_export_applies_per_head_qkv_scales_without_av_gain(
    tmp_path, monkeypatch
) -> None:
    heads = [{
        "scales_bf16": {
            "q": 0.01, "k": 0.02, "v": 0.03,
            "probability": 1.0 / 128.0,
        }
    } for _ in range(6)]
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "encoder": [{"attention": {"heads": heads}}]
    }))
    overrides = tmp_path / "qkv.json"
    overrides.write_text(json.dumps({
        "av_output_gain": 1.0,
        "heads": {
            "0": {"q": 0.04, "k": 0.05, "v": 0.06},
            "2": {"q": 0.07, "k": 0.02, "v": 0.03},
        },
    }))
    output = tmp_path / "models"
    monkeypatch.setattr(sys, "argv", [
        "export_u250_dual_range_attention_layer.py",
        "--contract", str(contract),
        "--layer", "0",
        "--qkv-scale-overrides", str(overrides),
        "--output-dir", str(output),
    ])

    assert main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    changed = manifest["kernels"][0]["scales_bf16"]
    q_only = manifest["kernels"][2]["scales_bf16"]
    unchanged = manifest["kernels"][1]["scales_bf16"]
    assert (changed["q"], changed["k"], changed["v"]) == (0.04, 0.05, 0.06)
    assert changed["av_v"] == changed["v"]
    assert changed["av_output_gain"] == 1.0
    assert (q_only["q"], q_only["k"], q_only["v"]) == (0.07, 0.02, 0.03)
    assert unchanged["q"] == 0.01


def test_export_rejects_non_unit_qkv_override_gain(tmp_path, monkeypatch) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "encoder": [{"attention": {"heads": [{
            "scales_bf16": {"q": 0.01, "k": 0.02, "v": 0.03}
        } for _ in range(6)]}}]
    }))
    overrides = tmp_path / "qkv.json"
    overrides.write_text(json.dumps({
        "av_output_gain": 1.1,
        "heads": {},
    }))
    monkeypatch.setattr(sys, "argv", [
        "export_u250_dual_range_attention_layer.py",
        "--contract", str(contract),
        "--qkv-scale-overrides", str(overrides),
        "--output-dir", str(tmp_path / "models"),
    ])

    with pytest.raises(ValueError, match="unit-amplitude"):
        main()


def attrs(node: onnx.NodeProto) -> dict:
    return {
        value.name: onnx.helper.get_attribute_value(value)
        for value in node.attribute
    }


def test_dual_range_two_chunk_topology_and_scales() -> None:
    model = make_model(0.01, 0.02, 0.03)
    assert [value.name for value in model.graph.input] == [
        "input0", "input1", "input2", "input3"
    ]
    assert [value.name for value in model.graph.output] == ["output0", "output1"]
    expected = [
        "MatMul", "Softmax", "MatMul", "Softmax", "Add", "Relu",
        "MatMul", "Add", "Reshape",
    ]
    assert [node.op_type for node in model.graph.node[:9]] == expected
    assert [node.op_type for node in model.graph.node[9:]] == expected

    fine = attrs(model.graph.node[1])
    subtract = attrs(model.graph.node[4])
    residual_av = attrs(model.graph.node[6])
    merge = attrs(model.graph.node[7])
    assert fine["output_scales"] == [1.0 / 16384.0]
    assert subtract["output_scales"] == [1.0 / 128.0]
    assert subtract["const_scale"] == 128.0
    assert residual_av["A_scales"] == [1.0 / 128.0]
    assert merge["left_bitdepth"] == 16
    assert merge["right_bitdepth"] == 16
    assert merge["output_bitdepth"] == 16

    values = {value.name: numpy_helper.to_array(value)
              for value in model.graph.initializer}
    assert float(values["dual_range_negative_threshold"]) == pytest.approx(
        -127.0 / 16384.0
    )


def test_dual_range_rejects_nonpositive_or_oversized_fine_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_model(0.01, 0.02, 0.03, fine_step=0.0)
    with pytest.raises(ValueError, match="below one"):
        make_model(0.01, 0.02, 0.03, fine_step=1.0 / 127.0)


def test_dual_range_residual_constant_uses_matching_quantizer() -> None:
    model = make_model(
        0.01, 0.02, 0.03,
        fine_step=1.0 / (288.0 * 127.0),
        residual_step=1.0 / 288.0,
    )
    subtract = attrs(model.graph.node[4])
    assert subtract["const_scale"] == 288.0
    assert subtract["output_scales"] == pytest.approx([1.0 / 288.0])
    values = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    assert float(values["dual_range_negative_threshold"]) == pytest.approx(
        -1.0 / 288.0
    )
