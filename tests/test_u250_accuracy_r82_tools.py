import json
import sys
import copy

import onnx
import numpy as np
import torch
from onnx import TensorProto, helper

from tools.export_u250_block_alignment_models import folded_qkv, head_gains
from tools.export_u250_qkv_scale_variants import main as export_qkv_variants
from tools.export_u250_matmul_scale_variants import main as export_matmul_variants
from tools.replace_u250_kernel_in_bank import record_kernel_replacement
from tools.rebind_u250_native_codec_report import abi_projection


def test_reference_qkv_fold_uses_head_width_and_preserves_v_amplitude():
    channels = 4
    heads = 2
    state = {
        "pretrained.blocks.0.norm1.weight": torch.tensor([1., 2., 3., 4.]),
        "pretrained.blocks.0.norm1.bias": torch.tensor([.1, .2, .3, .4]),
        "pretrained.blocks.0.attn.qkv.weight": torch.arange(
            3 * channels * channels, dtype=torch.float32
        ).reshape(3 * channels, channels),
        "pretrained.blocks.0.attn.qkv.bias": torch.arange(
            3 * channels, dtype=torch.float32
        ),
    }

    folded = folded_qkv(state, 0, heads)
    weight = state["pretrained.blocks.0.attn.qkv.weight"].numpy()
    gamma = state["pretrained.blocks.0.norm1.weight"].numpy()
    beta = state["pretrained.blocks.0.norm1.bias"].numpy()
    bias = state["pretrained.blocks.0.attn.qkv.bias"].numpy()
    q_scale = (channels // heads) ** -0.5
    for index, name in enumerate(("q", "k", "v")):
        source = weight[index * channels:(index + 1) * channels]
        factor = q_scale if name == "q" else 1.0
        np.testing.assert_allclose(folded[name][0], (source * gamma).T * factor)
        np.testing.assert_allclose(
            folded[name][1],
            (bias[index * channels:(index + 1) * channels] + source @ beta)
            * factor,
        )


def test_head_gain_audit_recovers_per_head_amplitude():
    expected = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    actual = expected.copy()
    actual[:, :3] *= 1.25
    actual[:, 3:] *= 0.75

    np.testing.assert_allclose(head_gains(actual, expected, 2), [1.25, 0.75])


def test_qkv_variant_name_retains_source_layer(tmp_path, monkeypatch):
    inputs = [helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2])]
    outputs = [
        helper.make_tensor_value_info(f"out{i}", TensorProto.FLOAT, [1, 2])
        for i in range(3)
    ]
    weights = [
        helper.make_tensor(f"w{i}", TensorProto.FLOAT, [2, 2], [1, 0, 0, 1])
        for i in range(3)
    ]
    nodes = [
        helper.make_node("MatMul", ["input0", f"w{i}"], [f"out{i}"],
                         A_scales=[0.1])
        for i in range(3)
    ]
    model = helper.make_model(helper.make_graph(
        nodes, "qkv", inputs, outputs, initializer=weights
    ))
    source = tmp_path / "qkv_projection_l10.onnx"
    onnx.save(model, source)
    output = tmp_path / "variants"
    monkeypatch.setattr(sys, "argv", [
        "export_u250_qkv_scale_variants.py", "--model", str(source),
        "--output-dir", str(output), "--scales", "0.2",
    ])

    assert export_qkv_variants() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["variants"][0]["name"] == "qkv_projection_l10_s0p200000000"


def test_resident_replacement_provenance_accumulates():
    baseline = {"kernel": "decoder_conv_31_tile74", "input_scale": 0.35}
    manifest = {"single_kernel_replacement": baseline}
    replacement = {"kernel": "decoder_conv_30_tile_first", "input_scale": 0.414}

    record_kernel_replacement(manifest, replacement)

    assert manifest["kernel_replacements"] == [baseline, replacement]
    assert "single_kernel_replacement" not in manifest


def test_matmul_variant_preserves_kernel_name_and_changes_scale(
    tmp_path, monkeypatch
):
    inputs = [helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2])]
    outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])]
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2, 2], [1, 0, 0, 1])
    node = helper.make_node(
        "MatMul", ["input0", "weight"], ["out"], A_scales=[0.1]
    )
    model = helper.make_model(helper.make_graph(
        [node], "post", inputs, outputs, initializer=[weight]
    ))
    source = tmp_path / "post_attention_l10.onnx"
    onnx.save(model, source)
    output = tmp_path / "variants"
    monkeypatch.setattr(sys, "argv", [
        "export_u250_matmul_scale_variants.py", "--model", str(source),
        "--output-dir", str(output), "--scales", "0.2",
    ])

    assert export_matmul_variants() == 0
    path = output / "s0p200000000" / "post_attention_l10.onnx"
    variant = onnx.load(path)
    attribute = next(
        item for item in variant.graph.node[0].attribute if item.name == "A_scales"
    )
    assert helper.get_attribute_value(attribute) == [0.20000000298023224]


def test_codec_report_rebind_allows_provenance_but_not_abi_changes():
    source = {
        "alignment_bytes": 4096,
        "bank_size_bytes": 8192,
        "format": "test",
        "required_fm_io_bytes": 1024,
        "shared_fm_base_units": 10,
        "shared_fm_placement": {},
        "shared_fm_workspace_bytes": 2048,
        "cases": [{
            "name": "post_attention_l10", "inputs": [{"dims": [1, 2]}],
            "outputs": [{"dims": [1, 2]}], "isa_ranges": [1, 1],
            "base_addresses_local": [0, 1], "offset_bytes": 4096,
            "source_cfg": "old", "sha256": "old",
        }],
    }
    replacement = copy.deepcopy(source)
    replacement["bank_sha256"] = "new"
    replacement["cases"][0]["source_cfg"] = "new"
    replacement["cases"][0]["sha256"] = "new"
    assert abi_projection(source) == abi_projection(replacement)
    replacement["cases"][0]["inputs"][0]["dims"] = [1, 3]
    assert abi_projection(source) != abi_projection(replacement)
