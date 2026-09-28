import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.collect_fp32_replacement_traces import (
    decoder_boundary_output, decoder_module_name, inverse_depth_to_space_crd,
    qkv_runtime_boundaries,
)
from tools.run_u250_depthanything_hybrid import depth_to_space
from tools.run_u250_depthanything_hybrid import load_reference_replacement
from tools.run_u250_depthanything_hybrid import parse_encoder_attention_head_target
from tools.run_u250_depthanything_hybrid import parse_encoder_internal_target


@pytest.mark.parametrize(("node", "module"), [
    ("/depth_head/projects.0/Conv", "depth_head.projects.0"),
    ("/depth_head/resize_layers.0/conv/Conv", "depth_head.resize_layers.0"),
    ("/depth_head/layer1_rn/Conv", "depth_head.scratch.layer1_rn"),
    ("/depth_head/refinenet3/out_conv/Conv",
     "depth_head.scratch.refinenet3.out_conv"),
    ("/depth_head/output_conv2/output_conv2.2/Conv",
     "depth_head.scratch.output_conv2.2"),
])
def test_decoder_module_name(node, module):
    assert decoder_module_name(node) == module


def test_reference_replacement_enforces_shape_and_fp32(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez(path, block_l03=np.ones((1, 7, 4), dtype=np.float16))
    with np.load(path, allow_pickle=False) as archive:
        actual = load_reference_replacement(
            archive, "block_l03", np.zeros((1, 7, 4), dtype=np.float32)
        )
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous
        with pytest.raises(ValueError, match="shape"):
            load_reference_replacement(
                archive, "block_l03", np.zeros((1, 8, 4), dtype=np.float32)
            )


def test_every_decoder_conv_maps_to_a_module():
    plan_path = Path(
        "build/depthanything_u250_r85_reference_qkv_unit_av/"
        "depthanything_u250_host_plan.json"
    )
    if not plan_path.is_file():
        pytest.skip("requires the external r85 deployment plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    names = [decoder_module_name(step["name"])
             for step in plan["decoder_steps"] if step["backend"] == "npu"]
    assert len(names) == 32
    assert len(set(names)) == 32


@pytest.mark.parametrize(("index", "shape"), [
    (1, (1, 768, 37, 37)),
    (3, (1, 384, 37, 37)),
])
def test_convtranspose_capture_is_pre_depth_to_space(index, shape):
    channels, size = ((48, 148) if index == 1 else (96, 74))
    output = torch.arange(channels * size * size, dtype=torch.float32).reshape(
        1, channels, size, size
    )
    boundary = decoder_boundary_output(index, output)
    assert tuple(boundary.shape) == shape
    restored = depth_to_space(boundary.numpy(), 2, "CRD")
    if index == 1:
        restored = depth_to_space(restored, 2, "CRD")
    np.testing.assert_array_equal(restored, output.numpy())


def test_qkv_capture_matches_runtime_shape_and_q_scaling():
    raw = torch.arange(2 * 3 * 18, dtype=torch.float32).reshape(2, 3, 18)
    q, k, v = qkv_runtime_boundaries(raw, num_heads=2, q_scale=0.125)
    assert q.shape == k.shape == v.shape == (2, 3, 6)
    expected = raw.reshape(2, 3, 3, 2, 3).permute(2, 0, 1, 3, 4)
    torch.testing.assert_close(q, expected[0].reshape(2, 3, 6) * 0.125)
    torch.testing.assert_close(k, expected[1].reshape(2, 3, 6))
    torch.testing.assert_close(v, expected[2].reshape(2, 3, 6))


@pytest.mark.parametrize(("value", "expected"), [
    ("0:norm1", (0, "norm1")),
    ("10:attention", (10, "attention")),
    ("10:attention_branch", (10, "attention_branch")),
    ("11:fc2", (11, "fc2")),
])
def test_parse_encoder_internal_target(value, expected):
    assert parse_encoder_internal_target(value) == expected


@pytest.mark.parametrize("value", ["10", "12:norm1", "3:unknown", "x:qkv"])
def test_parse_encoder_internal_target_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_encoder_internal_target(value)


@pytest.mark.parametrize(("value", "expected"), [
    ("0:0", (0, 0)), ("10:5", (10, 5)),
])
def test_parse_encoder_attention_head_target(value, expected):
    assert parse_encoder_attention_head_target(value) == expected


@pytest.mark.parametrize("value", ["10", "12:0", "3:6", "x:0"])
def test_parse_encoder_attention_head_target_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_encoder_attention_head_target(value)
