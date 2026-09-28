from pathlib import Path
import unittest

import torch


class DepthAnythingAdapterTest(unittest.TestCase):
    def test_profile_binder_matches_triplets_by_topology(self):
        import onnx
        from onnx import helper

        from ds_models.static_int8_attention import StaticAttentionProfile
        from tools.apply_static_attention_profile_to_onnx import bind_profile

        nodes = []
        for head in range(2):
            suffix = "" if head == 0 else f"_{head}"
            qk_output = f"qk{head}"
            probability = f"probability{head}"
            nodes.extend([
                helper.make_node(
                    "MatMul", [f"q{head}", f"k{head}"], [qk_output],
                    name=f"/blocks.0/attn/MatMul{suffix}",
                    output_bitdepth=16, output_scale=-1.0,
                ),
                helper.make_node(
                    "Softmax", [qk_output], [probability],
                    name=f"/blocks.0/attn/Softmax{suffix}", axis=-1,
                    output_bitdepth=16, output_scale=-1.0,
                ),
                helper.make_node(
                    "MatMul", [probability, f"v{head}"], [f"out{head}"],
                    name=f"/blocks.0/attn/MatMul_{head + 2}",
                    output_bitdepth=16, output_scale=-1.0,
                ),
            ])
        model = helper.make_model(helper.make_graph(nodes, "test", [], []))
        profile = StaticAttentionProfile({
            "schema_version": 2,
            "model": {"tokens": 5, "heads": 2, "head_dimension": 2,
                      "attention_layers": 1},
            "layers": {"0": {"heads": [
                {"selected_scales_bf16": {
                    "q": 0.1, "k": 0.2, "v": 0.3,
                    "probability": 0.01, "av_output_gain": 1.25,
                }},
                {"selected_scales_bf16": {
                    "q": 0.4, "k": 0.5, "v": 0.6,
                    "probability": 0.02, "av_output_gain": 0.75,
                }},
            ]}},
        })

        manifest = bind_profile(model, profile)
        attrs = lambda node: {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        bound = {node.name: node for node in model.graph.node}
        self.assertEqual(manifest["triplets_total"], 2)
        self.assertAlmostEqual(attrs(bound["/blocks.0/attn/MatMul"])["A_scales"][0], 0.1)
        self.assertAlmostEqual(attrs(bound["/blocks.0/attn/Softmax"])["output_scales"][0], 0.01)
        self.assertAlmostEqual(attrs(bound["/blocks.0/attn/MatMul_2"])["B_scales"][0], 0.375)
        self.assertAlmostEqual(attrs(bound["/blocks.0/attn/MatMul_1"])["A_scales"][0], 0.4)
        self.assertNotIn("output_scale", attrs(bound["/blocks.0/attn/Softmax_1"]))

    def test_profile_preserves_pre_rounded_av_accumulator_scale(self):
        from ds_models.static_int8_attention import StaticAttentionProfile

        profile = StaticAttentionProfile({
            "schema_version": 2,
            "model": {"tokens": 4, "heads": 1, "head_dimension": 2,
                      "attention_layers": 1},
            "layers": {"0": {"heads": [{"selected_scales_bf16": {
                "q": 0.1, "k": 0.2, "v": 0.3, "probability": 0.01,
                "av_output_gain": 1.1,
                "av_value_accumulator_scale": 0.328125,
            }}]}},
        })
        scale = profile.scale(0, 0)
        self.assertEqual(scale.effective_av_value_scale, 0.328125)

    def test_chunked_attention_matches_full_attention(self):
        from ds_models.static_int8_attention import chunked_float_attention, plan_query_chunks

        torch.manual_seed(7)
        q = torch.randn(1, 2, 1370, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        expected = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
        actual = chunked_float_attention(q, k, v)
        self.assertEqual(
            [(c.start, c.stop, c.padded_rows) for c in plan_query_chunks(1370)],
            [(0, 256, 256), (256, 512, 256), (512, 768, 256),
             (768, 1024, 256), (1024, 1280, 256), (1280, 1370, 256)],
        )
        self.assertLessEqual(float((expected - actual).abs().max()), 1e-6)

    def test_static_runtime_makes_six_calls_per_head_and_crops_tail(self):
        from ds_models.static_int8_attention import (
            StaticAttentionProfile, StaticInt8AttentionRuntime,
        )

        profile = StaticAttentionProfile({
            "schema_version": 2,
            "model": {"tokens": 1370, "heads": 2, "head_dimension": 8,
                      "attention_layers": 1},
            "layers": {"0": {"heads": [
                {"selected_scales_bf16": {"q": 0.02, "k": 0.03,
                                             "v": 0.04, "probability": 1 / 511}},
                {"selected_scales_bf16": {"q": 0.02, "k": 0.03,
                                             "v": 0.04, "probability": 1 / 511}},
            ]}},
        })
        calls = []

        def runner(layer, head, chunk, query, key, value, scale):
            calls.append((layer, head, chunk.index, chunk.valid_rows,
                          query.shape, key.shape, value.shape))
            return torch.zeros(1, 256, 8).numpy()

        runtime = StaticInt8AttentionRuntime(profile, runner)
        q = torch.randn(1, 2, 1370, 8)
        output = runtime.execute(0, q, torch.randn_like(q), torch.randn_like(q))
        self.assertEqual(tuple(output.shape), (1, 2, 1370, 8))
        self.assertEqual(len(calls), 12)
        self.assertEqual(calls[5][3], 90)
        self.assertEqual(calls[5][4], (1, 256, 8))

    def test_attention_module_routes_qkv_through_static_runtime(self):
        from ds_models.depth_anything_v2_vits import DSAttention4D
        from ds_models.static_int8_attention import (
            StaticAttentionProfile, StaticInt8AttentionRuntime,
        )

        profile = StaticAttentionProfile({
            "schema_version": 2,
            "model": {"tokens": 4, "heads": 2, "head_dimension": 2,
                      "attention_layers": 1},
            "layers": {"0": {"heads": [
                {"selected_scales_bf16": {
                    "q": 0.02, "k": 0.03, "v": 0.04,
                    "probability": 1 / 511,
                }},
                {"selected_scales_bf16": {
                    "q": 0.02, "k": 0.03, "v": 0.04,
                    "probability": 1 / 511,
                }},
            ]}},
        })
        calls = []

        def runner(layer, head, chunk, query, key, value, scale):
            calls.append((layer, head, chunk.index, query.shape))
            return torch.zeros(1, 256, 2).numpy()

        attention = DSAttention4D(4, 2, layer_index=0).eval()
        attention.static_runtime = StaticInt8AttentionRuntime(profile, runner)
        output = attention(torch.randn(1, 4, 4))
        self.assertEqual(tuple(output.shape), (1, 4, 4))
        self.assertEqual(calls, [(0, 0, 0, (1, 256, 2)),
                                 (0, 1, 0, (1, 256, 2))])

    def test_attention_static_export_splits_heads_with_native_tail(self):
        from ds_models.depth_anything_v2_vits import DSAttention4D
        from ds_models.static_int8_attention import StaticAttentionProfile

        profile = StaticAttentionProfile({
            "schema_version": 2,
            "model": {"tokens": 5, "heads": 2, "head_dimension": 2,
                      "attention_layers": 1},
            "layers": {"0": {"heads": [
                {"selected_scales_bf16": {
                    "q": 0.02, "k": 0.03, "v": 0.04,
                    "probability": 1 / 511, "av_output_gain": 1.25,
                    "av_value_accumulator_scale": 0.05,
                }},
                {"selected_scales_bf16": {
                    "q": 0.02, "k": 0.03, "v": 0.04,
                    "probability": 1 / 511, "av_output_gain": 0.75,
                    "av_value_accumulator_scale": 0.03,
                }},
            ]}},
        })
        attention = DSAttention4D(4, 2, layer_index=0).eval()
        attention.static_export_profile = profile
        value = torch.randn(1, 5, 4)
        output = attention(value)
        self.assertEqual(tuple(output.shape), (1, 5, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_phase_deconvolution_is_exact(self):
        from ds_models.depth_anything_v2_vits import DSConvTranspose2d

        source = torch.nn.ConvTranspose2d(2, 3, kernel_size=4, stride=4, bias=True)
        replacement = DSConvTranspose2d(source).eval()
        value = torch.randn(1, 2, 5, 7)
        with torch.no_grad():
            expected = source(value)
            actual = replacement(value)
        self.assertEqual(tuple(expected.shape), tuple(actual.shape))
        self.assertLessEqual(float((expected - actual).abs().max()), 1e-6)

    def test_runtime_class_token_is_exact_and_has_dynamic_dependency(self):
        from ds_models.depth_anything_v2_vits import _install_runtime_cls_token

        class PatchEmbed(torch.nn.Module):
            def forward(self, value):
                return value.flatten(2).transpose(1, 2)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cls_token = torch.nn.Parameter(torch.randn(1, 1, 3))
                self.mask_token = torch.nn.Parameter(torch.randn(3))
                self.register_tokens = None
                self.patch_embed = PatchEmbed()

            def interpolate_pos_encoding(self, x, width, height):
                return torch.zeros_like(x)

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pretrained = Backbone()

        wrapper = Wrapper().eval()
        _install_runtime_cls_token(wrapper)
        image = torch.randn(1, 3, 2, 2, requires_grad=True)
        patches = wrapper.pretrained.patch_embed(image)
        output = wrapper.pretrained.prepare_tokens_with_masks(image)
        self.assertTrue(torch.equal(output[:, :1], wrapper.pretrained.cls_token))
        self.assertTrue(torch.equal(output[:, 1:], patches))
        output[:, :1].sum().backward()
        self.assertIsNotNone(image.grad)

    def test_vits_ds_adapter_has_fixed_contract_and_checkpoint(self):
        from ds_models import depth_anything_v2_vits as adapter

        self.assertEqual(adapter.ifmap_sz, (3, 518, 518))
        self.assertEqual(adapter.input_layouts, ("CHW",))
        self.assertEqual(adapter.input_names, ("input0",))

        if not Path(adapter.CHECKPOINT).is_file():
            self.skipTest("official Depth Anything checkpoint is external")

        model = adapter.Model()
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 518, 518))

        self.assertEqual(tuple(output.shape), (1, 518, 518))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(Path(adapter.CHECKPOINT).is_file())
