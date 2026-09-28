import tempfile
import unittest
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper


class DepthAnythingOnnxAuditTest(unittest.TestCase):
    def test_erf_pwl_is_finite_and_covers_input_range(self):
        from tools.quantize_depth_anything_v2_ds import erf_pwl

        slopes, intercepts, change_pts = erf_pwl(
            torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)
        )
        self.assertEqual(len(slopes), len(intercepts))
        self.assertEqual(len(intercepts), len(change_pts))
        self.assertTrue(all(torch.isfinite(torch.tensor(slopes))))
        self.assertTrue(all(torch.isfinite(torch.tensor(intercepts))))
        self.assertGreaterEqual(change_pts[-1], 3.0)

    def test_reference_numpy_copy_owns_memory(self):
        from tools.export_depth_anything_v2_ds import detach_numpy

        value = detach_numpy(torch.ones(1, 2, 3))
        self.assertTrue(value.flags["OWNDATA"])
        self.assertTrue(value.flags["C_CONTIGUOUS"])

    def test_audit_reports_static_signatures_and_operator_counts(self):
        from tools.export_depth_anything_v2_ds import audit_graph

        graph = helper.make_graph(
            [helper.make_node("Identity", ["input0"], ["depth"])],
            "fixture",
            [helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 3, 518, 518])],
            [helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 3, 518, 518])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.onnx"
            onnx.save(model, path)
            audit = audit_graph(path)

        self.assertEqual(audit["inputs"], {"input0": [1, 3, 518, 518]})
        self.assertEqual(audit["outputs"], {"depth": [1, 3, 518, 518]})
        self.assertEqual(audit["operators"], {"Identity": 1})
        self.assertEqual(audit["dynamic_tensors"], [])
