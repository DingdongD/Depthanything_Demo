"""Create tiny ONNX graphs for DS-Compiler capability probes."""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_layernorm(path: Path):
    x = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 4, 1, 8])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 1, 8])
    scale = numpy_helper.from_array(np.ones((8,), np.float32), "scale")
    bias = numpy_helper.from_array(np.zeros((8,), np.float32), "bias")
    node = helper.make_node(
        "LayerNormalization", ["input0", "scale", "bias"], ["output0"],
        name="layernorm", axis=-1, epsilon=1e-5,
    )
    graph = helper.make_graph([node], "minimal_layernorm", [x], [y], [scale, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_resize(path: Path):
    x = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 16, 16])
    roi = numpy_helper.from_array(np.array([], np.float32), "roi")
    scales = numpy_helper.from_array(np.array([], np.float32), "scales")
    sizes = numpy_helper.from_array(np.array([1, 3, 16, 16], np.int64), "sizes")
    node = helper.make_node(
        "Resize", ["input0", "roi", "scales", "sizes"], ["output0"],
        name="resize", mode="nearest", coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph([node], "minimal_resize", [x], [y], [roi, scales, sizes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_layernorm_decomposed(path: Path):
    x = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4, 8])
    scale = numpy_helper.from_array(np.ones((8,), np.float32), "scale")
    bias = numpy_helper.from_array(np.zeros((8,), np.float32), "bias")
    eps = numpy_helper.from_array(np.array(1e-5, np.float32), "epsilon")
    axes = numpy_helper.from_array(np.array([-1], np.int64), "axes")
    nodes = [
        helper.make_node("ReduceMean", ["input0", "axes"], ["mean"], name="mean", keepdims=1),
        helper.make_node("Sub", ["input0", "mean"], ["centered"], name="center"),
        helper.make_node("Mul", ["centered", "centered"], ["squared"], name="square"),
        helper.make_node("ReduceMean", ["squared", "axes"], ["variance"], name="variance", keepdims=1),
        helper.make_node("Add", ["variance", "epsilon"], ["var_eps"], name="add_eps"),
        helper.make_node("Sqrt", ["var_eps"], ["std"], name="sqrt"),
        helper.make_node("Div", ["centered", "std"], ["normalized"], name="divide"),
        helper.make_node("Mul", ["normalized", "scale"], ["scaled"], name="scale"),
        helper.make_node("Add", ["scaled", "bias"], ["output0"], name="bias"),
    ]
    graph = helper.make_graph(nodes, "minimal_layernorm_decomposed", [x], [y], [scale, bias, eps, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["layernorm", "layernorm_decomposed", "resize"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "layernorm":
        make_layernorm(args.output)
    elif args.kind == "layernorm_decomposed":
        make_layernorm_decomposed(args.output)
    else:
        make_resize(args.output)


if __name__ == "__main__":
    main()
