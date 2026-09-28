#!/usr/bin/env python3
"""Create deterministic U250 probes for resident decoder convolution kernels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import external_data_helper, helper


STANDARD_CONV_ATTRIBUTES = {
    "auto_pad", "dilations", "group", "kernel_shape", "pads", "strides",
}
STANDARD_DEPTH_TO_SPACE_ATTRIBUTES = {"blocksize", "mode"}


def clean_for_ort(source: Path, target: Path) -> tuple[float, list[int], float | None]:
    model = onnx.load(str(source.resolve()), load_external_data=True)
    node = model.graph.node[0]
    scale = next(
        float(helper.get_attribute_value(value)[0])
        for value in node.attribute if value.name == "input_scales"
    )
    output_scale = next((
        float(helper.get_attribute_value(value)[0])
        for value in model.graph.node[-1].attribute
        if value.name == "output_scales"
    ), None)
    for graph_node in model.graph.node:
        if graph_node.op_type == "Conv":
            allowed = STANDARD_CONV_ATTRIBUTES
        elif graph_node.op_type == "DepthToSpace":
            allowed = STANDARD_DEPTH_TO_SPACE_ATTRIBUTES
        else:
            continue
        kept = [value for value in graph_node.attribute
                if value.name in allowed]
        del graph_node.attribute[:]
        graph_node.attribute.extend(kept)
    external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model, str(target), save_as_external_data=False)
    shape = [int(value.dim_value)
             for value in model.graph.input[0].type.tensor_type.shape.dim]
    return scale, shape, output_scale


def make_code(rng: np.random.Generator, shape: list[int]) -> np.ndarray:
    # Exercise both signs and a useful fraction of the calibrated A8 range.
    return np.clip(np.rint(rng.normal(0.0, 24.0, shape)), -100, 100).astype(np.int8)


def package_single(name: str, model_dir: Path, compiled_dir: Path,
                   output_dir: Path, rng: np.random.Generator) -> None:
    case = output_dir / ("board_" + name)
    case.mkdir(parents=True, exist_ok=True)
    compiled = compiled_dir / name
    shutil.copy2(compiled / (name + "_cfg.txt"), case / (name + "_cfg.txt"))
    clean = case / (name + "_float.onnx")
    scale, shape, output_scale = clean_for_ort(
        model_dir / (name + ".onnx"), clean
    )
    code = make_code(rng, shape)
    session = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input0": code.astype(np.float32) * scale})[0]
    np.savez(case / "logical_input0.npz", input=code)
    np.save(case / (name + "_golden.npy"), output.astype(np.float32))
    (case / "case.json").write_text(json.dumps({
        "name": name, "input_scale": scale, "input_shape": shape,
        "output_shape": list(output.shape), "output_scale": output_scale,
    }, indent=2, sort_keys=True) + "\n")
    print(case)


def package_tiled(index: int, model_dir: Path, compiled_dir: Path,
                  output_dir: Path, rng: np.random.Generator) -> None:
    base = f"decoder_conv_{index:02d}"
    case = output_dir / ("tiled_" + base)
    case.mkdir(parents=True, exist_ok=True)
    clean = case / (base + "_float.onnx")
    scale, shape, output_scale = clean_for_ort(
        model_dir / (base + ".onnx"), clean
    )
    code = make_code(rng, shape)
    session = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input0": code.astype(np.float32) * scale})[0]
    np.savez(case / "logical_full_input.npz", input=code)
    np.save(case / (base + "_golden.npy"), output.astype(np.float32))
    cfg_names = []
    for suffix in ("tile_first", "tile_middle", "tile_last", "tile37", "tile74"):
        name = base + "_" + suffix
        source = compiled_dir / name / (name + "_cfg.txt")
        if source.is_file():
            shutil.copy2(source, case / source.name)
            cfg_names.append(name)
    if not cfg_names:
        raise FileNotFoundError(f"no tile cfg found for {base}")
    (case / "case.json").write_text(json.dumps({
        "name": base, "input_scale": scale, "input_shape": shape,
        "output_shape": list(output.shape), "output_scale": output_scale,
        "tile_cases": cfg_names,
    }, indent=2, sort_keys=True) + "\n")
    print(case)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--probe", action="append",
                        help="package only these single-kernel probes")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.probe:
        for name in args.probe:
            package_single(name, args.model_dir, args.compiled_dir,
                           args.output_dir, rng)
        return 0
    for name in (
        "decoder_conv_11", "decoder_conv_24_tile_middle",
        "decoder_conv_28_tile37", "decoder_conv_30_tile_middle",
        "decoder_conv_31_tile74",
    ):
        package_single(name, args.model_dir, args.compiled_dir, args.output_dir, rng)
    for index in (24, 28, 30, 31):
        package_tiled(index, args.model_dir, args.compiled_dir, args.output_dir, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
