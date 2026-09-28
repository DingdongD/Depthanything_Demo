#!/usr/bin/env python3
"""Package deterministic block-tail probes and produce float ONNX goldens."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def clean_for_ort(source: Path, target: Path) -> None:
    model = onnx.load(str(source), load_external_data=True)
    for node in model.graph.node:
        # These extracted graphs only use attribute-free standard operators;
        # every attached attribute is a DS quantization/compiler contract.
        del node.attribute[:]
    onnx.save(model, str(target))


def package(
    name: str,
    compiled: Path,
    model_path: Path,
    output_root: Path,
    inputs_physical: list[np.ndarray],
    inputs_float: dict[str, np.ndarray],
) -> None:
    case = output_root / ("board_" + name)
    case.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled / name / (name + "_ddr.bin"), case / (name + "_ddr.bin"))
    shutil.copy2(compiled / name / (name + "_cfg.txt"), case / (name + "_cfg.txt"))
    clean = case / (name + "_float.onnx")
    clean_for_ort(model_path, clean)
    session = ort.InferenceSession(str(clean), providers=["CPUExecutionProvider"])
    golden = session.run(None, inputs_float)[0]
    np.save(case / (name + "_golden.npy"), golden.astype(np.float32))
    for index, value in enumerate(inputs_physical):
        np.savez(case / f"logical_input{index}.npz", input=value)
    print(case)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    shape = (1, 1370, 384)

    attention_scale = 0.0528128482401371
    attention_code = np.clip(np.rint(rng.normal(0.0, 18.0, shape)), -90, 90).astype(np.int8)
    residual = rng.normal(0.0, 0.8, shape).astype(np.float32)
    package(
        "post_attention_l00", args.compiled_dir,
        args.model_dir / "post_attention_l00.onnx", args.output_dir,
        [attention_code[:, None], residual[:, None]],
        {
            "attention_input": attention_code.astype(np.float32) * attention_scale,
            "residual_input": residual,
        },
    )

    fc1_scale = 1.1491037607192993
    norm_code = np.clip(np.rint(rng.normal(0.0, 2.0, shape)), -8, 8).astype(np.int8)
    package(
        "mlp_l00", args.compiled_dir,
        args.model_dir / "mlp_l00.onnx", args.output_dir,
        [norm_code[:, None]],
        {"norm2_input": norm_code.astype(np.float32) * fc1_scale},
    )
    for chunk in range(6):
        fc1_slice_name = f"mlp_fc1_l00_c{chunk:02d}"
        if (args.compiled_dir / fc1_slice_name / (fc1_slice_name + "_ddr.bin")).is_file():
            package(
                fc1_slice_name, args.compiled_dir,
                args.model_dir / (fc1_slice_name + ".onnx"), args.output_dir,
                [norm_code[:, None]],
                {"norm2_input": norm_code.astype(np.float32) * fc1_scale},
            )
        partial_name = f"mlp_partial_l00_c{chunk:02d}"
        if not (args.compiled_dir / partial_name / (partial_name + "_ddr.bin")).is_file():
            continue
        package(
            partial_name, args.compiled_dir,
            args.model_dir / (partial_name + ".onnx"), args.output_dir,
            [norm_code[:, None]],
            {"norm2_input": norm_code.astype(np.float32) * fc1_scale},
        )
    fc2_name = "mlp_fc2_l00"
    if (args.compiled_dir / fc2_name / (fc2_name + "_ddr.bin")).is_file():
        fc2_scale = 0.07290246337652206
        gelu_shape = (1, 1370, 1536)
        gelu_code = np.clip(
            np.rint(rng.normal(2.0, 10.0, gelu_shape)), -24, 80
        ).astype(np.int8)
        package(
            fc2_name, args.compiled_dir,
            args.model_dir / (fc2_name + ".onnx"), args.output_dir,
            [gelu_code[:, None]],
            {"gelu_input": gelu_code.astype(np.float32) * fc2_scale},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
