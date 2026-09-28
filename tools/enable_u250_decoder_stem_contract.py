#!/usr/bin/env python3
"""Replace four decoder project Conv calls with fused resident token stems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYERS = (2, 5, 8, 11)


def enable_stems(
    contract: dict, manifest: dict, bank: dict,
    projects: tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    selected = set(projects)
    if not selected or not selected <= {0, 1, 2, 3}:
        raise ValueError("projects must be a non-empty subset of 0..3")
    by_project = {}
    for item in manifest["kernels"]:
        by_project.setdefault(int(item["project_index"]), []).append(item)
    if not selected <= set(by_project):
        raise ValueError("stem manifest does not cover every selected project")
    resident = {item["name"] for item in bank["cases"]}
    expected = {
        item["name"] for project, values in by_project.items()
        if project in selected for item in values
    }
    if expected - resident:
        raise ValueError(f"resident bank is missing stems: {sorted(expected - resident)}")

    predecessors = {
        index: [
            "/norm" + ("" if index == 0 else f"_{index}") + "/LayerNormalization",
            f"/Constant_{5 + 4 * index}", f"/Constant_{6 + 4 * index}",
            f"/Constant_{7 + 4 * index}", f"/Constant_{8 + 4 * index}",
            f"/Slice_{index + 1}",
            "/depth_head/Transpose" + ("" if index == 0 else f"_{index}"),
            "/depth_head/Constant" + ("" if index == 0 else f"_{index}"),
            "/depth_head/Reshape" + ("" if index == 0 else f"_{index}"),
        ] for index in range(4)
    }
    stem_input = manifest.get("stem_input", "capture")
    replaced = 0
    decoder = []
    for step in contract["decoder"]:
        source = step.get("source_node", "")
        if not source.startswith("/depth_head/projects."):
            decoder.append(step)
            continue
        index = int(source.split("projects.", 1)[1].split("/", 1)[0])
        if index not in selected:
            decoder.append(step)
            continue
        kernels = sorted(by_project[index], key=lambda item: item["channel_start"])
        fused_nodes = predecessors[index]
        if stem_input == "normalized":
            fused_nodes = fused_nodes[1:]
        elif stem_input == "patches":
            fused_nodes = fused_nodes[-3:]
        decoder.append({
            **step,
            "fused_decoder_stem": True,
            "capture_layer": LAYERS[index],
            "stem_input": stem_input,
            "input_tensor": kernels[0].get(
                "input_tensor", f"/blocks.{LAYERS[index]}/Add_1_output_0"
            ),
            "input_shape": kernels[0].get("input_shape", [1, 1370, 384]),
            "input_dtype": "BF16",
            "input_quantization": None,
            "kernels": [{
                "name": item["name"],
                "channel_start": item["channel_start"],
                "channel_end": item["channel_end"],
                "output_shape": item["output_shape"],
            } for item in kernels],
            "fused_host_nodes": fused_nodes,
            "affine": manifest.get("affine_mode", "fold"),
        })
        replaced += 1
    if replaced != len(selected):
        raise ValueError(
            f"expected {len(selected)} decoder project calls, replaced {replaced}"
        )
    result = {**contract, "decoder": decoder}
    result["bank"] = {
        **result.get("bank", {}), "bytes": bank["bank_size_bytes"],
        "sha256": bank["bank_sha256"],
        "resident_kernels": len(bank["cases"]),
        "required_fm_io_bytes": bank["required_fm_io_bytes"],
        "shared_fm_workspace_bytes": bank["shared_fm_workspace_bytes"],
    }
    totals = dict(result["execution_totals"])
    totals["resident_kernel_variants"] = len(bank["cases"])
    result["execution_totals"] = totals
    if stem_input == "capture":
        result["decoder_capture_policy"] = {
            "layers": [LAYERS[index] for index in sorted(selected)],
            "storage": "resident_BF16_NDWC",
            "consumer": "fused LayerNorm/layout/project-Conv stems",
            "host_materialization": False,
        }
    result["decoder_stem_policy"] = {
        "projects": sorted(selected), "input": stem_input,
        "fused_ops": ((["Slice"] if stem_input != "patches" else [])
                      + ["Transpose", "Reshape", "Conv"]),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stem-manifest", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projects", default="0,1,2,3",
        help="comma-separated decoder project indices to fuse",
    )
    args = parser.parse_args()
    projects = tuple(int(value) for value in args.projects.split(",") if value)
    result = enable_stems(
        json.loads(args.contract.read_text()),
        json.loads(args.stem_manifest.read_text()),
        json.loads(args.bank_manifest.read_text()),
        projects,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "bank_sha256": result["bank"]["sha256"],
        "resident_kernels": result["bank"]["resident_kernels"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
