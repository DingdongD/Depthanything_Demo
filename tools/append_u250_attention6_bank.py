#!/usr/bin/env python3
"""Append twelve six-head attention programs to a production resident bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

try:
    from .append_u250_fused_qkv_attention_bank import (
        append_programs, validate_and_rebind_codec_report,
    )
    from .replace_u250_fc1_pairs_in_bank import parse_cfg
except ImportError:
    from append_u250_fused_qkv_attention_bank import (
        append_programs, validate_and_rebind_codec_report,
    )
    from replace_u250_fc1_pairs_in_bank import parse_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument(
        "--layer-source", action="append", default=[],
        help="override one compiler directory as LAYER=DIRECTORY",
    )
    parser.add_argument(
        "--layers",
        help="comma-separated layer subset; defaults to all twelve",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        selected_layers = (
            set(range(12)) if args.layers is None else
            {int(item) for item in args.layers.split(",") if item}
        )
    except ValueError as exc:
        raise ValueError("--layers must be comma-separated integers") from exc
    if not selected_layers or not selected_layers <= set(range(12)):
        raise ValueError("--layers must select values within [0, 11]")

    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    source_manifest_path = (
        args.source_package / "resident_kernel_bank_manifest.json"
    )
    source_report_path = args.source_package / "native_codec_report_active.json"
    if not source_report_path.exists():
        source_report_path = args.source_package / "native_codec_report.json"
    manifest = json.loads(source_manifest_path.read_text())
    bank = (args.source_package / manifest["bank_file"]).read_bytes()
    if len(bank) != int(manifest["bank_size_bytes"]):
        raise ValueError("source resident bank extent differs from manifest")

    overrides = {}
    for value in args.layer_source:
        layer_text, separator, directory = value.partition("=")
        if not separator:
            raise ValueError("--layer-source must use LAYER=DIRECTORY")
        layer = int(layer_text)
        if not 0 <= layer < 12 or layer in overrides:
            raise ValueError(f"invalid or duplicate fused layer: {layer}")
        overrides[layer] = Path(directory)

    programs = []
    for layer in sorted(selected_layers):
        directory = overrides.get(layer, args.compiled_root / f"l{layer:02d}")
        stem = f"attention6_l{layer:02d}_a8_to_12xbf16"
        cfg = directory / f"{stem}_cfg.txt"
        binary = directory / f"{stem}_ddr.bin"
        metadata = parse_cfg(cfg)
        if (len(metadata["inputs"]) != 24
                or any(item["bitdepth"] != 8
                       for item in metadata["inputs"])
                or len(metadata["outputs"]) != 12
                or any(item["bitdepth"] != 16
                       for item in metadata["outputs"])):
            raise ValueError(f"layer {layer}: unexpected attention6 tensor ABI")
        programs.append({
            "name": f"attention6_l{layer:02d}",
            "layer": layer,
            "cfg": cfg,
            "binary": binary,
            "payload": binary.read_bytes(),
            "metadata": metadata,
        })

    updated, image = append_programs(
        manifest, bank, programs, int(manifest.get("alignment_bytes", 4096))
    )
    updated["attention6_append"] = updated.pop("fused_qkv_attention_append")
    updated["attention6_append"]["policy"] = (
        "append-six-head-attention-preserve-bf16-qkv-host-a8-no-gain"
    )
    for record in updated["cases"]:
        if record["name"].startswith("attention6_l"):
            record["group"] = "six-head-attention/no-gain"
            record["output_order"] = "head-major"

    shutil.copytree(args.source_package, args.output_dir)
    (args.output_dir / updated["bank_file"]).write_bytes(image)
    manifest_path = args.output_dir / "resident_kernel_bank_manifest.json"
    manifest_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    cfg_dir = args.output_dir / "cfg"
    for program in programs:
        shutil.copyfile(
            program["cfg"], cfg_dir / f"{program['name']}_cfg.txt"
        )

    contract_path = args.output_dir / "depthanything_u250_runtime_contract.json"
    contract = json.loads(contract_path.read_text())
    for layer, block in enumerate(contract["encoder"]):
        if layer not in selected_layers:
            continue
        heads = block["attention"]["heads"]
        call_counts = {len(head["calls"]) for head in heads}
        if len(heads) != 6 or len(call_counts) != 1 or not next(iter(call_counts)):
            raise ValueError(
                f"layer {layer}: expected an equal positive call count for six heads"
            )
        calls_per_head = next(iter(call_counts))
        block["attention_fused"] = {
            "kernel": f"attention6_l{layer:02d}",
            "heads": 6,
            "calls_per_head": calls_per_head,
            "inputs_per_head": 4,
            "input_order": "head-major-q0-k-v-q1",
            "outputs_per_head": 2,
            "output_order": "head-major",
            "valid_widths_head_major": [
                width
                for head in heads
                for call in head["calls"]
                for width in (
                    int(call["q0_rows"][1]) - int(call["q0_rows"][0]),
                    int(call["q1_rows"][1]) - int(call["q1_rows"][0]),
                )
            ],
            "preserve_qkv_path": "BF16-output-to-host-static-A8",
            "amplitude_gain": 1.0,
        }
    contract["bank"].update({
        "bytes": len(image),
        "sha256": updated["bank_sha256"],
        "resident_kernels": len(updated["cases"]),
    })
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    validate_and_rebind_codec_report(
        source_report_path, source_manifest_path, manifest_path, cfg_dir,
        args.output_dir / "native_codec_report_active.json",
    )
    print(json.dumps({
        "appended_kernels": len(programs),
        "bank_bytes": len(image),
        "bank_sha256": updated["bank_sha256"],
        "output_dir": str(args.output_dir),
        "resident_kernels": len(updated["cases"]),
        "shared_fm_base_units": updated["shared_fm_base_units"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
