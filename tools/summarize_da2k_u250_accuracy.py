#!/usr/bin/env python3
"""Build the compact r80 DA-2K calibration and U250 gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as values:
        return values["depth"].astype(np.float64).ravel()


def metrics(value: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(value - reference)
    return {
        "rel_l2": float(np.linalg.norm(value - reference) / np.linalg.norm(reference)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((value - reference) ** 2))),
        "cosine": float(np.dot(value, reference)
                        / (np.linalg.norm(value) * np.linalg.norm(reference))),
        "pearson": float(np.corrcoef(value, reference)[0, 1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--fp32-trace", type=Path, required=True)
    parser.add_argument("--r79-demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_dir
    baseline = json.loads((root / "r79_da2k_board_baseline.json").read_text())
    selected = json.loads((root / "finalconv_selected_s0p350000000/evaluation.json").read_text())
    scales = json.loads((root / "u250_scale_analysis.json").read_text())
    final_boundary = next(item for item in scales["priorities"]
                          if item.get("source_node", "").endswith("output_conv2.2/Conv"))

    with np.load(args.fp32_trace, allow_pickle=False) as values:
        reference = values["depth"].astype(np.float64).ravel()
    r79_demo = metrics(depth(args.r79_demo), reference)
    r80_demo_path = root / "finalconv_demo_gate/0p350000000/demo05.npz"
    r80_demo = metrics(depth(r80_demo_path), reference)

    summaries = [json.loads(path.read_text()) for path in sorted(
        (root / "formal_post_reboot_s0p350000000").glob(
            "demo_run*/demo05.summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected three formal demo summaries, got {len(summaries)}")

    qualification = json.loads((root /
        "formal_post_reboot_s0p350000000/native_codec_all_oracle.json").read_text())
    manifest = args.candidate_dir / "resident_kernel_bank_manifest.json"
    bank = args.candidate_dir / "depthanything_u250_resident_kernel_bank.bin"
    contract = args.candidate_dir / "depthanything_u250_runtime_contract.json"
    plan = args.candidate_dir / "depthanything_u250_host_plan.json"
    report = {
        "schema": "depthanything-u250-da2k-accuracy-r80-v1",
        "version": "r80_da2k_finalconv_s0p350000000",
        "calibration": {
            "samples": 512,
            "tuning_samples": 128,
            "holdout_samples": 393,
            "scenes": 8,
            "selected_scale": 0.35,
            "original_scale": final_boundary["current_scale"],
            "p999_scale": final_boundary["p999_scale"],
            "histogram_mse_scale": final_boundary["optimal_scale"],
            "original_clipping_fraction": final_boundary["current_clipping_fraction"],
            "selection_policy": "tuning continuous metrics; no DA-2K pair regression; independent holdout and demo05 gates",
        },
        "da2k_board_32": {"r79": baseline["all"], "r80": selected["all"]},
        "da2k_holdout_16": {
            "r79": baseline["by_split"]["holdout"],
            "r80": selected["by_split"]["holdout"],
        },
        "demo05": {"r79": r79_demo, "r80": r80_demo},
        "formal_board_gate": {
            "runs": len(summaries),
            "bit_identical": len({item["output_sha256"] for item in summaries}) == 1,
            "output_sha256": summaries[0]["output_sha256"],
            "resident_bank_sha256": summaries[0]["resident_bank_sha256"],
            "npu_ms_median": statistics.median(item["npu_ms_total"] for item in summaries),
            "wall_ms_median": statistics.median(item["wall_ms"] for item in summaries),
            "wall_ms_runs": [item["wall_ms"] for item in summaries],
            "npu_dispatches": summaries[0]["npu_calls"],
            "static_reloads": summaries[0]["static_reloads"],
            "vendor_pack_calls": summaries[0]["vendor_pack_calls"],
            "vendor_unpack_calls": summaries[0]["vendor_unpack_calls"],
        },
        "native_codec_qualification": {
            "manifest_sha256": qualification["manifest_sha256"],
            "descriptors": len(qualification["descriptors"]),
            "all_native_exact": all(item.get("native_exact") is True
                                    for item in qualification["descriptors"]),
        },
        "package": {
            "bank_sha256": sha256(bank), "manifest_sha256": sha256(manifest),
            "contract_sha256": sha256(contract), "host_plan_sha256": sha256(plan),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
