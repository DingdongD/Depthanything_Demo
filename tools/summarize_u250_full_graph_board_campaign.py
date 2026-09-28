#!/usr/bin/env python3
"""Summarize the unified r199 board gates, latency, and family ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LATENCY_FIELDS = (
    "frame_wall_ms", "request_wall_ms", "npu_ms", "h2c_ms", "c2h_ms",
    "pack_ms", "unpack_ms",
)


def load_depth(root: Path, sample_id: str) -> np.ndarray:
    with np.load(root / Path(sample_id).with_suffix(".npz"), allow_pickle=False) as data:
        return np.asarray(data["depth"], dtype=np.float64)


def relative_l2(root: Path, reference: Path, samples: list[dict]) -> float:
    error_sq = reference_sq = 0.0
    for sample in samples:
        actual = load_depth(root, sample["sample_id"])
        target = load_depth(reference, sample["sample_id"])
        error = actual - target
        error_sq += float(np.vdot(error, error))
        reference_sq += float(np.vdot(target, target))
    return float(np.sqrt(error_sq / max(reference_sq, 1e-30)))


def latency(summary: dict) -> dict:
    result = {}
    for name in LATENCY_FIELDS:
        values = np.asarray([item[name] for item in summary["samples"]], np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--board-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    payload = {
        "schema": "depthanything-u250-full-graph-board-campaign-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(manifest["samples"]),
        "shapes": {},
    }
    for shape in (280, 518):
        baseline_root = args.board_root / f"{shape}_baseline_160"
        candidate_root = args.board_root / f"{shape}_candidate_160"
        baseline_batch = json.loads(
            (baseline_root / "resident_batch_summary.json").read_text()
        )
        candidate_batch = json.loads(
            (candidate_root / "resident_batch_summary.json").read_text()
        )
        baseline_rows = {item["request_id"]: item
                         for item in baseline_batch["samples"]}
        candidate_rows = {item["request_id"]: item
                          for item in candidate_batch["samples"]}
        paired = {}
        for field in LATENCY_FIELDS:
            values = np.asarray([
                candidate_rows[key][field] - baseline_rows[key][field]
                for key in baseline_rows
            ], np.float64)
            paired[field] = {
                "mean_candidate_minus_baseline": float(values.mean()),
                "p50_candidate_minus_baseline": float(np.percentile(values, 50)),
                "p95_candidate_minus_baseline": float(np.percentile(values, 95)),
                "candidate_faster_samples": int(np.count_nonzero(values < 0)),
            }
        reference = args.reference_root / str(shape)
        validation = {
            domain: [sample for sample in manifest["samples"]
                     if sample["split"] == "validation"
                     and sample["domain"] == domain][:(8 if domain == "nyu" else None)]
            for domain in ("nyu", "da2k")
        }
        family = {}
        for name in ("attention", "encoder_activation", "decoder"):
            root = (args.board_root / "family_val16"
                    / f"{shape}_family_{name}_val16")
            family[name] = {
                domain: {
                    "baseline_relative_l2": relative_l2(
                        baseline_root, reference, samples
                    ),
                    "candidate_relative_l2": relative_l2(root, reference, samples),
                }
                for domain, samples in validation.items()
            }
            for result in family[name].values():
                result["candidate_over_baseline"] = (
                    result["candidate_relative_l2"]
                    / result["baseline_relative_l2"]
                )
        gate = json.loads(
            (args.board_root / f"{shape}_final_depth_gate.json").read_text()
        )
        payload["shapes"][str(shape)] = {
            "deployment_status": "qualified" if gate["accepted"] else "rejected",
            "full_final_depth_gate": gate,
            "latency": {
                "baseline": latency(baseline_batch),
                "candidate": latency(candidate_batch),
                "paired": paired,
            },
            "runtime_invariants": {
                "candidate_samples": candidate_batch["measured_samples"],
                "persistent_dma": candidate_batch["persistent_dma"],
                "all_bank_reused": all(
                    item["resident_bank_reused"] for item in candidate_batch["samples"]
                ),
                "all_cpp_runtime_reused": all(
                    item["cpp_runtime_reused"] for item in candidate_batch["samples"]
                ),
            },
            "family_validation_screen": family,
        }
    payload["deployment_status"] = (
        "qualified" if all(item["deployment_status"] == "qualified"
                           for item in payload["shapes"].values()) else "rejected"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "deployment_status": payload["deployment_status"],
        "shapes": {shape: {
            "status": item["deployment_status"],
            "candidate_frame_mean_ms": item["latency"]["candidate"]["frame_wall_ms"]["mean"],
            "candidate_frame_p95_ms": item["latency"]["candidate"]["frame_wall_ms"]["p95"],
        } for shape, item in payload["shapes"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
