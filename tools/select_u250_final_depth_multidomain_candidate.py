#!/usr/bin/env python3
"""Select a full-model candidate under strict multi-domain final-depth gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.u250_calibration_manifest import (
        file_sha256,
        load_manifest,
        samples_for_shape,
    )
except ModuleNotFoundError:  # Direct execution keeps tools/ on sys.path.
    from u250_calibration_manifest import file_sha256, load_manifest, samples_for_shape


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, root = value.partition("=")
    if not separator or not name or not root:
        raise argparse.ArgumentTypeError("candidate must use NAME=OUTPUT_ROOT")
    return name, Path(root)


def output_path(root: Path, sample_id: str) -> Path:
    return root / Path(sample_id).with_suffix(".npz")


def package_provenance(root: Path) -> dict:
    manifest_path = root / "resident_kernel_bank_manifest.json"
    contract_path = root / "depthanything_u250_runtime_contract.json"
    host_plan_path = root / "depthanything_u250_host_plan.json"
    codec_path = root / "native_codec_report_active.json"
    required = (manifest_path, contract_path, host_plan_path, codec_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"candidate package is incomplete: {missing}")
    manifest = json.loads(manifest_path.read_text())
    codec = json.loads(codec_path.read_text())
    bank_path = root / manifest["bank_file"]
    if not bank_path.is_file():
        raise ValueError(f"candidate package bank is missing: {bank_path}")
    bank_sha256 = file_sha256(bank_path)
    if bank_sha256 != manifest["bank_sha256"]:
        raise ValueError(f"candidate package bank digest mismatch: {root}")
    return {
        "root": str(root.resolve()),
        "bank_sha256": bank_sha256,
        "resident_manifest_sha256": file_sha256(manifest_path),
        "runtime_contract_sha256": file_sha256(contract_path),
        "host_plan_sha256": file_sha256(host_plan_path),
        "codec_report_sha256": file_sha256(codec_path),
        "codec_extension_sha256": codec["extension_sha256"],
    }


def group_metrics(records: list[dict], root: Path,
                  reference_root: Path) -> dict[str, dict]:
    values = {}
    for item in records:
        group = f"{item['split']}:{item['domain']}"
        metric = values.setdefault(group, {
            "error_sq": 0.0, "reference_sq": 0.0, "elements": 0,
        })
        with np.load(output_path(root, item["sample_id"]), allow_pickle=False) as out, \
             np.load(output_path(reference_root, item["sample_id"]),
                     allow_pickle=False) as ref:
            actual = np.asarray(out["depth"], dtype=np.float64)
            target = np.asarray(ref["depth"], dtype=np.float64)
        if (actual.shape != target.shape or not np.isfinite(actual).all()
                or not np.isfinite(target).all()):
            raise ValueError(f"invalid final depth for {item['sample_id']}")
        error = actual - target
        metric["error_sq"] += float(np.vdot(error, error))
        metric["reference_sq"] += float(np.vdot(target, target))
        metric["elements"] += int(target.size)
    return {
        name: {
            "relative_l2": float(np.sqrt(
                item["error_sq"] / max(item["reference_sq"], 1e-30)
            )),
            "rmse": float(np.sqrt(item["error_sq"] / max(item["elements"], 1))),
            "elements": item["elements"],
        }
        for name, item in sorted(values.items())
    }


def rank_candidates(baseline: dict[str, dict], candidates: dict[str, dict],
                    tolerance: float) -> tuple[str, dict]:
    evaluated = {}
    for name, metrics in candidates.items():
        ratios = {
            group: metrics[group]["relative_l2"]
            / max(baseline[group]["relative_l2"], 1e-30)
            for group in baseline
        }
        feasible = all(value <= 1.0 + tolerance for value in ratios.values())
        evaluated[name] = {
            "metrics": metrics,
            "ratios_to_baseline": ratios,
            "worst_group_ratio": max(ratios.values()),
            "mean_group_ratio": float(np.mean(list(ratios.values()))),
            "feasible": feasible,
        }
    feasible = [name for name, item in evaluated.items() if item["feasible"]]
    if not feasible:
        raise ValueError("baseline must be included as a feasible candidate")
    selected = min(feasible, key=lambda name: (
        evaluated[name]["mean_group_ratio"],
        evaluated[name]["worst_group_ratio"],
        name != "baseline", name,
    ))
    return selected, evaluated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shape", type=int, choices=(280, 518), required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate,
                        default=[])
    parser.add_argument(
        "--candidate-package", action="append", type=parse_candidate,
        default=[], help="optional NAME=PACKAGE_ROOT provenance binding",
    )
    parser.add_argument("--splits", nargs="+", choices=("training", "validation"),
                        default=["training", "validation"])
    parser.add_argument("--domains", nargs="+", choices=("nyu", "da2k"),
                        default=["nyu", "da2k"])
    parser.add_argument("--regression-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--max-samples-per-group", type=int,
        help="deterministic screening limit; omit for the complete gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.regression_tolerance < 0:
        parser.error("--regression-tolerance must be non-negative")
    if args.max_samples_per_group is not None and args.max_samples_per_group < 1:
        parser.error("--max-samples-per-group must be positive")
    manifest = load_manifest(args.manifest)
    records = [item for item in samples_for_shape(
        manifest, args.shape, verify_files=False
    ) if item["split"] in args.splits and item["domain"] in args.domains]
    if args.max_samples_per_group is not None:
        counts = {}
        limited = []
        for item in records:
            group = f"{item['split']}:{item['domain']}"
            if counts.get(group, 0) >= args.max_samples_per_group:
                continue
            counts[group] = counts.get(group, 0) + 1
            limited.append(item)
        records = limited
    roots = {"baseline": args.baseline_root}
    for name, root in args.candidate:
        if name == "baseline" or name in roots:
            parser.error(f"duplicate or reserved candidate name: {name}")
        roots[name] = root
    package_roots = dict(args.candidate_package)
    unknown_packages = sorted(set(package_roots) - set(roots))
    if unknown_packages:
        parser.error(f"package has no matching candidate: {unknown_packages}")
    package_records = {
        name: package_provenance(root)
        for name, root in package_roots.items()
    }
    missing = {
        name: [item["sample_id"] for item in records
               if not output_path(root, item["sample_id"]).is_file()]
        for name, root in roots.items()
    }
    missing = {name: items for name, items in missing.items() if items}
    if missing:
        raise ValueError(f"candidate outputs are incomplete: {missing}")
    baseline = group_metrics(records, roots["baseline"], args.reference_root)
    metrics = {name: group_metrics(records, root, args.reference_root)
               for name, root in roots.items()}
    selected, evaluated = rank_candidates(
        baseline, metrics, args.regression_tolerance
    )
    required_groups = {
        "training:nyu", "training:da2k",
        "validation:nyu", "validation:da2k",
    }
    measured_groups = {
        f"{item['split']}:{item['domain']}" for item in records
    }
    complete_four_group_gate = (
        measured_groups == required_groups
        and args.max_samples_per_group is None
    )
    sample_ids_by_group = {}
    for item in records:
        group = f"{item['split']}:{item['domain']}"
        sample_ids_by_group.setdefault(group, []).append(item["sample_id"])
    result = {
        "schema": "depthanything-u250-final-depth-multidomain-selection-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "shape": args.shape,
        "objective": "minimize mean group relL2 ratio under per-group Pareto gate",
        "constraints": {
            "metric": "final_depth_relative_l2",
            "splits": args.splits,
            "domains": args.domains,
            "per_group_regression_tolerance": args.regression_tolerance,
            "max_samples_per_group": args.max_samples_per_group,
            "teacher_forcing": False,
        },
        "gate_scope": (
            "complete_four_group" if complete_four_group_gate else "screening"
        ),
        "sample_count": len(records),
        "sample_ids_by_group": sample_ids_by_group,
        "output_roots": {
            name: str(root.resolve()) for name, root in roots.items()
        },
        "candidate_packages": package_records,
        "baseline": baseline,
        "candidates": evaluated,
        "selected": selected,
        "selected_is_update": selected != "baseline",
        "deployment_status": (
            "qualified_candidate"
            if selected != "baseline" and complete_four_group_gate else
            "candidate_requires_complete_four_group_gate"
            if selected != "baseline" else
            "baseline_retained"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()), "selected": selected,
        "selected_is_update": selected != "baseline",
        "sample_count": len(records),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
