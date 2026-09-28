"""Validation helpers for the immutable full-graph calibration manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "depthanything-u250-full-graph-calibration-set-v1"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported calibration manifest schema: {path}")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual = canonical_sha256(unsigned)
    if claimed != actual:
        raise ValueError(
            f"calibration manifest fingerprint mismatch: {claimed} != {actual}"
        )
    samples = manifest.get("samples", [])
    sample_ids = [item.get("sample_id") for item in samples]
    if not samples or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("calibration manifest has no samples or duplicate sample IDs")
    splits = manifest.get("sample_ids", {})
    expected = {
        split: sorted(item["sample_id"] for item in samples
                      if item.get("split") == split)
        for split in ("training", "validation")
    }
    if any(sorted(splits.get(split, [])) != expected[split] for split in expected):
        raise ValueError("calibration manifest split lists do not match samples")
    if set(expected["training"]) & set(expected["validation"]):
        raise ValueError("training and validation calibration samples overlap")
    policy = manifest.get("calibration_policy", {})
    if (policy.get("mode") != "single-pass-full-graph-joint"
            or policy.get("teacher_forcing") is not False
            or policy.get("sequential_layer_freeze") is not False):
        raise ValueError("manifest does not enforce full-graph joint calibration")
    for stage in policy.get("stages", []):
        stage_ids = stage.get("sample_ids", {})
        if any(sorted(stage_ids.get(split, [])) != expected[split]
               for split in expected):
            raise ValueError(
                f"stage {stage.get('name')} does not use the unified sample set"
            )
    if len(policy.get("stages", [])) != 6:
        raise ValueError("full-graph manifest must define exactly stages 1-6")
    return manifest


def samples_for_shape(manifest: dict, shape: int,
                      verify_files: bool = True) -> list[dict]:
    if shape not in manifest.get("shapes", []):
        raise ValueError(f"shape {shape} is absent from calibration manifest")
    result = []
    for sample in manifest["samples"]:
        tensor = sample.get("tensors", {}).get(str(shape))
        if tensor is None:
            raise ValueError(f"{sample['sample_id']}: missing shape {shape} tensor")
        if tensor.get("shape") != [1, 3, shape, shape]:
            raise ValueError(f"{sample['sample_id']}: invalid shape metadata")
        path = Path(tensor["path"])
        if verify_files:
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = file_sha256(path)
            if actual != tensor.get("sha256"):
                raise ValueError(
                    f"{sample['sample_id']}: tensor checksum mismatch "
                    f"{actual} != {tensor.get('sha256')}"
                )
        result.append({**sample, "tensor_path": path})
    return result
