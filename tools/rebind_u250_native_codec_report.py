#!/usr/bin/env python3
"""Rebind an exact native-codec qualification to an ABI-identical bank manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


CASE_ABI_FIELDS = (
    "name", "inputs", "outputs", "isa_ranges", "base_addresses_local",
    "offset_bytes",
)
MANIFEST_ABI_FIELDS = (
    "alignment_bytes", "bank_size_bytes", "format", "required_fm_io_bytes",
    "shared_fm_base_units", "shared_fm_placement", "shared_fm_workspace_bytes",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def abi_projection(manifest: dict) -> dict:
    return {
        "manifest": {key: manifest.get(key) for key in MANIFEST_ABI_FIELDS},
        "cases": [
            {key: case.get(key) for key in CASE_ABI_FIELDS}
            for case in sorted(manifest["cases"], key=lambda item: item["name"])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_bytes = args.source_manifest.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    manifest = json.loads(manifest_bytes)
    report = json.loads(args.source_report.read_bytes())
    source_digest = sha256_bytes(source_manifest_bytes)
    if report.get("qualified") is not True:
        raise ValueError("source native-codec report is not qualified")
    if report.get("manifest_sha256") != source_digest:
        raise ValueError("source report does not match source manifest")
    if abi_projection(source_manifest) != abi_projection(manifest):
        raise ValueError("new manifest changes a codec-visible tensor ABI")

    report["rebound_from_manifest_sha256"] = source_digest
    report["manifest_sha256"] = sha256_bytes(manifest_bytes)
    report["rebind_reason"] = args.reason
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest_sha256": report["manifest_sha256"],
        "output": str(args.output),
        "rebound_from_manifest_sha256": source_digest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
