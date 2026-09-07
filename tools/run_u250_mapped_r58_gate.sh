#!/usr/bin/env bash
set -euo pipefail

pkg=/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_lnfold_decoderretuned
python_bin=/home/visitor/anaconda3/envs/ds/bin/python
run_dir="$pkg/mapped_r58_gate"
expected=2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725

exec 9>/tmp/ds-u250-runtime.lock
if ! flock -n 9; then
  echo "U250 lock is busy; no device access attempted" >&2
  exit 75
fi
mkdir -p "$run_dir"

common=(
  --case-dir "$pkg"
  --runtime-dir /home/visitor/Documents/nn_inference
  --manifest "$pkg/resident_kernel_bank_manifest.json"
  --contract "$pkg/depthanything_u250_runtime_contract.json"
  --host-plan "$pkg/depthanything_u250_host_plan.json"
  --host-params "$pkg/depthanything_u250_host_params.npz"
  --cfg-dir "$pkg/cfg"
  --input "$pkg/demo05.npy"
  --golden "$pkg/demo05_board_r43_depth.npy"
  --dma-runtime cpp_mapped
  --fpga-dma-batch "$pkg"
  --attention-launch-group 3
  --decoder-launch-group 32
  --depth-only
)

"$python_bin" "$pkg/run_u250_depthanything_hybrid_mapped_r58.py" \
  "${common[@]}" \
  --encoder-captures "$pkg/demo05_holdout_trace_r52.npz" \
  --output "$run_dir/demo05_decoder_only.npz" \
  >"$run_dir/demo05_decoder_only.log" 2>&1

"$python_bin" "$pkg/run_u250_depthanything_hybrid_mapped_r58.py" \
  "${common[@]}" \
  --encoder-resume "$pkg/demo05_holdout_trace_r52.npz" \
  --encoder-start-layer 11 \
  --output "$run_dir/demo05_resume_l11.npz" \
  >"$run_dir/demo05_resume_l11.log" 2>&1

requests="$run_dir/resident_requests.jsonl"
printf '%s\n' \
  "{\"input\":\"$pkg/demo05.npy\",\"output\":\"$run_dir/demo05_full_first.npz\",\"golden\":\"$pkg/demo05_board_r43_depth.npy\"}" \
  "{\"input\":\"$pkg/demo05.npy\",\"output\":\"$run_dir/demo05_full_resident.npz\",\"golden\":\"$pkg/demo05_board_r43_depth.npy\"}" \
  '{"command":"shutdown"}' >"$requests"
"$python_bin" "$pkg/depthanything_u250_resident_server.py" \
  --runner "$pkg/run_u250_depthanything_hybrid_mapped_r58.py" \
  --base-args "$pkg/resident_base_args_r58.json" \
  <"$requests" >"$run_dir/resident_server.jsonl" 2>"$run_dir/resident_server.stderr"

EXPECTED="$expected" RUN_DIR="$run_dir" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_DIR"])
expected = os.environ["EXPECTED"]
names = (
    "demo05_decoder_only", "demo05_resume_l11",
    "demo05_full_first", "demo05_full_resident",
)
reports = {name: json.loads((root / f"{name}.summary.json").read_text())
           for name in names}
for name, report in reports.items():
    if report["output_sha256"] != expected:
        raise SystemExit(
            f"{name}: {report['output_sha256']} does not match r43 {expected}"
        )
if not reports["demo05_full_resident"]["resident_bank_reused"]:
    raise SystemExit("second resident request reloaded the bank")
(root / "gate_summary.json").write_text(json.dumps({
    "passed": True,
    "expected_sha256": expected,
    "reports": {name: {
        "wall_ms": report["wall_ms"],
        "npu_ms": report["npu_ms_total"],
        "h2c_ms": report["h2c_ms_total"],
        "c2h_ms": report["c2h_ms_total"],
        "submission_groups": report["submission_groups"],
        "resident_bank_reused": report["resident_bank_reused"],
        "metrics": report.get("metrics"),
    } for name, report in reports.items()},
}, indent=2, sort_keys=True) + "\n")
print(root / "gate_summary.json")
PY
