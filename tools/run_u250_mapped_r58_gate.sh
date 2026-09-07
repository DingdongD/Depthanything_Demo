#!/usr/bin/env bash
set -euo pipefail

# CPU-only entry points exit before the board runtime lock or package writes.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--check-native-controlflow" ]]; then
  [[ $# == 2 ]] || { echo "usage: $0 --check-native-controlflow SUMMARY" >&2; exit 2; }
  exec "${PYTHON:-python3}" "$script_dir/run_u250_native_codec_controlflow.py" \
    --check-summary "$2"
fi
if [[ "${1:-}" == "--native-controlflow" ]]; then
  shift
  exec "${PYTHON:-/home/visitor/anaconda3/envs/ds/bin/python}" \
    "$script_dir/run_u250_native_codec_controlflow.py" "$@"
fi
if [[ "${1:-}" == "--check-native-board" ]]; then
  [[ $# == 2 ]] || { echo "usage: $0 --check-native-board RUN_DIR" >&2; exit 2; }
  exec "${PYTHON:-python3}" "$script_dir/check_u250_native_board_gate.py" --run-dir "$2"
fi
[[ $# == 0 ]] || { echo "unknown board gate argument" >&2; exit 2; }

pkg="${U250_NATIVE_PACKAGE:-/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_nativecodec_r60}"
python_bin="${PYTHON:-/home/visitor/anaconda3/envs/ds/bin/python}"
run_dir="$pkg/native_r60_gate"
checker="$script_dir/check_u250_native_board_gate.py"

exec 9>/tmp/ds-u250-runtime.lock
if ! flock -n 9; then
  echo "U250 lock is busy; no device access attempted" >&2
  exit 75
fi
# The caller pins the independently prepared deployment inventory. Verification
# happens before importing the extension or opening any hardware descriptor.
: "${U250_DEPLOYMENT_SHA256:?set the verified deployment inventory SHA-256}"
verified=$("$python_bin" "$checker" --verify-package "$pkg" \
  --deployment-sha256 "$U250_DEPLOYMENT_SHA256")
# A fresh directory prevents stale summaries from satisfying a failed new run.
mkdir "$run_dir"
printf '%s\n' "$verified" >"$run_dir/deployment_verified.json"
export PYTHONDONTWRITEBYTECODE=1

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
  --fpga-dma-batch "$pkg/build/native_codec"
  --layout-codec native
  --layout-codec-report "$pkg/artifacts/u250_native_codec/all_oracle.json"
  --attention-launch-group 3
  --decoder-launch-group 32
  --depth-only
)

"$python_bin" "$pkg/tools/run_u250_depthanything_hybrid.py" \
  "${common[@]}" \
  --encoder-captures "$pkg/demo05_holdout_trace_r52.npz" \
  --output "$run_dir/demo05_decoder_only.npz" \
  >"$run_dir/demo05_decoder_only.log" 2>&1
"$python_bin" "$checker" --frame "$run_dir/demo05_decoder_only.summary.json"

"$python_bin" "$pkg/tools/run_u250_depthanything_hybrid.py" \
  "${common[@]}" \
  --encoder-resume "$pkg/demo05_holdout_trace_r52.npz" \
  --encoder-start-layer 11 \
  --output "$run_dir/demo05_resume_l11.npz" \
  >"$run_dir/demo05_resume_l11.log" 2>&1
"$python_bin" "$checker" --frame "$run_dir/demo05_resume_l11.summary.json"

requests="$run_dir/resident_requests.jsonl"
"$python_bin" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
  "${common[@]}" >"$run_dir/resident_base_args_r60.json"
printf '%s\n' \
  "{\"input\":\"$pkg/demo05.npy\",\"output\":\"$run_dir/demo05_full_first.npz\",\"golden\":\"$pkg/demo05_board_r43_depth.npy\"}" \
  "{\"input\":\"$pkg/demo05.npy\",\"output\":\"$run_dir/demo05_full_resident.npz\",\"golden\":\"$pkg/demo05_board_r43_depth.npy\"}" \
  '{"command":"shutdown"}' >"$requests"
"$python_bin" "$pkg/tools/depthanything_u250_resident_server.py" \
  --runner "$pkg/tools/run_u250_depthanything_hybrid.py" \
  --base-args "$run_dir/resident_base_args_r60.json" \
  <"$requests" >"$run_dir/resident_server.jsonl" 2>"$run_dir/resident_server.stderr"

"$python_bin" "$checker" --run-dir "$run_dir"
printf '%s\n' "$run_dir/gate_summary.json"
