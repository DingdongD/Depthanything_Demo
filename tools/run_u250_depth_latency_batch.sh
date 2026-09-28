#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 PACKAGE OUTPUT_ROOT SAMPLE [...]" >&2
  exit 2
fi

package=$1
output_root=$2
shift 2
base=${U250_DA_BASE:-/home/visitor/Documents/depthanything_u250_accuracy_r80_da2k_s035}
runtime_dir=${U250_RUNTIME_DIR:-/home/visitor/Documents/nn_inference}
python_bin=${U250_PYTHON:-/home/visitor/anaconda3/envs/ds/bin/python}
lock_file=${U250_LOCK_FILE:-/tmp/ds-u250-runtime.lock}
lock_timeout=${U250_LOCK_TIMEOUT:-60}
runner=$base/tools/run_u250_depthanything_hybrid.py

for sample in "$@"; do
  split=${sample%%/*}
  name=${sample##*/}
  input=$base/da2k_r80_baseline/inputs/$split/$name.npy
  output=$output_root/$split/$name.npz
  log=$output_root/$split/$name.log
  mkdir -p "$(dirname "$output")"
  echo "RUN_$name"
  env PYTHONPATH="$base/tools:$base" flock -w "$lock_timeout" "$lock_file" \
    "$python_bin" "$runner" \
      --case-dir "$package" \
      --runtime-dir "$runtime_dir" \
      --manifest "$package/resident_kernel_bank_manifest.json" \
      --contract "$package/depthanything_u250_runtime_contract.json" \
      --host-plan "$package/depthanything_u250_host_plan.json" \
      --host-params "$base/depthanything_u250_host_params.npz" \
      --cfg-dir "$base/cfg" \
      --input "$input" \
      --output "$output" \
      --depth-only \
      --timeout-ms 5000 \
      --dma-runtime cpp_mapped \
      --layout-codec vendor \
      --fpga-dma-batch "$base/build/native_codec" \
      --host-executor cpp \
      --host-executor-report "$base/artifacts/u250_accuracy_r80_da2k/host_executor_qualification.json" \
      --attention-launch-group 3 \
      --decoder-launch-group 32 >"$log" 2>&1
  grep -q '"finite": true' "$log"
  grep -q '"static_reloads": 0' "$log"
  "$python_bin" - "$output" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as values:
    if values.files != ["depth"] or not np.isfinite(values["depth"]).all():
        raise SystemExit(f"invalid depth-only output keys: {values.files}")
PY
  echo "PASS_$name"
done
