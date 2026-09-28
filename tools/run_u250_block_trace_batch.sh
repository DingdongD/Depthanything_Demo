#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 PACKAGE OUTPUT_ROOT LAYER REPLACE_BLOCK|none SAMPLE [...]" >&2
  exit 2
fi

package=$1
output_root=$2
layer=$3
replace_block=$4
shift 4

base=${U250_DA_BASE:-/home/visitor/Documents/depthanything_u250_accuracy_r80_da2k_s035}
runtime_dir=${U250_RUNTIME_DIR:-/home/visitor/Documents/nn_inference}
python_bin=${U250_PYTHON:-/home/visitor/anaconda3/envs/ds/bin/python}
fp32_root=${U250_FP32_ROOT:-/home/visitor/Documents/u250_all_blocks_fp32_10}
lock_file=${U250_LOCK_FILE:-/tmp/ds-u250-runtime.lock}
lock_timeout=${U250_LOCK_TIMEOUT:-60}
runner=$base/tools/run_u250_depthanything_hybrid.py

for path in \
  "$package/resident_kernel_bank_manifest.json" \
  "$package/depthanything_u250_runtime_contract.json" \
  "$package/depthanything_u250_host_plan.json" \
  "$base/depthanything_u250_host_params.npz" \
  "$runner"; do
  [[ -r "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

replace_args=()
if [[ "$replace_block" != none ]]; then
  replace_args=(--replace-encoder-block "$replace_block")
fi

for sample in "$@"; do
  split=${sample%%/*}
  name=${sample##*/}
  input=$base/da2k_r80_baseline/inputs/$split/$name.npy
  replacement=$fp32_root/$split/$name.npz
  output=$output_root/$split/$name.npz
  log=$output_root/$split/$name.log
  [[ -r "$input" ]] || { echo "missing input: $input" >&2; exit 1; }
  mkdir -p "$(dirname "$output")"
  echo "RUN_$name"
  command=(
    "$python_bin" "$runner"
    --case-dir "$package"
    --runtime-dir "$runtime_dir"
    --manifest "$package/resident_kernel_bank_manifest.json"
    --contract "$package/depthanything_u250_runtime_contract.json"
    --host-plan "$package/depthanything_u250_host_plan.json"
    --host-params "$base/depthanything_u250_host_params.npz"
    --cfg-dir "$base/cfg"
    --input "$input"
    --output "$output"
    --depth-only
    --trace-attention-layers "$layer"
    --trace-encoder-internal-layers "$layer"
    --timeout-ms 5000
    --dma-runtime cpp_mapped
    --layout-codec vendor
    --fpga-dma-batch "$base/build/native_codec"
    --host-executor cpp
    --host-executor-report "$base/artifacts/u250_accuracy_r80_da2k/host_executor_qualification.json"
    --attention-launch-group 3
    --decoder-launch-group 32
  )
  if [[ "$replace_block" != none ]]; then
    [[ -r "$replacement" ]] || { echo "missing replacement: $replacement" >&2; exit 1; }
    command+=(--replacement-trace "$replacement" "${replace_args[@]}")
  fi
  env PYTHONPATH="$base/tools:$base" \
    flock -w "$lock_timeout" "$lock_file" "${command[@]}" >"$log" 2>&1
  grep -q '"finite": true' "$log"
  grep -q '"static_reloads": 0' "$log"
  echo "PASS_$name"
done
