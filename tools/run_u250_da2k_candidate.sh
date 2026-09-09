#!/usr/bin/env bash
set -euo pipefail

if [[ ${U250_LOCK_HELD:-0} != 1 ]]; then
  exec flock -n /tmp/ds-u250-runtime.lock \
    env U250_LOCK_HELD=1 "$0" "$@"
fi

if [[ $# -lt 5 ]]; then
  echo "usage: $0 BASE_PACKAGE CANDIDATE_DIR INPUT_DIR OUTPUT_DIR SAMPLE..." >&2
  exit 2
fi
base=$(realpath "$1")
candidate=$(realpath "$2")
input_dir=$(realpath "$3")
output_dir=$(realpath -m "$4")
shift 4

python_bin=/home/visitor/anaconda3/envs/ds/bin/python
runtime_dir=/home/visitor/Documents/nn_inference
default_codec_report=$base/artifacts/u250_accuracy_r80_da2k/native_codec_all_oracle.json
if [[ ! -f $default_codec_report ]]; then
  default_codec_report=$base/artifacts/u250_accuracy_r79/native_codec_all_oracle.json
fi
codec_report=${U250_CODEC_REPORT:-$default_codec_report}
default_host_report=$base/artifacts/u250_accuracy_r80_da2k/host_executor_qualification.json
if [[ ! -f $default_host_report ]]; then
  default_host_report=$base/artifacts/u250_accuracy_r79/host_executor_qualification.json
fi
host_report=${U250_HOST_REPORT:-$default_host_report}
export PYTHONPATH="$base"
mkdir -p "$output_dir"
cd "$base"

if [[ $# == 1 && $1 == "--all" ]]; then
  mapfile -t samples < <(find "$input_dir" -maxdepth 1 -type f -name '*.npy' \
    -printf '%f\n' | sed 's/\.npy$//' | sort)
else
  samples=("$@")
fi

for sample in "${samples[@]}"; do
  "$python_bin" "$base/tools/run_u250_depthanything_hybrid.py" \
    --case-dir "$candidate" \
    --runtime-dir "$runtime_dir" \
    --manifest "$candidate/resident_kernel_bank_manifest.json" \
    --contract "$candidate/depthanything_u250_runtime_contract.json" \
    --host-plan "$candidate/depthanything_u250_host_plan.json" \
    --host-params "$base/depthanything_u250_host_params.npz" \
    --cfg-dir "$base/cfg" \
    --input "$input_dir/$sample.npy" \
    --output "$output_dir/$sample.npz" \
    --depth-only \
    --dma-runtime cpp_mapped \
    --layout-codec native \
    --layout-codec-report "$codec_report" \
    --fpga-dma-batch "$base/build/native_codec" \
    --host-executor cpp \
    --host-executor-report "$host_report" \
    --attention-launch-group 3 \
    --decoder-launch-group 32 \
    >"$output_dir/$sample.log" 2>&1
done
