#!/usr/bin/env bash
set -euo pipefail

if [[ ${U250_LOCK_HELD:-0} != 1 ]]; then
  exec flock -n /tmp/ds-u250-runtime.lock \
    env U250_LOCK_HELD=1 "$0" "$@"
fi

if [[ $# -lt 6 ]]; then
  echo "usage: $0 BASE_PACKAGE CANDIDATE INPUT_ROOT TRACE_ROOT OUTPUT_ROOT TARGET..." >&2
  echo "target syntax: encoder_block:0, encoder_internal:10:norm1, attention_head:10:0, or decoder_conv:18" >&2
  exit 2
fi
base=$(realpath "$1")
candidate=$(realpath "$2")
input_root=$(realpath "$3")
trace_root=$(realpath "$4")
output_root=$(realpath -m "$5")
shift 5

python_bin=/home/visitor/anaconda3/envs/ds/bin/python
runtime_dir=/home/visitor/Documents/nn_inference
codec_report=$base/artifacts/u250_accuracy_r80_da2k/native_codec_all_oracle.json
if [[ -s $candidate/native_codec_report.json ]]; then
  codec_report=$candidate/native_codec_report.json
fi
layout_codec=${U250_LAYOUT_CODEC:-native}
codec_args=(--layout-codec "$layout_codec")
if [[ $layout_codec == native ]]; then
  codec_args+=(--layout-codec-report "$codec_report")
fi
host_report=${U250_HOST_EXECUTOR_REPORT:-$base/artifacts/u250_accuracy_r80_da2k/host_executor_qualification.json}
fpga_dma_batch=${U250_FPGA_DMA_BATCH:-$base/build/native_codec}
runner=${U250_CAPTURE_RUNNER:-$base/tools/run_u250_depthanything_hybrid_capture_replace.py}
export PYTHONPATH="$(dirname "$runner"):$base/tools:$base"
retries=${U250_CAPTURE_RETRIES:-8}

wait_for_xdma_idle() {
  local stable=0
  while (( stable < 2 )); do
    if fuser /dev/xdma0_events_* >/dev/null 2>&1; then
      stable=0
    else
      stable=$((stable + 1))
    fi
    sleep 1
  done
}

mapfile -t inputs < <(find -L "$input_root" -mindepth 2 -maxdepth 2 -type f \
  -name '*.npy' -printf '%P\n' | sort)
if [[ ${#inputs[@]} == 0 ]]; then
  echo "no inputs under $input_root" >&2
  exit 2
fi

for target in "$@"; do
  IFS=: read -r kind index stage extra <<<"$target"
  if [[ -n ${extra:-} ]]; then
    echo "invalid target: $target" >&2; exit 2
  fi
  case "$kind" in
    encoder_block) option=--replace-encoder-block ;;
    decoder_conv) option=--replace-decoder-conv ;;
    encoder_captures) option=--encoder-captures ;;
    encoder_internal)
      if [[ -z ${stage:-} ]]; then
        echo "invalid target: $target" >&2; exit 2
      fi
      option=--replace-encoder-internal
      ;;
    attention_head)
      if [[ -z ${stage:-} ]]; then
        echo "invalid target: $target" >&2; exit 2
      fi
      option=--replace-encoder-attention-head
      ;;
    *) echo "invalid target: $target" >&2; exit 2 ;;
  esac
  if [[ $kind == encoder_internal || $kind == attention_head ]]; then
    target_dir=$output_root/$(printf '%s_%02d_%s' "$kind" "$index" "$stage")
  else
    target_dir=$output_root/$(printf '%s_%02d' "$kind" "$index")
  fi
  for relative in "${inputs[@]}"; do
    sample=${relative%.npy}
    output=$target_dir/$sample.npz
    if [[ -s $output && -s ${output%.npz}.summary.json ]]; then
      continue
    fi
    mkdir -p "$(dirname "$output")"
    if [[ $kind == encoder_captures ]]; then
      replacement_args=(--encoder-captures "$trace_root/$sample.npz")
    else
      option_value=$index
      if [[ $kind == encoder_internal || $kind == attention_head ]]; then
        option_value=$index:$stage
      fi
      replacement_args=(
        --replacement-trace "$trace_root/$sample.npz" "$option" "$option_value"
      )
    fi
    attempt=1
    while true; do
      wait_for_xdma_idle
      if "$python_bin" "$runner" \
        --case-dir "$candidate" \
        --runtime-dir "$runtime_dir" \
        --manifest "$candidate/resident_kernel_bank_manifest.json" \
        --contract "$candidate/depthanything_u250_runtime_contract.json" \
        --host-plan "$candidate/depthanything_u250_host_plan.json" \
        --host-params "$base/depthanything_u250_host_params.npz" \
        --cfg-dir "$base/cfg" \
        --input "$input_root/$relative" \
        "${replacement_args[@]}" \
        --output "$output" \
        --depth-only \
        --dma-runtime cpp_mapped \
        "${codec_args[@]}" \
        --fpga-dma-batch "$fpga_dma_batch" \
        --host-executor cpp \
        --host-executor-report "$host_report" \
        --attention-launch-group 3 \
        --decoder-launch-group 32 \
        --encoder-fc1-launch-group 6 \
        --attention-resident-kv \
        --decoder-quantize-pack \
        --encoder-fc-frame-graph \
        --cpp-mixed-signature-groups \
        >"${output%.npz}.log" 2>&1; then
        break
      fi
      mv "${output%.npz}.log" "${output%.npz}.attempt${attempt}.log"
      if (( attempt >= retries )); then
        echo "failed after $attempt attempts: $kind:$index $sample" >&2
        exit 1
      fi
      attempt=$((attempt + 1))
    done
    echo "completed $kind:$index $sample"
  done
done
