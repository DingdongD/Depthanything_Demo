#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_DIR OUTPUT_DIR [JOBS]" >&2
  exit 2
fi

model_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
jobs=${3:-8}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/u250_compile_env.sh"
u250_require_compile_env
export output_dir
mkdir -p "$output_dir"

compile_one() {
  local model=$1 name work_dir
  name=$(basename "$model" .onnx)
  if [[ -s "$output_dir/${name}_ddr.bin" && -s "$output_dir/${name}_cfg.txt" ]]; then
    printf 'SKIP %s\n' "$name"
    return
  fi
  work_dir="$output_dir/.work/$name"
  mkdir -p "$work_dir"
  if (cd "$work_dir" && "$python_bin" "$compiler" \
      --model "$model" \
      --output_path "$output_dir/$name" \
      --log_path "$work_dir" \
      --arch_path "$arch" \
      --layouts input0=BWC,input1=BWC,input2=BWC,input3=BWC \
      --codegen 2 --sim 1 --addr 1 --l2_size 100 --spill_threshold 0 \
      >"$output_dir/${name}.compile.log" 2>&1); then
    printf 'OK %s\n' "$name"
  else
    printf 'FAIL %s\n' "$name" >&2
    return 1
  fi
}
export -f compile_one

find "$model_dir" -maxdepth 1 -type f -name 'attention2_l??_h??.onnx' -print0 \
  | sort -z | xargs -r -0 -n1 -P "$jobs" bash -c 'compile_one "$1"' _
