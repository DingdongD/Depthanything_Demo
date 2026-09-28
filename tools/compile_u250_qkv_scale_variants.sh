#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VARIANT_DIR OUTPUT_DIR" >&2
  exit 2
fi

variant_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/u250_compile_env.sh"
u250_require_compile_env
mkdir -p "$output_dir"

compiled=0
for model in "$variant_dir"/qkv_projection_l??_s*.onnx; do
  [[ -e "$model" ]] || {
    echo "no QKV scale variant ONNX files in $variant_dir" >&2
    exit 1
  }
  name=$(basename "$model" .onnx)
  prefix="$output_dir/$name"
  "$python_bin" "$compiler" \
    --model "$model" --output_path "$prefix" --log_path "$output_dir" \
    --arch_path "$arch" \
    --layouts input0=BWC --codegen 3 --sim 1 --addr 1 --l2_size 100 \
    --spill_threshold 0 >"$prefix.compile.log" 2>&1
  if [[ ! -s "${prefix}_cfg.txt" || ! -s "${prefix}_ddr.bin" ]]; then
    echo "$name: compiler returned without a non-empty cfg/bin" >&2
    exit 1
  fi
  compiled=$((compiled + 1))
  echo "$name compiled"
done
echo "compiled=$compiled"
