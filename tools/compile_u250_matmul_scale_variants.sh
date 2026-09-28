#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VARIANT_DIR OUTPUT_DIR" >&2
  exit 2
fi

variant_dir=$1
output_dir=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/u250_compile_env.sh"
u250_require_compile_env
mkdir -p "$output_dir"

compiled=0
for model in "$variant_dir"/s*/*.onnx; do
  [[ -e "$model" ]] || { echo "no scale variant ONNX files in $variant_dir" >&2; exit 1; }
  tag=$(basename "$(dirname "$model")")
  name=$(basename "$model" .onnx)
  case "$name" in
    post_attention_l??)
      layouts=attention_input=BWC,residual_input=BWC
      ;;
    mlp_fc2_l??)
      layouts=gelu_input=BWC
      ;;
    *)
      echo "unsupported MatMul scale kernel: $name" >&2
      exit 1
      ;;
  esac
  prefix="$output_dir/$tag/$name"
  if [[ -s "${prefix}_cfg.txt" && -s "${prefix}_ddr.bin" ]]; then
    continue
  fi
  mkdir -p "$prefix"
  "$python_bin" "$compiler" \
    --model "$model" --output_path "$prefix" --log_path "$prefix" \
    --arch_path "$arch" \
    --layouts "$layouts" --codegen 2 --sim 1 --addr 1 --l2_size 100 \
    --spill_threshold 0 >"$prefix/compile.log" 2>&1
  if [[ ! -s "${prefix}_cfg.txt" || ! -s "${prefix}_ddr.bin" ]]; then
    echo "$tag/$name: compiler returned without a non-empty cfg/bin" >&2
    exit 1
  fi
  compiled=$((compiled + 1))
  echo "$tag/$name compiled"
done
echo "compiled=$compiled"
