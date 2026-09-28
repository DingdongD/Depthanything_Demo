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
export model_dir output_dir
mkdir -p "$output_dir"

compile_one() {
  local model=$1
  local name out layout
  name=$(basename "$model" .onnx)
  out="$output_dir/$name"
  mkdir -p "$out"
  if [[ -s "$out/${name}_ddr.bin" && -s "$out/${name}_cfg.txt" ]]; then
    printf 'SKIP %s\n' "$name"
    return
  fi
  case "$name" in
    post_attention_*) layout=attention_input=BWC,residual_input=BWC ;;
    mlp_fc1_*_c*) layout=norm2_input=BWC ;;
    mlp_fc2_*) layout=gelu_input=BWC ;;
    *) printf 'REJECT %s\n' "$name" >&2; return 1 ;;
  esac
  if (cd "$out" && "$python_bin" "$compiler" \
      --model "$model" \
      --output_path "$out/$name" \
      --log_path "$out" \
      --arch_path "$arch" \
      --layouts "$layout" \
      --codegen 2 --sim 1 --addr 1 --l2_size 100 --spill_threshold 0 \
      >"$out/compile.log" 2>&1); then
    printf 'OK %s\n' "$name"
  else
    printf 'FAIL %s\n' "$name" >&2
    return 1
  fi
}
export -f compile_one

find "$model_dir" -maxdepth 1 -type f \
  \( -name 'post_attention_l??.onnx' \
     -o -name 'mlp_fc1_l??_c??.onnx' \
     -o -name 'mlp_fc2_l??.onnx' \) \
  -print0 | sort -z | xargs -r -0 -n1 -P "$jobs" bash -c 'compile_one "$1"' _
