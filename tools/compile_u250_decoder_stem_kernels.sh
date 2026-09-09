#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_DIR OUTPUT_DIR [JOBS]" >&2
  exit 2
fi
model_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
jobs=${3:-4}
compiler=/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py
arch=/root/demo/DS_Toolchain_Demo_full/arch_16_mono.yaml,/root/demo/DS_Toolchain_Demo_full/arch_256_mono.yaml
export PYTHON_ROOT=/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320
export ACOMPILER_EXTENSION_DIR=/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320/RelWithDebInfo/lib
export output_dir compiler arch
mkdir -p "$output_dir"

compile_one() {
  local model=$1 name out
  name=$(basename "$model" .onnx)
  out="$output_dir/$name"
  mkdir -p "$out"
  if [[ -s "$out/${name}_ddr.bin" && -s "$out/${name}_cfg.txt" ]]; then
    printf 'SKIP %s\n' "$name"
    return
  fi
  if (cd "$out" && /opt/conda/bin/python "$compiler" \
      --model "$model" --output_path "$out/$name" --log_path "$out" \
      --arch_path "$arch" --layouts input0=BWC \
      --codegen 2 --sim 1 --addr 1 --l2_size 100 --spill_threshold 0 \
      >"$out/compile.log" 2>&1); then
    if [[ -s "$out/${name}_ddr.bin" && -s "$out/${name}_cfg.txt" ]]; then
      printf 'OK %s\n' "$name"
    else
      printf 'FAIL %s: compiler returned success without cfg/bin\n' "$name" >&2
      return 1
    fi
  else
    printf 'FAIL %s\n' "$name" >&2
    return 1
  fi
}
export -f compile_one
find -L "$model_dir" -maxdepth 1 -type f \( \
  -name 'decoder_stem_?_co*.onnx' -o \
  -name 'decoder_layout_project_?_co*.onnx' -o \
  -name 'decoder_patch_project_?_co*.onnx' \) \
  -print0 | sort -z | xargs -r -0 -n1 -P "$jobs" bash -c 'compile_one "$1"' _
