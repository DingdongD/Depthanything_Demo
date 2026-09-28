#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VARIANT_DIR OUTPUT_DIR" >&2
  exit 2
fi

variant_dir=$1
output_dir=$2
compiler=${DS_COMPILER:-/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py}
python_bin=${DS_COMPILER_PYTHON:-/opt/conda/bin/python}
arch_dir=${DS_ARCH_DIR:-/root/demo/DS_Toolchain_Demo_full}
export PYTHON_ROOT=${PYTHON_ROOT:-/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320}
export ACOMPILER_EXTENSION_DIR=${ACOMPILER_EXTENSION_DIR:-$PYTHON_ROOT/RelWithDebInfo/lib}
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
    --arch_path "$arch_dir/arch_16_mono.yaml,$arch_dir/arch_256_mono.yaml" \
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
