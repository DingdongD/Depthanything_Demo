#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ONNX_DIR OUTPUT_DIR" >&2
  exit 2
fi

onnx_dir=$1
output_dir=$2
compiler=${DS_COMPILER:-/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py}
python_bin=${DS_COMPILER_PYTHON:-/opt/conda/bin/python}
arch_dir=${DS_ARCH_DIR:-/root/demo/DS_Toolchain_Demo_full}
export PYTHON_ROOT=${PYTHON_ROOT:-/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320}
export ACOMPILER_EXTENSION_DIR=${ACOMPILER_EXTENSION_DIR:-$PYTHON_ROOT/RelWithDebInfo/lib}
mkdir -p "$output_dir"

compiled=0
reused=0
for model in "$onnx_dir"/mlp_fc1_pair_l??_p??.onnx; do
  [[ -e "$model" ]] || { echo "no paired FC1 ONNX files in $onnx_dir" >&2; exit 1; }
  name=$(basename "$model" .onnx)
  prefix="$output_dir/$name"
  if [[ -s "${prefix}_cfg.txt" && -s "${prefix}_ddr.bin" ]]; then
    reused=$((reused + 1))
    continue
  fi
  mkdir -p "$prefix"
  "$python_bin" "$compiler" \
    --model "$model" --output_path "$prefix" --log_path "$prefix" \
    --arch_path "$arch_dir/arch_16_mono.yaml,$arch_dir/arch_256_mono.yaml" \
    --layouts input0=BWC --codegen 2 --sim 1 --addr 1 --l2_size 100 \
    --spill_threshold 0 >"$prefix/compile.stdout.log" 2>&1
  if [[ ! -s "${prefix}_cfg.txt" || ! -s "${prefix}_ddr.bin" ]]; then
    echo "$name: compiler returned without a non-empty cfg/bin" >&2
    exit 1
  fi
  compiled=$((compiled + 1))
  echo "$name compiled"
done

python - "$output_dir" "$compiled" "$reused" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
cfg = sorted(directory.glob("mlp_fc1_pair_l??_p??_cfg.txt"))
binary = sorted(directory.glob("mlp_fc1_pair_l??_p??_ddr.bin"))
if len(cfg) != 36 or len(binary) != 36:
    raise SystemExit(f"expected 36 cfg/bin pairs, got {len(cfg)}/{len(binary)}")
print(json.dumps({
    "compiled": int(sys.argv[2]), "reused": int(sys.argv[3]),
    "qualified_cfg": len(cfg), "qualified_bin": len(binary),
}, sort_keys=True))
PY
