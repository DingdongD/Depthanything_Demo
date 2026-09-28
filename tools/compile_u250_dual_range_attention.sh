#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 MODEL_DIR OUTPUT_DIR [JOBS]" >&2
  exit 2
fi
model_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
jobs=${3:-2}
compiler=${DS_COMPILER:-/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py}
python_bin=${DS_COMPILER_PYTHON:-/opt/conda/bin/python}
arch_dir=${DS_ARCH_DIR:-/root/demo/DS_Toolchain_Demo_full}
export PYTHON_ROOT=${PYTHON_ROOT:-/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320}
export ACOMPILER_EXTENSION_DIR=${ACOMPILER_EXTENSION_DIR:-$PYTHON_ROOT/RelWithDebInfo/lib}
export output_dir compiler python_bin arch_dir
mkdir -p "$output_dir/.work"

compile_one() {
  local model=$1 name work prefix
  name=$(basename "$model" .onnx)
  work="$output_dir/.work/$name"
  prefix="$output_dir/$name"
  mkdir -p "$work"
  if (cd "$work" && "$python_bin" "$compiler" \
      --model "$model" --output_path "$prefix" --log_path "$output_dir" \
      --arch_path "$arch_dir/arch_16_mono.yaml,$arch_dir/arch_256_mono.yaml" \
      --layouts input0=BWC,input1=BWC,input2=BWC,input3=BWC \
      --codegen 3 --sim 1 --addr 1 --l2_size 100 --spill_threshold 0 \
      >"$prefix.compile.log" 2>&1); then
    if [[ ! -s "${prefix}_cfg.txt" || ! -s "${prefix}_ddr.bin" ]]; then
      echo "FAIL $name: compiler returned without cfg/bin" >&2
      return 1
    fi
    echo "OK $name"
  else
    echo "FAIL $name" >&2
    return 1
  fi
}
export -f compile_one

find "$model_dir" -maxdepth 1 -type f -name 'attention2_l??_h??.onnx' \
  -print0 | sort -z | xargs -0 -r -n1 -P "$jobs" bash -c 'compile_one "$1"' _

python3 - "$model_dir" "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

models = sorted(Path(sys.argv[1]).glob("attention2_l??_h??.onnx"))
output = Path(sys.argv[2])
cfg = sorted(output.glob("attention2_l??_h??_cfg.txt"))
binary = sorted(output.glob("attention2_l??_h??_ddr.bin"))
if len(models) != 6 or len(cfg) != 6 or len(binary) != 6:
    raise SystemExit(
        f"expected 6 model/cfg/bin files, got {len(models)}/{len(cfg)}/{len(binary)}"
    )
print(json.dumps({
    "models": len(models),
    "cfg": len(cfg),
    "bin": len(binary),
    "bin_bytes": {path.stem: path.stat().st_size for path in binary},
}, sort_keys=True))
PY
