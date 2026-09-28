#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 EXPORT_ROOT OUTPUT_ROOT [JOBS]" >&2
  exit 2
fi
export_root=$(realpath "$1")
output_root=$(realpath -m "$2")
jobs=${3:-2}
compiler=${DS_COMPILER:-/root/demo/DS_Toolchain_Demo_full/python_bin/compile.py}
python_bin=${DS_COMPILER_PYTHON:-/opt/conda/bin/python}
arch_dir=${DS_ARCH_DIR:-/root/demo/DS_Toolchain_Demo_full}
export PYTHON_ROOT=${PYTHON_ROOT:-/root/demo/ACMLIR_DS_remote_20260813/build_cspn_pb320}
export ACOMPILER_EXTENSION_DIR=${ACOMPILER_EXTENSION_DIR:-$PYTHON_ROOT/RelWithDebInfo/lib}
export output_root compiler python_bin arch_dir
layouts=$(python3 - <<'PY'
print(','.join(f'input{i}=BWC' for i in range(24)))
PY
)
export layouts

compile_one() {
  local model=$1 name layer directory work prefix
  name=$(basename "$model" .onnx)
  layer=${name#attention6_l}
  layer=${layer%%_*}
  directory="$output_root/l$layer"
  work="$directory/.work"
  prefix="$directory/$name"
  mkdir -p "$work"
  if (cd "$work" && "$python_bin" "$compiler" \
      --model "$model" --output_path "$prefix" --log_path "$directory" \
      --arch_path "$arch_dir/arch_16_mono.yaml,$arch_dir/arch_256_mono.yaml" \
      --layouts "$layouts" \
      --codegen 3 --sim 1 --addr 1 --l2_size 100 --spill_threshold 0 \
      >"$prefix.compile.stdout.log" 2>&1); then
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

find "$export_root" -mindepth 2 -maxdepth 2 -type f \
  -name 'attention6_l??_a8_to_12xbf16.onnx' -print0 \
  | sort -z | xargs -0 -r -n1 -P "$jobs" bash -c 'compile_one "$1"' _

python3 - "$export_root" "$output_root" <<'PY'
from pathlib import Path
import sys
models = sorted(Path(sys.argv[1]).glob("l??/attention6_l??_a8_to_12xbf16.onnx"))
cfg = sorted(Path(sys.argv[2]).glob("l??/attention6_l??_a8_to_12xbf16_cfg.txt"))
binary = sorted(Path(sys.argv[2]).glob("l??/attention6_l??_a8_to_12xbf16_ddr.bin"))
if (len(models), len(cfg), len(binary)) != (12, 12, 12):
    raise SystemExit(
        f"expected 12 model/cfg/bin files, got {len(models)}/{len(cfg)}/{len(binary)}"
    )
print("compiled=12")
PY
