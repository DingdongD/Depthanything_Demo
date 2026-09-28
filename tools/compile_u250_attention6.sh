#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 EXPORT_ROOT OUTPUT_ROOT [JOBS]" >&2
  exit 2
fi
export_root=$(realpath "$1")
output_root=$(realpath -m "$2")
jobs=${3:-2}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/u250_compile_env.sh"
u250_require_compile_env
export output_root
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
      --arch_path "$arch" \
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
