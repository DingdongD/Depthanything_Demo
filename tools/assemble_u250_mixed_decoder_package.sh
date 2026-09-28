#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 ]]; then
  echo "usage: $0 BASE_PACKAGE VARIANT_MANIFEST COMPILED_ROOT CALIBRATION OUTPUT_PACKAGE" >&2
  exit 2
fi
base=$(realpath "$1")
variants=$(realpath "$2")
compiled=$(realpath "$3")
calibration=$(realpath "$4")
output=$(realpath -m "$5")
tools_dir=$(cd "$(dirname "$0")" && pwd)

if [[ -e "$output" ]]; then
  echo "output already exists: $output" >&2
  exit 1
fi
cp -a "$base" "$output"

while IFS=$'\t' read -r kernel scale; do
  cfg="$compiled/$kernel/${kernel}_cfg.txt"
  binary="$compiled/$kernel/${kernel}_ddr.bin"
  [[ -s $cfg && -s $binary ]] || {
    echo "missing compiled CFG/BIN for $kernel" >&2
    exit 1
  }
  python3 "$tools_dir/replace_u250_kernel_in_bank.py" \
    --manifest "$output/resident_kernel_bank_manifest.json" \
    --contract "$output/depthanything_u250_runtime_contract.json" \
    --host-plan "$output/depthanything_u250_host_plan.json" \
    --kernel "$kernel" --cfg "$cfg" --binary "$binary" \
    --input-scale "$scale" --output-dir "$output"
  cp "$cfg" "$output/cfg/${kernel}_cfg.txt"
done < <(python3 - "$variants" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1]))["variants"]:
    print(f"{item['kernel']}\t{item['input_scale']:.17g}")
PY
)

cp "$variants" "$output/mixed_decoder_variants.json"
cp "$calibration" "$output/mixed_decoder_calibration.json"
python3 "$tools_dir/rebind_u250_native_codec_report.py" \
  --source-report "$base/native_codec_report.json" \
  --source-manifest "$base/resident_kernel_bank_manifest.json" \
  --manifest "$output/resident_kernel_bank_manifest.json" \
  --reason "ABI-identical mixed NYU-train plus DA-2K decoder A8 scale replacement" \
  --output "$output/native_codec_report.json"

python3 - "$output" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
with (root / "resident_kernel_bank_manifest.json").open() as stream:
    manifest = json.load(stream)
bank = root / manifest["bank_file"]
digest = hashlib.sha256(bank.read_bytes()).hexdigest()
if digest != manifest["bank_sha256"]:
    raise SystemExit("bank SHA mismatch")
print(json.dumps({
    "output": str(root), "bank_bytes": bank.stat().st_size,
    "bank_sha256": digest,
    "kernel_replacements": len(manifest.get("kernel_replacements", [])),
}, sort_keys=True))
PY
