#!/usr/bin/env bash
set -euo pipefail

root=/home/visitor/Documents/u250_r111_localization
script="$root/tools/run_u250_native_resident_latency_batch.sh"
package=/home/visitor/Documents/depthanything_u250_r113_conv17_s0p731_no_gain
nyu_inputs=/home/visitor/Documents/u250_r110_nyu32/inputs
output=/home/visitor/Documents/u250_r113_conv17_s0p731_nyu32
selected='nyu_00021 nyu_00105 nyu_00190 nyu_00274 nyu_00358 nyu_00442 nyu_00527 nyu_00611'

mapfile -t remaining < <(
  find "$nyu_inputs/nyu32" -maxdepth 1 -type f -name '*.npy' -printf '%f\n' \
    | sed 's/\.npy$//' \
    | grep -vwFf <(tr ' ' '\n' <<<"$selected") \
    | sed 's#^#nyu32/#' \
    | sort
)
if [[ ${#remaining[@]} != 24 ]]; then
  echo "expected 24 NYU holdout samples, found ${#remaining[@]}" >&2
  exit 1
fi
U250_INPUT_ROOT="$nyu_inputs" "$script" "$package" "$output" "${remaining[@]}"

base=/home/visitor/Documents/depthanything_u250_accuracy_r80_da2k_s035
mapfile -t da_samples < <(
  find /home/visitor/Documents/u250_r111_da2k_gate10 \
    -mindepth 2 -maxdepth 2 -type f -name '*.npz' -printf '%P\n' \
    | sed 's/\.npz$//' \
    | sort
)
if [[ ${#da_samples[@]} != 10 ]]; then
  echo "expected 10 DA-2K samples, found ${#da_samples[@]}" >&2
  exit 1
fi
U250_INPUT_ROOT="$base/da2k_r80_baseline/inputs" \
  "$script" "$package" \
  /home/visitor/Documents/u250_r113_conv17_s0p731_da2k_gate10 \
  "${da_samples[@]}"
