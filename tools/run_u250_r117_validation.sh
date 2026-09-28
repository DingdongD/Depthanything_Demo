#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT SAMPLE [...]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export U250_ENCODER_FC_FRAME_GRAPH=${U250_ENCODER_FC_FRAME_GRAPH:-1}
exec "$script_dir/run_u250_r116_validation.sh" "$@"
