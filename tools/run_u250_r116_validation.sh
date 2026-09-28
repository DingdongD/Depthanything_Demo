#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT SAMPLE [...]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export U250_ENCODER_FC1_LAUNCH_GROUP=${U250_ENCODER_FC1_LAUNCH_GROUP:-4}
export U250_FRONTEND_LAUNCH_GROUP=${U250_FRONTEND_LAUNCH_GROUP:-6}
export U250_CPP_MIXED_SIGNATURE_GROUPS=${U250_CPP_MIXED_SIGNATURE_GROUPS:-0}
exec "$script_dir/run_u250_r115_validation.sh" "$@"
