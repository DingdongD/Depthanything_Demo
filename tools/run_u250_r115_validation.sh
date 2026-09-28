#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT SAMPLE [...]" >&2
  exit 2
fi

base=${U250_DA_BASE:-/home/visitor/Documents/depthanything_u250_accuracy_r80_da2k_s035}
package=${U250_R115_PACKAGE:-/home/visitor/Documents/depthanything_u250_r113_conv17_s0p731_no_gain}
qualification=${U250_R115_QUALIFICATION:-/home/visitor/Documents/u250_r115_quantize_pack_qualification}
batch=$base/tools/run_u250_native_resident_latency_batch.sh

export U250_FPGA_DMA_BATCH=${U250_FPGA_DMA_BATCH:-$base/build/native_codec_r115}
export U250_CODEC_REPORT=${U250_CODEC_REPORT:-$qualification/native_codec_report.json}
export U250_HOST_EXECUTOR_REPORT=${U250_HOST_EXECUTOR_REPORT:-$qualification/host_executor_qualification.json}
export U250_CPP_PERSISTENT_DMA=${U250_CPP_PERSISTENT_DMA:-1}
export U250_DECODER_QUANTIZE_PACK=${U250_DECODER_QUANTIZE_PACK:-1}
export U250_CPP_MIXED_SIGNATURE_GROUPS=${U250_CPP_MIXED_SIGNATURE_GROUPS:-0}
export U250_ENCODER_FC1_LAUNCH_GROUP=${U250_ENCODER_FC1_LAUNCH_GROUP:-1}
export U250_FRONTEND_LAUNCH_GROUP=${U250_FRONTEND_LAUNCH_GROUP:-1}

output=$1
shift
exec "$batch" "$package" "$output" "$@"
