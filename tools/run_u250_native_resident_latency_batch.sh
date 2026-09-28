#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 PACKAGE OUTPUT_ROOT SAMPLE [...]" >&2
  exit 2
fi

package=$1
output_root=$2
shift 2
base=${U250_DA_BASE:-/home/visitor/Documents/depthanything_u250_accuracy_r80_da2k_s035}
runtime_dir=${U250_RUNTIME_DIR:-/home/visitor/Documents/nn_inference}
python_bin=${U250_PYTHON:-/home/visitor/anaconda3/envs/ds/bin/python}
lock_file=${U250_LOCK_FILE:-/tmp/ds-u250-runtime.lock}
lock_timeout=${U250_LOCK_TIMEOUT:-60}
runner=${U250_RUNNER:-$base/tools/run_u250_depthanything_hybrid.py}
resident_server=$base/tools/depthanything_u250_resident_server.py
contract=${U250_CONTRACT:-$package/depthanything_u250_runtime_contract.json}
codec_report=${U250_CODEC_REPORT:-$package/native_codec_report.json}
fpga_dma_batch=${U250_FPGA_DMA_BATCH:-$base/build/native_codec}
host_executor_report=${U250_HOST_EXECUTOR_REPORT:-$base/artifacts/u250_accuracy_r80_da2k/host_executor_qualification.json}
host_params=${U250_HOST_PARAMS:-$base/depthanything_u250_host_params.npz}
input_root=${U250_INPUT_ROOT:-$base/da2k_r80_baseline/inputs}
attention_launch_group=${U250_ATTENTION_LAUNCH_GROUP:-3}
decoder_launch_group=${U250_DECODER_LAUNCH_GROUP:-32}
encoder_fc1_launch_group=${U250_ENCODER_FC1_LAUNCH_GROUP:-1}
frontend_launch_group=${U250_FRONTEND_LAUNCH_GROUP:-1}
# This batch path owns the board lock for its complete lifetime, so descriptor
# reuse is safe and avoids reopening four XDMA character devices per transfer.
# Set U250_CPP_PERSISTENT_DMA=0 only for driver troubleshooting.
persistent_dma=${U250_CPP_PERSISTENT_DMA:-1}
decoder_quantize_pack=${U250_DECODER_QUANTIZE_PACK:-0}
encoder_quantize_pack=${U250_ENCODER_QUANTIZE_PACK:-0}
mixed_signature_groups=${U250_CPP_MIXED_SIGNATURE_GROUPS:-0}
encoder_fc_frame_graph=${U250_ENCODER_FC_FRAME_GRAPH:-0}
attention_resident_kv=${U250_ATTENTION_RESIDENT_KV:-1}
encoder_resident_intermediates=${U250_ENCODER_RESIDENT_INTERMEDIATES:-1}
attention_post_frame_graph=${U250_ATTENTION_POST_FRAME_GRAPH:-0}
fused_qkv_attention=${U250_FUSED_QKV_ATTENTION:-0}
fused_qkv_attention_layers=${U250_FUSED_QKV_ATTENTION_LAYERS:-}
fused_attention6=${U250_FUSED_ATTENTION6:-0}
fused_attention6_layers=${U250_FUSED_ATTENTION6_LAYERS:-}
qkv_attention6_frame_graph=${U250_QKV_ATTENTION6_FRAME_GRAPH:-0}

for path in \
  "$package/resident_kernel_bank_manifest.json" \
  "$contract" \
  "$package/depthanything_u250_host_plan.json" \
  "$host_params" \
  "$codec_report" \
  "$fpga_dma_batch" \
  "$host_executor_report" \
  "$runner" \
  "$resident_server"; do
  [[ -r "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done

mkdir -p "$output_root"
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
base_args=$temporary/base_args.json
requests=$temporary/requests.jsonl
server_log=$output_root/resident_server.log
batch_summary=$output_root/resident_batch_summary.json

runner_args=(
  --case-dir "$package"
  --runtime-dir "$runtime_dir"
  --manifest "$package/resident_kernel_bank_manifest.json"
  --contract "$contract"
  --host-plan "$package/depthanything_u250_host_plan.json"
  --host-params "$host_params"
  --cfg-dir "$package/cfg"
  --depth-only
  --timeout-ms 5000
  --dma-runtime cpp_mapped
  --layout-codec native
  --layout-codec-report "$codec_report"
  --fpga-dma-batch "$fpga_dma_batch"
  --host-executor cpp
  --host-executor-report "$host_executor_report"
  --attention-launch-group "$attention_launch_group"
  --decoder-launch-group "$decoder_launch_group"
  --encoder-fc1-launch-group "$encoder_fc1_launch_group"
  --frontend-launch-group "$frontend_launch_group"
)
if [[ "$attention_resident_kv" == 1 ]]; then
  runner_args+=(--attention-resident-kv)
fi
if [[ "$encoder_resident_intermediates" == 1 ]]; then
  runner_args+=(--encoder-resident-intermediates)
fi
if [[ "$attention_post_frame_graph" == 1 ]]; then
  runner_args+=(--attention-post-frame-graph)
fi
if [[ "$fused_qkv_attention" == 1 ]]; then
  runner_args+=(--fused-qkv-attention)
  if [[ -n "$fused_qkv_attention_layers" ]]; then
    runner_args+=(--fused-qkv-attention-layers "$fused_qkv_attention_layers")
  fi
fi
if [[ "$fused_attention6" == 1 ]]; then
  runner_args+=(--fused-attention6)
  if [[ -n "$fused_attention6_layers" ]]; then
    runner_args+=(--fused-attention6-layers "$fused_attention6_layers")
  fi
fi
if [[ "$qkv_attention6_frame_graph" == 1 ]]; then
  runner_args+=(--qkv-attention6-frame-graph)
fi
if [[ "$persistent_dma" == 1 ]]; then
  runner_args+=(--cpp-persistent-dma)
fi
if [[ "$decoder_quantize_pack" == 1 ]]; then
  runner_args+=(--decoder-quantize-pack)
fi
if [[ "$encoder_quantize_pack" == 1 ]]; then
  runner_args+=(--encoder-quantize-pack)
fi
if [[ "$mixed_signature_groups" == 1 ]]; then
  runner_args+=(--cpp-mixed-signature-groups)
fi
if [[ "$encoder_fc_frame_graph" == 1 ]]; then
  runner_args+=(--encoder-fc-frame-graph)
fi

"$python_bin" - "$base_args" "${runner_args[@]}" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2) + "\n")
PY

request_arguments=()
samples=("$@")
for sample in "${samples[@]}"; do
  name=${sample##*/}
  input=$input_root/$sample.npy
  output=$output_root/$sample.npz
  [[ -r "$input" ]] || { echo "missing input: $input" >&2; exit 1; }
  mkdir -p "$(dirname "$output")"
  request_arguments+=("$sample" "$input" "$output")
done

# Warm the bank, cfg registry, native extension and C++ runtime with the first
# sample.  Every requested output below is therefore measured in steady state.
warmup_output=$output_root/.warmup/${samples[0]##*/}.npz
mkdir -p "$(dirname "$warmup_output")"
"$python_bin" - "$requests" "$warmup_output" "${request_arguments[@]}" <<'PY'
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1])
warmup_output = sys.argv[2]
items = sys.argv[3:]
if len(items) % 3:
    raise SystemExit("request arguments must be sample/input/output triples")
events = []
first_sample, first_input, _ = items[:3]
events.append({"id": f"warmup:{first_sample}", "input": first_input,
               "output": warmup_output})
for index in range(0, len(items), 3):
    sample, input_path, output_path = items[index:index + 3]
    events.append({"id": sample, "input": input_path, "output": output_path})
events.append({"command": "shutdown"})
destination.write_text("\n".join(json.dumps(item) for item in events) + "\n")
PY

echo "RUN_RESIDENT_BATCH samples=${#samples[@]} warmup=1"
env PYTHONPATH="$base/tools:$base" \
  flock -w "$lock_timeout" "$lock_file" \
  "$python_bin" "$resident_server" --runner "$runner" --base-args "$base_args" \
  <"$requests" >"$server_log" 2>&1

"$python_bin" - "$server_log" "$batch_summary" "${request_arguments[@]}" <<'PY'
import json
from pathlib import Path
import statistics
import sys
import numpy as np

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
items = sys.argv[3:]
expected = {items[index]: Path(items[index + 2])
            for index in range(0, len(items), 3)}
responses = []
for line in log_path.read_text().splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "ok" in event:
        if not event["ok"]:
            raise SystemExit(f"resident request failed: {event}")
        responses.append(event)
measured = [item for item in responses
            if not str(item.get("request_id", "")).startswith("warmup:")]
if {item.get("request_id") for item in measured} != set(expected):
    raise SystemExit("resident responses do not match requested samples")

rows = []
for response in measured:
    sample = response["request_id"]
    output = expected[sample]
    with np.load(output, allow_pickle=False) as values:
        if values.files != ["depth"] or not np.isfinite(values["depth"]).all():
            raise SystemExit(f"invalid depth-only output {output}: {values.files}")
    detail = json.loads(Path(response["summary"]).read_text())
    if not detail["finite"] or detail["static_reloads"] != 0:
        raise SystemExit(f"invalid runtime invariants for {sample}")
    if detail["layout_codec"] != "native":
        raise SystemExit(f"non-native execution for {sample}")
    rows.append({
        **response,
        "frame_wall_ms": detail["wall_ms"],
        "npu_ms": detail["npu_ms_total"],
        "h2c_ms": detail["h2c_ms_total"],
        "c2h_ms": detail["c2h_ms_total"],
        "pack_ms": detail["codec_pack_ms_total"],
        "unpack_ms": detail["codec_unpack_ms_total"],
        "physical_dispatches": detail["npu_calls"],
        "submission_groups": detail["submission_groups"],
        "persistent_dma": not detail["cpp_runtime"]["safe_dma"],
        "decoder_quantize_pack": detail.get("decoder_quantize_pack", False),
        "encoder_quantize_pack": detail.get("encoder_quantize_pack", False),
        "mixed_signature_groups": detail.get("cpp_mixed_signature_groups", False),
        "encoder_fc1_launch_group": detail.get("encoder_fc1_launch_group", 1),
        "frontend_launch_group": detail.get("frontend_launch_group", 1),
        "encoder_fc_frame_graph": detail.get("encoder_fc_frame_graph", False),
    })
metric_names = (
    "request_wall_ms", "process_wall_ms", "frame_wall_ms", "npu_ms",
    "h2c_ms", "c2h_ms", "pack_ms", "unpack_ms", "physical_dispatches",
    "submission_groups",
)
means = {name: statistics.mean(float(row[name]) for row in rows)
         for name in metric_names}
payload = {
    "schema": "depthanything-u250-resident-batch-v1",
    "persistent_dma": all(row["persistent_dma"] for row in rows),
    "decoder_quantize_pack": all(row["decoder_quantize_pack"] for row in rows),
    "encoder_quantize_pack": all(row["encoder_quantize_pack"] for row in rows),
    "mixed_signature_groups": all(row["mixed_signature_groups"] for row in rows),
    "encoder_fc1_launch_group": min(
        row["encoder_fc1_launch_group"] for row in rows
    ),
    "frontend_launch_group": min(row["frontend_launch_group"] for row in rows),
    "encoder_fc_frame_graph": all(row["encoder_fc_frame_graph"] for row in rows),
    "warmup_requests": len(responses) - len(measured),
    "measured_samples": len(rows),
    "means": means,
    "samples": rows,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("RESIDENT_BATCH_SUMMARY=" + json.dumps(payload, sort_keys=True))
PY

echo "PASS_RESIDENT_BATCH $batch_summary"
