# Reproducing the Depth Anything V2 U250 build

This document describes the source-to-package boundary in this repository. A
clean checkout contains all Depth Anything model adapters, graph transforms,
calibration utilities, kernel exporters, DS-Compiler wrappers, resident-bank
linker, host/NPU runtime, and validation code. Generated weights, calibration
tensors, CFG/BIN files, the DS toolchain, and the U250 bitstream remain external.

The qualified deployment is a hybrid graph. DS-Compiler produces one
`*_ddr.bin` and `*_cfg.txt` pair per resident kernel; the linker relocates those
images into one DDR-resident bank. It is not a single monolithic `model.bin`.

## 1. Supported contracts

| Input | Adapter | Patch grid | Tokens |
|---|---|---:|---:|
| `[1,3,518,518]` | `ds_models.depth_anything_v2_vits` | 37x37 | 1370 |
| `[1,3,280,280]` | `ds_models.depth_anything_v2_vits_280` | 20x20 | 401 |

Both use the official Depth Anything V2-Small checkpoint. Preprocessing stays
outside the compiled graph and must produce normalized FP32 NCHW RGB.

## 2. External inputs

Install the public Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-u250.txt
```

The pinned environment includes PyTorch/TorchVision, NumPy, ONNX,
ONNX Runtime, OpenCV, HDF5, Matplotlib, PyYAML and pybind11. Native DS and
board extensions must match the selected Python minor ABI; verify their shared
libraries with `ldd` before use. The complete OS, library and XDMA checklist is
kept in the homepage [dependency contract](../README.md#dependency-contract).

Copy the example environment and replace every path:

```bash
cp configs/depthanything_u250.example.env /tmp/depthanything_u250.env
editor /tmp/depthanything_u250.env
source /tmp/depthanything_u250.env
```

Required external components are:

1. official `depth_anything_v2_vits.pth` checkpoint;
2. DS toolchain containing `python_bin/compile.py`, `python_libs/`,
   `arch_16_mono.yaml`, and `arch_256_mono.yaml`;
3. matching ACMLIR/ACPC build exposed through `PYTHON_ROOT` and
   `ACOMPILER_EXTENSION_DIR`;
4. NYU and DA-2K images for recalibration;
5. on the board host, the matching U250 bitstream, XDMA driver, and vendor
   runtime directory.

The DS compiler and bitstream are not redistributable parts of this Git
repository. Record their revision and SHA-256 alongside every build.

Run the fail-closed environment check before generating anything:

```bash
python tools/check_u250_repro_env.py \
  --output "$DEPTHANYTHING_U250_BUILD_ROOT/environment.json"
```

Add `--require-board` only on a U250 host after loading XDMA.

## 3. Export the fixed-shape FP32 graph

For 518:

```bash
mkdir -p "$DEPTHANYTHING_U250_BUILD_ROOT/518/models"
python tools/export_depth_anything_v2_ds.py \
  --adapter-module ds_models.depth_anything_v2_vits \
  --output "$DEPTHANYTHING_U250_BUILD_ROOT/518/models/depthanything_fp32.onnx"
```

For 280, change the adapter to
`ds_models.depth_anything_v2_vits_280` and the output root to `280`.
The exporter writes the ONNX graph, deterministic input, PyTorch output,
ONNX Runtime output, and an audit JSON. Do not continue if the audit contains
dynamic tensors or non-finite output.

Quantize through the same DS toolchain used for compilation:

```bash
python tools/quantize_depth_anything_v2_ds.py \
  --toolchain-root "$DS_TOOLCHAIN_ROOT" \
  --input "$DEPTHANYTHING_U250_BUILD_ROOT/518/models/depthanything_fp32.onnx"
```

The generated `_sc.onnx` is the source graph for kernel decomposition. It is
not, by itself, the qualified U250 package.

## 4. Rebuild calibration profiles

Create a deterministic NYU+DA-2K input set rather than calibrating individual
layers on unrelated samples. `DA2K_DATASET_ROOT` contains `annotations.json`
and `images/`; `DA2K_IMAGE_ROOT` is its `images/` child:

```bash
export UNIFIED_CAL_ROOT="$DEPTHANYTHING_U250_BUILD_ROOT/unified_calibration"
python tools/build_u250_full_graph_calibration_set.py \
  --nyu-root "$NYU_H5_ROOT" \
  --da2k-root "$DA2K_IMAGE_ROOT" \
  --nyu-count 128 \
  --da2k-count 32 \
  --shapes 280,518 \
  --output-root "$UNIFIED_CAL_ROOT"

python tools/prepare_da2k_calibration_manifest.py \
  --dataset-root "$DA2K_DATASET_ROOT" \
  --output "$DA2K_MANIFEST" \
  --calibration-count 512 \
  --tuning-count 128

python tools/prepare_mixed_calibration_inputs.py \
  --nyu-input-root "$UNIFIED_CAL_ROOT/inputs/518/nyu" \
  --nyu-count 128 \
  --da2k-root "$DA2K_DATASET_ROOT" \
  --da2k-manifest "$DA2K_MANIFEST" \
  --da2k-count 32 \
  --size 518 \
  --output-dir "$DEPTHANYTHING_U250_BUILD_ROOT/518/calibration"
```

Attention statistics are produced by:

```bash
python tools/calibrate_static_int8_attention.py \
  --checkpoint "$DEPTH_ANYTHING_CHECKPOINT" \
  --images "$ATTENTION_JPEG_DIR" \
  --input-size 518 \
  --output "$DEPTHANYTHING_U250_BUILD_ROOT/518/calibration/attention.json"
```

Encoder linear and decoder convolution scales use
`tools/calibrate_a8b8_linears_convs.py`. Its `extract`, `profile`, and `apply`
subcommands deliberately separate trace capture, scale selection, and graph
mutation. The profile manifest must contain both NYU and DA-2K domains. The
newer final-depth workflow is implemented by
`build_u250_full_graph_calibration_set.py`,
`collect_fp32_replacement_traces.py`, and
`calibrate_u250_full_graph_joint.py`.

No empirical output gain is part of the canonical build. Attention accuracy is
controlled by Q/K/V scales and dual-range probability representation.

## 5. Export resident kernels

Use the quantized/calibrated model and the profiles from the previous stage.
Define the build paths once:

```bash
export BUILD="$DEPTHANYTHING_U250_BUILD_ROOT/518"
export MODEL="$BUILD/models/depthanything_fp32_sc.onnx"
```

The reproducible family exporter calls the individual exporters in a fixed
order and records their commands and hashes:

```bash
python tools/export_u250_base_kernels.py \
  --shape 518 \
  --model "$MODEL" \
  --attention-profile "$BUILD/calibration/attention.json" \
  --linear-profile "$BUILD/calibration/linear.json" \
  --decoder-profile "$BUILD/calibration/decoder.json" \
  --output-root "$BUILD/export"
```

The main kernel families are:

```text
export_u250_patch_projection_kernels.py  -> patch_projection_*.onnx
export_u250_qkv_projection_kernels.py    -> qkv_projection_l??.onnx
export_u250_attention_kernels.py         -> six-slice attention source manifest
export_u250_attention_2chunk_kernels.py  -> attention2_l??_h??.onnx
export_u250_encoder_tail_kernels.py       -> post/FC1/FC2 kernels
export_u250_decoder_conv_kernels.py       -> decoder_conv_*.onnx
```

Every exporter writes `manifest.json`. Preserve those manifests: the runtime
contract generator uses names, shapes, scales, and ABI metadata from them.
Run each command with `--help` to see the required model/profile inputs. Shape
parameters must be consistent: 1370 tokens and patch grid 37 for 518; 401
tokens and patch grid 20 for 280.

## 6. Compile kernel ONNX files

All compile wrappers share `tools/u250_compile_env.sh`; no workstation path is
embedded in the scripts. After sourcing the environment file:

```bash
python tools/compile_u250_base_kernels.py \
  --export-root "$BUILD/export" \
  --output-root "$BUILD/compiled" \
  --jobs 8
```

The Python entry point delegates to the five family-specific shell wrappers;
those wrappers remain available for isolated compiler debugging.

Each successful kernel directory must contain non-empty:

```text
<kernel>_cfg.txt
<kernel>_ddr.bin
compile.log
```

For a calibrated replacement set, `compile_u250_full_graph_joint.py` performs
the same operation from the exported manifest and stores SHA-256 values for
every CFG/BIN.

## 7. Link the resident DDR bank

The normal package entry point links the bank and generates the runtime
contract, host plan, host parameters, and a package hash manifest:

```bash
python tools/package_u250_base.py \
  --model "$MODEL" \
  --export-root "$BUILD/export" \
  --compiled-root "$BUILD/compiled" \
  --output-dir "$BUILD/package"
```

The equivalent low-level linker invocation is:

```bash
python tools/link_u250_resident_kernel_bank.py \
  --case-dir "$BUILD/compiled/patch" \
  --case-dir "$BUILD/compiled/qkv" \
  --case-dir "$BUILD/compiled/attention2" \
  --case-dir "$BUILD/compiled/encoder_tail" \
  --case-dir "$BUILD/compiled/decoder" \
  --recursive \
  --shared-fm-workspace-bytes 25165824 \
  --output-dir "$BUILD/package"
```

The linker verifies 256-byte image alignment, parses CFG addresses and tensor
records, relocates every program, reserves the shared feature-map arena, copies
active CFG/IO-order files, and writes:

```text
package/depthanything_u250_resident_kernel_bank.bin
package/resident_kernel_bank_manifest.json
package/cfg/*.txt
package/npz_util.py
```

Generate the NPU/host schedule and host constants:

```bash
python tools/generate_u250_runtime_contract.py \
  --model "$MODEL" \
  --bank-manifest "$BUILD/package/resident_kernel_bank_manifest.json" \
  --qkv-manifest "$BUILD/export/qkv/manifest.json" \
  --qkv-model-dir "$BUILD/export/qkv" \
  --attention-manifest "$BUILD/export/attention2/manifest.json" \
  --encoder-tail-manifest "$BUILD/export/encoder_tail/manifest.json" \
  --decoder-manifest "$BUILD/export/decoder/manifest.json" \
  --patch-manifest "$BUILD/export/patch/manifest.json" \
  --output "$BUILD/package/depthanything_u250_runtime_contract.json"

python tools/export_u250_host_plan.py \
  --model "$MODEL" \
  --decoder-manifest "$BUILD/export/decoder/manifest.json" \
  --patch-manifest "$BUILD/export/patch/manifest.json" \
  --frontend-model "$MODEL" \
  --plan "$BUILD/package/depthanything_u250_host_plan.json" \
  --params "$BUILD/package/depthanything_u250_host_params.npz"
```

## 8. Build and validate the runtime

Build the C++ mapped-buffer/native-codec extension on the U250 host:

```bash
make -f tools/Makefile.u250_runtime dma-batch PYTHON="$DS_COMPILER_PYTHON"
python -m pytest -q tests
```

Run `tools/run_u250_depthanything_hybrid.py --help` for the explicit package,
runtime, input, codec, and output arguments. Production validation must use the
resident server/batch scripts, acquire the board lock, perform a warm-up frame,
and record free-running final-depth metrics. A successful compile or CModel run
is not evidence of U250 board success.

## 9. Reproduction record

Archive, outside Git, a manifest containing:

- Git commit and clean/dirty status;
- checkpoint SHA-256;
- DS toolchain, ACMLIR/ACPC, architecture YAML, and bitstream SHA-256;
- calibration-set manifest SHA-256;
- every exported ONNX, CFG, DDR BIN, resident bank, runtime contract, and host
  plan SHA-256;
- board serial/bitstream, XDMA version, accuracy metrics, mean/p95 latency, and
  raw logs.

Generated artifacts are intentionally ignored by the source repository and
should be stored in a versioned artifact store.
