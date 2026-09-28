# DepthAnything U250 Native Layout Codec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all r58 vendor tensor layout conversions with bit-exact C++ codecs and reduce steady U250 latency without changing the r43 model output.

**Architecture:** Vendor the qualified `fpgaDmaBatch` source into this repository, extend it with descriptor-driven NCHW/NDWC INT8/BF16 codecs, and select those codecs through a fail-closed Python registry. Official `npz2bin` remains the oracle for descriptor qualification but is not called during a production `native` inference.

**Tech Stack:** C++17, pybind11, NumPy, pytest, DS `npz2bin`, XDMA/U250, Bash.

**Spec:** `docs/superpowers/specs/2026-09-07-depthanything-u250-native-layout-codec.md`

## Global Constraints

- Preserve the r43 resident bank SHA-256 `9d01d1fd4ecae67755a4314e98a2f9d4cbe7182f3985d7573113de579b7f9577`.
- Preserve final-depth SHA-256 `2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.
- Keep 443 hardware dispatches and 248 C++ submission groups for a complete frame.
- `native` mode must fail before U250 access unless every active descriptor is qualified for its requested direction.
- Production qualification requires zero vendor pack calls and zero vendor unpack calls.
- Never interrupt unrelated board processes; acquire `/tmp/ds-u250-runtime.lock` with non-blocking `flock` before U250 access.
- Do not modify quantization scales, BF16 numerical boundaries, operators, BIN instruction ranges, weights, or model scheduling.
- Preserve all existing `fpgaDmaBatch` APIs used by CompletionFormer.

---

### Task 1: Parse Stable Tensor Descriptors

**Files:**
- Create: `tools/u250_layout_descriptors.py`
- Create: `tests/test_u250_layout_descriptors.py`
- Modify: `tools/run_u250_depthanything_hybrid.py:25-104`

**Interfaces:**
- Produces: `TensorLayoutDescriptor.from_tensor(case_name: str, tensor: dict, direction: str, index: int) -> TensorLayoutDescriptor`
- Produces: `TensorLayoutDescriptor.identity() -> str`
- Produces: `build_case_descriptors(records: dict[str, dict]) -> dict[str, dict[str, list[TensorLayoutDescriptor]]]`
- Consumes: manifest tensor fields `layout`, `dims`, `bitdepth`, `c_align`, `w_align`, and `size_per_bank`.

- [ ] **Step 1: Write failing descriptor tests**

```python
def test_descriptor_identity_ignores_relocated_address():
    a = {"address": 0, "layout": "NDWC", "dims": [1, 1, 256, 64],
         "bitdepth": 8, "c_align": 4, "w_align": 64,
         "size_per_bank": 16384}
    b = {**a, "address": 9000}
    left = TensorLayoutDescriptor.from_tensor("qkv_projection_l00", a, "input", 0)
    right = TensorLayoutDescriptor.from_tensor("qkv_projection_l00", b, "input", 0)
    assert left.identity() == right.identity()


def test_descriptor_rejects_invalid_extent():
    tensor = {"layout": "NCHW", "dims": [1, 64, 37, 37],
              "bitdepth": 16, "c_align": 8, "w_align": 24,
              "size_per_bank": 227327}
    with pytest.raises(ValueError, match="256-byte aligned"):
        TensorLayoutDescriptor.from_tensor("decoder_conv_00", tensor, "output", 0)
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `pytest -q tests/test_u250_layout_descriptors.py`

Expected: collection fails because `tools.u250_layout_descriptors` does not exist.

- [ ] **Step 3: Implement the immutable descriptor**

```python
@dataclass(frozen=True)
class TensorLayoutDescriptor:
    layout: str
    dims: tuple[int, int, int, int]
    bitdepth: int
    c_align: int
    w_align: int
    combined_bytes: int
    direction: str
    index: int
    matrix_role: str

    def identity(self) -> str:
        payload = asdict(self)
        payload.pop("index")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
```

Validate rank four, positive dimensions/alignment, layout in `NCHW/NDWC`,
bitdepth in `8/16`, direction in `input/output`, and `combined_bytes % 256 == 0`.
Set `matrix_role="netio"` for NCHW, `"output"` for every NDWC output,
`"right"` for attention input indices 1 and 2, and `"left"` for all other
NDWC inputs. Pass the case name into `from_tensor` so this rule is explicit;
include `matrix_role` in the stable identity.

- [ ] **Step 4: Replace textual signatures with descriptors**

Make `CfgCodecRegistry.signature(name)` return a tuple of descriptor identities
and retain descriptors under `registry.descriptors[name][direction]`. Preserve
the existing grouping result of 248 groups.

- [ ] **Step 5: Run unit and static grouping tests**

Run: `pytest -q tests/test_u250_layout_descriptors.py tests/test_u250_cpp_mapped_runtime.py`

Expected: all tests pass and a static manifest check prints `262 cases`, `30
case codec signatures`, `443 dispatches`, and `248 groups`.

- [ ] **Step 6: Commit**

```bash
git add tools/u250_layout_descriptors.py tools/run_u250_depthanything_hybrid.py tests/test_u250_layout_descriptors.py
git commit -m "feat: parse stable U250 tensor layout descriptors"
```

---

### Task 2: Establish the Qualified Extension Baseline

**Files:**
- Create: `tools/fpga_dma_batch.cpp`
- Create: `tools/Makefile.u250_runtime`
- Create: `tests/test_fpga_dma_batch_api.py`

**Interfaces:**
- Consumes: qualified source `/root/demo/artifacts/ds_completionformer_original_20260901/static_runtime/fpga_dma_batch.cpp`.
- Produces: `fpgaDmaBatch.DmaBatch` with every existing method unchanged.
- Produces: `make -f tools/Makefile.u250_runtime dma-batch PYTHON=/home/visitor/anaconda3/envs/ds/bin/python`.

- [ ] **Step 1: Write a failing API compatibility test**

```python
EXPECTED = {
    "h2c_batch", "h2c_batch_safe", "c2h_batch", "c2h_batch_safe",
    "run_npu_chain", "run_cbam_fused_pool", "pack_int8_nchw",
    "pack_int8_nchw_segments", "unpack_int8_nchw",
    "interleave_polyphase_normal16", "requantize_normal16",
    "crop_normal16_tiles", "scatter_normal16_tiles", "stats", "reset_stats",
}


def test_extension_preserves_qualified_api(extension_type):
    assert EXPECTED <= set(dir(extension_type))
```

The fixture reads the extension path from `FPGA_DMA_BATCH_SO` and skips only
when that environment variable is absent.

- [ ] **Step 2: Verify the test fails against a missing repository build**

Run: `FPGA_DMA_BATCH_SO=build/native_codec/fpgaDmaBatch.so pytest -q tests/test_fpga_dma_batch_api.py`

Expected: failure because the repository-owned extension has not been built.

- [ ] **Step 3: Add the qualified source without functional edits**

Add the exact CompletionFormer source as `tools/fpga_dma_batch.cpp`. Verify it
before changes:

```bash
cmp tools/fpga_dma_batch.cpp /root/demo/artifacts/ds_completionformer_original_20260901/static_runtime/fpga_dma_batch.cpp
```

Expected: exit code 0.

- [ ] **Step 4: Add the reproducible build target**

```make
dma-batch:
	mkdir -p build/native_codec
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Werror -pthread -shared -fPIC \
	  $$($(PYTHON) -m pybind11 --includes) tools/fpga_dma_batch.cpp \
	  -o build/native_codec/fpgaDmaBatch$$($(PYTHON) -c \
	  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
```

- [ ] **Step 5: Build and run compatibility tests on the U250 host**

Deploy the source and Makefile without taking the board lock, build with the
DS Python 3.13 environment, then run:

```bash
FPGA_DMA_BATCH_SO=build/native_codec/fpgaDmaBatch.cpython-313-x86_64-linux-gnu.so \
  /home/visitor/anaconda3/envs/ds/bin/python -m pytest -q tests/test_fpga_dma_batch_api.py
```

Expected: API test passes; `sha256sum` of the newly built unmodified extension
is recorded in `artifacts/u250_native_codec/extension_baseline.json`.

- [ ] **Step 6: Commit**

```bash
git add tools/fpga_dma_batch.cpp tools/Makefile.u250_runtime tests/test_fpga_dma_batch_api.py artifacts/u250_native_codec/extension_baseline.json
git commit -m "build: vendor qualified U250 DMA extension source"
```

---

### Task 3: Add Descriptor Validation and BF16 Conversion

**Files:**
- Modify: `tools/fpga_dma_batch.cpp`
- Modify: `tests/test_fpga_dma_batch_api.py`
- Create: `tests/test_u250_native_codec.py`

**Interfaces:**
- Produces: `DmaBatch.validate_descriptor(descriptor: dict) -> dict`
- Produces internally: `LayoutDescriptor parse_descriptor(const py::dict&)`
- Produces internally: `uint16_t fp32_to_bf16_rne(float)` and `float bf16_to_fp32(uint16_t)`.

- [ ] **Step 1: Write failing validation and BF16 behavior tests**

```python
def test_validate_descriptor_normalizes_fields(codec):
    result = codec.validate_descriptor({
        "layout": "NDWC", "dims": [1, 1, 1370, 384], "bitdepth": 16,
        "c_align": 48, "w_align": 4128, "combined_bytes": 1056768,
        "direction": "output", "index": 0,
    })
    assert result["half_bytes"] == 528384
    assert result["elements"] == 526080


@pytest.mark.parametrize("value,bits", [
    (0.0, 0x0000), (-0.0, 0x8000), (1.0, 0x3F80), (-2.0, 0xC000),
])
def test_bf16_known_values(codec, value, bits):
    assert codec.test_fp32_to_bf16(np.float32(value)) == bits
```

Also test rank, layout, bitdepth, direction, dtype, unequal bank sizes, and
undersized/unaligned extents.

- [ ] **Step 2: Verify RED**

Run the extension tests with `FPGA_DMA_BATCH_SO` set.

Expected: failure because `validate_descriptor` and the test conversion hook
are absent.

- [ ] **Step 3: Implement parsing and exact BF16 conversion**

Use integer round-to-nearest-even conversion:

```cpp
uint32_t bits;
std::memcpy(&bits, &value, sizeof(bits));
const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
return static_cast<uint16_t>((bits + rounding) >> 16U);
```

Reject NaN/Inf inputs to match the current runtime's finite-tensor contract.
Expose `validate_descriptor`; keep `test_fp32_to_bf16` module-private by naming
it `_test_fp32_to_bf16`.

- [ ] **Step 4: Verify GREEN and existing API compatibility**

Run: `pytest -q tests/test_fpga_dma_batch_api.py tests/test_u250_native_codec.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/fpga_dma_batch.cpp tests/test_fpga_dma_batch_api.py tests/test_u250_native_codec.py
git commit -m "feat: validate DS layout descriptors in C++"
```

---

### Task 4: Implement NCHW INT8 and BF16 Codecs

**Files:**
- Modify: `tools/fpga_dma_batch.cpp`
- Modify: `tests/test_u250_native_codec.py`
- Create: `tools/validate_u250_native_codecs.py`

**Interfaces:**
- Produces: `DmaBatch.pack_tensor(array, descriptor) -> (even, odd)` for NCHW.
- Produces: `DmaBatch.unpack_tensor(even, odd, descriptor) -> ndarray` for NCHW.
- Consumes: existing `normal16_index`, `layout_kind`, and `parallel_rows`.

- [ ] **Step 1: Write failing NCHW round-trip tests**

```python
@pytest.mark.parametrize("shape,bitdepth", [
    ((1, 64, 37, 37), 8), ((1, 48, 76, 148), 8),
    ((1, 64, 74, 148), 16), ((1, 1, 74, 518), 16),
])
def test_nchw_round_trip(codec, descriptor_factory, shape, bitdepth):
    logical = deterministic_tensor(shape, bitdepth)
    descriptor = descriptor_factory("NCHW", shape, bitdepth)
    even, odd = codec.pack_tensor(logical, descriptor)
    restored = codec.unpack_tensor(even, odd, descriptor)
    expected = bf16_reference(logical) if bitdepth == 16 else logical
    np.testing.assert_array_equal(restored, expected)
```

- [ ] **Step 2: Verify RED**

Expected: `pack_tensor` is absent.

- [ ] **Step 3: Implement NCHW physical indexing**

Reuse the established normal16/compact4 bank selection and index formulas.
For 16-bit elements, store two little-endian BF16 bytes per lane and double
the lane byte offset. Allocate each bank to exactly `combined_bytes / 2` and
zero it before writing logical values so padded C/W tails remain zero.

- [ ] **Step 4: Add official NCHW oracle comparison**

`validate_u250_native_codecs.py` loads the manifest and cfg directory, selects
all unique NCHW descriptors, produces deterministic logical arrays, invokes
official `npz2bin` pack/unpack, invokes C++, and compares complete physical
buffers and logical arrays. Its result schema is:

```json
{
  "manifest_sha256": "...",
  "descriptors": [{
    "identity": "...", "layout": "NCHW", "bitdepth": 8,
    "pack_exact": true, "unpack_exact": true,
    "native_pack_ms": 0.0, "vendor_pack_ms": 0.0
  }]
}
```

- [ ] **Step 5: Run every NCHW descriptor on the U250 host without board access**

Run the validator with the r43 package and write
`artifacts/u250_native_codec/nchw_oracle.json`.

Expected: every listed direction is exact; rejected descriptors include an
explicit first mismatch instead of being marked qualified.

- [ ] **Step 6: Commit**

```bash
git add tools/fpga_dma_batch.cpp tools/validate_u250_native_codecs.py tests/test_u250_native_codec.py artifacts/u250_native_codec/nchw_oracle.json
git commit -m "feat: add bit-exact NCHW U250 codecs"
```

---

### Task 5: Implement NDWC Matrix INT8 and BF16 Codecs

**Files:**
- Modify: `tools/fpga_dma_batch.cpp`
- Modify: `tests/test_u250_native_codec.py`
- Modify: `tools/validate_u250_native_codecs.py`

**Interfaces:**
- Extends: `DmaBatch.pack_tensor` and `unpack_tensor` for NDWC.
- Produces internally: `matrix_physical_index(descriptor, n, d, w, c) -> (bank, byte_offset)`.

- [ ] **Step 1: Write failing NDWC geometry tests**

```python
@pytest.mark.parametrize("shape,bitdepth,combined", [
    ((1, 1, 256, 64), 8, 16384),
    ((1, 1, 64, 1370), 8, 88064),
    ((1, 1, 1370, 384), 8, 528384),
    ((1, 1, 1370, 384), 16, 1056768),
    ((1, 1, 256, 64), 16, 32768),
])
def test_ndwc_oracle_fixture_exact(codec, oracle_fixture, shape, bitdepth, combined):
    logical, vendor_even, vendor_odd, descriptor = oracle_fixture(
        "NDWC", shape, bitdepth, combined
    )
    even, odd = codec.pack_tensor(logical, descriptor)
    np.testing.assert_array_equal(even, vendor_even)
    np.testing.assert_array_equal(odd, vendor_odd)
```

- [ ] **Step 2: Verify RED**

Expected: the C++ codec rejects `layout=NDWC`.

- [ ] **Step 3: Implement matrix tiling from DS loop rules**

Implement the NetIOMM/MM rules as nested padded blocks, using descriptor
`matrix_role`, `c_align`, and `w_align` to select left-matrix, right-matrix, or
output ordering. Keep bank selection and lane ordering in one
`matrix_physical_index` function. Do not infer a role from tensor values.

Validate the explicit role against all attention inputs: Q0/Q1 use `left`,
transposed K and V use `right`; qkv input, post-attention input, MLP FC1 input,
and FC2 input use `left`; every NDWC output uses `output`.

- [ ] **Step 4: Run full NDWC oracle qualification**

Run `validate_u250_native_codecs.py` for every unique NDWC direction and save
`artifacts/u250_native_codec/ndwc_oracle.json`.

Expected: all required descriptors are byte-exact. No descriptor may be
declared exact solely from a logical round trip; physical bytes must match the
vendor oracle.

- [ ] **Step 5: Benchmark each descriptor**

Use five warm repetitions and report median native/vendor time. Mark a
descriptor production-enabled only when it is exact and native median time is
strictly lower than vendor median time. Save results in the same oracle JSON.

- [ ] **Step 6: Commit**

```bash
git add tools/fpga_dma_batch.cpp tools/validate_u250_native_codecs.py tests/test_u250_native_codec.py artifacts/u250_native_codec/ndwc_oracle.json
git commit -m "feat: add bit-exact NDWC U250 codecs"
```

---

### Task 6: Integrate Fail-Closed Runtime Codec Selection

**Files:**
- Modify: `tools/u250_cpp_mapped_runtime.py:88-273`
- Modify: `tools/run_u250_depthanything_hybrid.py:238-560`
- Modify: `tests/test_u250_cpp_mapped_runtime.py`
- Create: `tests/test_u250_codec_selection.py`

**Interfaces:**
- Produces: CLI `--layout-codec {vendor,auto,native}`.
- Consumes: qualification JSON through `--layout-codec-report PATH`.
- Produces: `CppMappedRuntime.pack_tensor(array, descriptor)` and `unpack_tensor(even, odd, descriptor)`.
- Produces counters: `native_pack_calls`, `native_unpack_calls`, `vendor_pack_calls`, `vendor_unpack_calls`, byte counts, elapsed milliseconds, and `fallback_reasons`.

- [ ] **Step 1: Write failing fail-closed selection tests**

```python
def test_native_rejects_unqualified_descriptor_before_bank_load(runtime_factory):
    runtime = runtime_factory(mode="native", qualified=set())
    with pytest.raises(RuntimeError, match="not native_exact"):
        runtime.prepare(active_descriptors={"abc"})
    assert runtime.transport.h2c == []


def test_auto_records_vendor_fallback(runtime_factory):
    runtime = runtime_factory(mode="auto", qualified=set())
    runtime.pack(np.zeros((1, 1, 1, 4), np.int8), descriptor("abc"))
    assert runtime.stats()["vendor_pack_calls"] == 1
    assert runtime.stats()["fallback_reasons"]["abc"] == "not native_exact"
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_u250_codec_selection.py`

Expected: failure because codec mode and qualification report loading do not
exist.

- [ ] **Step 3: Implement preflight and native bank-pair data flow**

Before `ensure_bank`, verify the report's `manifest_sha256`, all active
descriptor identities, required directions, and performance-enabled flags.
Change grouped calls to carry `(even, odd)` bank pairs for native tensors.
Vendor fallback may still return a combined buffer, but native execution must
not invoke `split_combined_ddr` or `merge_combined_ddr`.

- [ ] **Step 4: Add codec accounting to summaries**

Replace the aggregate `codec_pack_ms_total`/`codec_unpack_ms_total` fields with
native and vendor subfields while retaining the old totals for report
compatibility. Assert zero vendor calls in `native` mode.

- [ ] **Step 5: Verify unit tests and legacy behavior**

Run:

```bash
pytest -q tests/test_u250_layout_descriptors.py \
  tests/test_u250_cpp_mapped_runtime.py tests/test_u250_codec_selection.py
```

Expected: all pass; `--layout-codec vendor` retains the r58 control-flow hash
and metrics.

- [ ] **Step 6: Commit**

```bash
git add tools/u250_cpp_mapped_runtime.py tools/run_u250_depthanything_hybrid.py tests/test_u250_cpp_mapped_runtime.py tests/test_u250_codec_selection.py
git commit -m "feat: select native U250 codecs fail closed"
```

---

### Task 7: Qualify the Complete CPU Control Flow

**Files:**
- Modify: `tools/run_u250_mapped_r58_gate.sh`
- Create: `tools/run_u250_native_codec_controlflow.py`
- Create: `artifacts/u250_native_codec/full_controlflow.summary.json`

**Interfaces:**
- Consumes: r43 manifest, cfg, contract, host plan, host parameters, demo05 input, and qualification report.
- Produces: 443-call zero-device summary with codec call counts and stage timings.

- [ ] **Step 1: Write a failing control-flow gate assertion**

```python
def assert_native_controlflow(summary):
    assert summary["npu_calls"] == 443
    assert summary["submission_groups"] == 248
    assert summary["vendor_pack_calls"] == 0
    assert summary["vendor_unpack_calls"] == 0
    assert summary["native_pack_calls"] == 1103
    assert summary["native_unpack_calls"] == 683
```

- [ ] **Step 2: Verify RED against the r58 summary**

Run the assertion on
`artifacts/u250_mapped_runtime_r58/full_controlflow_fake_transport.summary.json`.

Expected: failure because r58 does not report native codec counters.

- [ ] **Step 3: Implement the zero-device native control-flow runner**

Reuse the existing fake NPU/DMA transport, but execute real native pack/unpack
for all tensors. Emit total and per-layout timing without opening XDMA nodes.

- [ ] **Step 4: Run on the U250 host CPU**

Expected summary:

```text
npu_calls=443
submission_groups=248
vendor_pack_calls=0
vendor_unpack_calls=0
native_pack_calls=1103
native_unpack_calls=683
```

Require native pack+unpack wall time below the r58 vendor total of
12,841.621 ms before proceeding to the board.

- [ ] **Step 5: Commit**

```bash
git add tools/run_u250_native_codec_controlflow.py tools/run_u250_mapped_r58_gate.sh artifacts/u250_native_codec/full_controlflow.summary.json
git commit -m "test: qualify full native codec control flow"
```

---

### Task 8: Run U250 Accuracy and Latency Gates

**Files:**
- Modify: `tools/run_u250_mapped_r58_gate.sh`
- Modify: `docs/U250_MAPPED_RUNTIME_R58.md`
- Create: `artifacts/u250_native_codec/gate_summary.json`
- Create: `artifacts/u250_native_codec/demo05_full_first.summary.json`
- Create: `artifacts/u250_native_codec/demo05_full_resident.summary.json`

**Interfaces:**
- Consumes: deployed native extension and qualification report.
- Produces: decoder-only, layer-11 resume, full first-frame, and full resident-frame board evidence.

- [ ] **Step 1: Extend the non-blocking board gate**

Add `--layout-codec native`, `--layout-codec-report`, and assertions for zero
fallback. Preserve exit code 75 when the lock is busy.

- [ ] **Step 2: Deploy to a versioned directory**

Deploy under
`/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_nativecodec_r59`
without replacing the r43/r58 package. Verify source, extension, manifest, and
report SHA-256 values before execution.

- [ ] **Step 3: Run the four-stage gate under the board lock**

Run decoder-only, layer-11 resume, full first frame, and two same-process full
frames. Do not use persistent-descriptor diagnostic DMA unless the safe mode
gate has already passed.

- [ ] **Step 4: Assert accuracy and residency**

For every stage require exact final hash, relative L2 0, RMSE 0, no timeout,
and no stale event. For the second frame additionally require:

```text
resident_bank_reused=true
load_ms=0
codec_yaml_reused=true
vendor_pack_calls=0
vendor_unpack_calls=0
npu_calls=443
submission_groups=248
```

- [ ] **Step 5: Report measured latency**

Record first/steady wall, pack, unpack, H2C, NPU, C2H, decoder host, and
residual times. Compare measured steady latency with 60,561.193 ms and
14,202.047 ms; do not substitute estimates for missing measurements.

- [ ] **Step 6: Run final verification**

```bash
pytest -q tests/test_u250_layout_descriptors.py \
  tests/test_fpga_dma_batch_api.py tests/test_u250_native_codec.py \
  tests/test_u250_cpp_mapped_runtime.py tests/test_u250_codec_selection.py
python3 -m py_compile tools/u250_layout_descriptors.py \
  tools/u250_cpp_mapped_runtime.py tools/run_u250_depthanything_hybrid.py \
  tools/validate_u250_native_codecs.py tools/run_u250_native_codec_controlflow.py
bash -n tools/run_u250_mapped_r58_gate.sh
git diff --check
```

Expected: all tests pass, all scripts compile, shell syntax passes, and the
worktree contains no unintended changes.

- [ ] **Step 7: Commit**

```bash
git add tools/run_u250_mapped_r58_gate.sh docs/U250_MAPPED_RUNTIME_R58.md artifacts/u250_native_codec
git commit -m "perf: qualify native tensor codecs on U250"
```
