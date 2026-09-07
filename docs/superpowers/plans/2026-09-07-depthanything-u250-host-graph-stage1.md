# DepthAnything U250 Host Graph Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the subtraction-only host residual with attributable timings, then move qualified quantization, GELU-to-INT8, residual, and tensor-assembly hot paths into a device-free C++ host executor while preserving the r60 U250 result bit for bit.

**Architecture:** Add a small Python timing/accounting module and a `HostGraphExecutor` pybind class to the existing `fpgaDmaBatch` extension. The host executor never opens XDMA devices, owns reusable CPU buffers, and is selected independently from the DMA transport; Python remains the request boundary and retains exact fallbacks for qualification and bisecting. This plan implements Stage 1 only because Stages 2–4 depend on the measured Stage 1 board profile; each subsequent stage receives its own evidence-derived implementation plan.

**Tech Stack:** Python 3.13, NumPy, C++17, pybind11, pytest, DS runtime, XDMA/U250, Bash.

**Spec:** `docs/superpowers/specs/2026-09-07-depthanything-u250-host-graph-optimization.md`

## Global Constraints

- Preserve resident bank SHA-256 `9d01d1fd4ecae67755a4314e98a2f9d4cbe7182f3985d7573113de579b7f9577`.
- Preserve final-depth SHA-256 `2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.
- Preserve 443 physical NPU dispatches and 248 C++ submission groups in Stage 1.
- Preserve safe DMA, per-frame stale-event reset, resident bank reuse, and `load_ms=0` for the second full request.
- Preserve all existing `DmaBatch` methods and the CompletionFormer ABI.
- Do not change BIN bytes, cfg files, scales, model weights, attention partitioning, decoder partitioning, or the bitstream.
- `--host-executor cpp` must fail before device access if the extension lacks the required host API or its qualification digest does not match.
- Python fallback remains available as `--host-executor python`; `auto` may choose C++ only with complete qualification evidence.
- New CPU and board evidence goes under fresh r61 directories; never overwrite r60 evidence or its deployed package.
- Acquire `/tmp/ds-u250-runtime.lock` non-blockingly before any board execution; lock contention exits without interrupting another process.

---

### Task 1: Add Attributable Host Timing

**Files:**
- Create: `tools/u250_host_profile.py`
- Create: `tests/test_u250_host_profile.py`
- Modify: `tools/run_u250_depthanything_hybrid.py:465-1090`
- Modify: `tools/check_u250_native_board_gate.py:18-76`

**Interfaces:**
- Produces: `HostProfiler.measure(name: str, *, elements: int = 0, nbytes: int = 0)` context manager.
- Produces: `HostProfiler.record(name: str, elapsed_ms: float, *, elements: int = 0, nbytes: int = 0) -> None`.
- Produces: `HostProfiler.summary(process_wall_ms: float, externally_accounted_ms: float) -> dict`.
- Produces summary fields `host_profile`, `host_profile_ms_total`, and `unattributed_host_residual_ms`.
- Consumes the existing separately measured bank, cfg, codec, DMA, NPU, and decoder-host times.

- [ ] **Step 1: Write failing profiler tests**

```python
from tools.u250_host_profile import HostProfiler


def test_profiler_aggregates_calls_work_and_time(monkeypatch):
    ticks = iter((10.000, 10.002, 20.000, 20.003))
    monkeypatch.setattr("tools.u250_host_profile.time.perf_counter", lambda: next(ticks))
    profile = HostProfiler()
    with profile.measure("encoder.quantize", elements=4, nbytes=16):
        pass
    with profile.measure("encoder.quantize", elements=6, nbytes=24):
        pass
    assert profile.operations["encoder.quantize"] == {
        "calls": 2, "ms": 5.0, "elements": 10, "bytes": 40,
    }


def test_summary_reconciles_process_wall_without_hiding_negative_time():
    profile = HostProfiler()
    profile.record("encoder.gelu", 25.0, elements=100, nbytes=400)
    result = profile.summary(process_wall_ms=100.0, externally_accounted_ms=60.0)
    assert result["host_profile_ms_total"] == 25.0
    assert result["unattributed_host_residual_ms"] == 15.0
    with pytest.raises(ValueError, match="exceeds process wall"):
        profile.summary(process_wall_ms=50.0, externally_accounted_ms=40.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_u250_host_profile.py`

Expected: collection fails with `ModuleNotFoundError: tools.u250_host_profile`.

- [ ] **Step 3: Implement the timing accumulator**

```python
@dataclass
class HostProfiler:
    operations: dict[str, dict[str, int | float]] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str, *, elements: int = 0, nbytes: int = 0):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - started) * 1000.0,
                        elements=elements, nbytes=nbytes)

    def record(self, name: str, elapsed_ms: float, *, elements: int = 0,
               nbytes: int = 0) -> None:
        if not name or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("invalid host timing")
        item = self.operations.setdefault(
            name, {"calls": 0, "ms": 0.0, "elements": 0, "bytes": 0})
        item["calls"] += 1
        item["ms"] += float(elapsed_ms)
        item["elements"] += int(elements)
        item["bytes"] += int(nbytes)
```

Implement `summary()` with a one-microsecond tolerance. It raises if explicit
time exceeds the process wall beyond that tolerance and otherwise returns a
sorted operation dictionary plus the non-negative residual.

- [ ] **Step 4: Instrument material host families in the runner**

Create one `HostProfiler` after input loading. Measure these exact names around
the existing expressions without reordering arithmetic:

```python
"frontend.patchify"
"frontend.token_assembly"
"encoder.layernorm_affine"
"encoder.quantize_qkv"
"encoder.attention_input_assembly"
"encoder.attention_output_assembly"
"encoder.post_attention_quantize"
"encoder.residual"
"encoder.quantize_fc1"
"encoder.gelu"
"encoder.quantize_fc2"
"decoder.quantize"
"decoder.tile_assembly"
"decoder.output_assembly"
"result.serialize"
```

Use the tensor's logical element count and bytes at the boundary, for example:

```python
with host_profiler.measure("encoder.gelu", elements=hidden.size,
                           nbytes=hidden.nbytes):
    activated = gelu(hidden)
```

Do not include codec, DMA, NPU, or `execute_host()` intervals in these scopes.
Keep `decoder_host_ops` unchanged so historical operator comparisons remain
valid.

- [ ] **Step 5: Replace subtraction-only reporting with reconciled fields**

Keep `host_graph_and_python_residual_ms` as a compatibility alias for the sum
of `host_profile_ms_total + unattributed_host_residual_ms`. Add both new fields
to `latency_breakdown` and add the detailed `host_profile` at summary top
level. Rename the NPU-only FC1 statistics key from
`encoder_mlp_fc1_gelu` to `encoder_mlp_fc1`; host GELU is reported only under
`host_profile["encoder.gelu"]`.

- [ ] **Step 6: Strengthen board-summary validation**

```python
profile = report.get("host_profile")
require(isinstance(profile, dict) and profile, f"{name}: missing host profile")
for op, item in profile.items():
    require(isinstance(op, str) and op and type(item.get("calls")) is int,
            f"{name}: malformed host profile")
    for field in ("ms", "elements", "bytes"):
        require(type(item.get(field)) in (int, float)
                and math.isfinite(item[field]) and item[field] >= 0,
                f"{name}: invalid host profile {op}.{field}")
```

Validate that the new explicit components plus
`unattributed_host_residual_ms` reconcile to `process_wall_ms` within 0.5 ms.

- [ ] **Step 7: Verify GREEN and regression safety**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_u250_host_profile.py \
  tests/test_u250_native_board_gate.py tests/test_u250_native_controlflow.py
```

Expected: all tests pass; existing r60 evidence is accepted through a schema-1
compatibility branch, while new schema-2 fixtures fail if detailed host timing
is absent or inconsistent.

- [ ] **Step 8: Commit the profiler**

```bash
git add tools/u250_host_profile.py tools/run_u250_depthanything_hybrid.py \
  tools/check_u250_native_board_gate.py tests/test_u250_host_profile.py \
  tests/test_u250_native_board_gate.py tests/test_u250_native_controlflow.py
git commit -m "perf: attribute U250 host graph latency"
```

---

### Task 2: Add a Device-Free C++ Host Executor

**Files:**
- Create: `tools/u250_host_graph.hpp`
- Modify: `tools/fpga_dma_batch.cpp:1-20,1592-1649`
- Modify: `tools/Makefile.u250_runtime`
- Create: `tests/test_u250_host_graph_executor.py`
- Modify: `tests/test_fpga_dma_batch_api.py:11-19`

**Interfaces:**
- Produces: `fpgaDmaBatch.HostGraphExecutor()`; constructing it must not open a device.
- Produces: `HostGraphExecutor.quantize(input: float32[N], scale: float) -> int8[N]`.
- Produces: `HostGraphExecutor.gelu_quantize(input: float32[N], scale: float) -> int8[N]`.
- Produces: `HostGraphExecutor.add(left: float32[N], right: float32[N]) -> float32[N]`.
- Produces: `HostGraphExecutor.add_quantize(left: float32[N], right: float32[N], scale: float) -> int8[N]`.
- Produces: `HostGraphExecutor.concatenate(inputs: list[array], axis: int) -> array` for C-contiguous equal-rank arrays.
- Produces: `HostGraphExecutor.stats() -> dict` and `reset_stats() -> None`.

- [ ] **Step 1: Write failing extension API and numerical tests**

Compile the extension fixture from repository source without constructing
`DmaBatch`; instantiate only `HostGraphExecutor`.

```python
def python_quantize(value, scale):
    return np.clip(np.rint(np.asarray(value, np.float32) / scale),
                   -128, 127).astype(np.int8)


def test_host_executor_construction_does_not_open_devices(extension):
    executor = extension.HostGraphExecutor()
    assert executor.stats()["host_calls"] == 0


@pytest.mark.parametrize("scale", [0.00390625, 0.03993530943989754, 0.10580708831548691])
def test_quantize_is_bit_exact(extension, scale):
    values = np.array([-1000, -127.5 * scale, -0.5 * scale, -0.0,
                       0.5 * scale, 126.5 * scale, 1000], np.float32)
    actual = extension.HostGraphExecutor().quantize(values, scale)
    assert actual.dtype == np.int8
    assert np.array_equal(actual, python_quantize(values, scale))


def test_gelu_quantize_matches_runtime_boundary(extension):
    values = np.linspace(-12, 12, 65537, dtype=np.float32).reshape(1, 1, 1, -1)
    expected = python_quantize(hybrid.gelu(values), 0.03993530943989754)
    actual = extension.HostGraphExecutor().gelu_quantize(
        values, 0.03993530943989754)
    assert np.array_equal(actual, expected)
```

Also cover non-contiguous input, wrong dtype, shape mismatch, invalid scale,
NaN/Inf, negative axes, and concatenation on axes 0 through rank minus one.

- [ ] **Step 2: Run extension tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_u250_host_graph_executor.py \
  tests/test_fpga_dma_batch_api.py
```

Expected: failures report that `HostGraphExecutor` is absent.

- [ ] **Step 3: Implement the host executor without device state**

Define `HostGraphExecutor` in `u250_host_graph.hpp`. It owns only a mutex,
reusable vectors, and counters. It must not contain or construct a `DmaBatch`.
Use `py::gil_scoped_release` during element loops and the existing
`parallel_rows` helper. Quantization uses `std::nearbyint`, clamps to
`[-128,127]`, and rejects non-finite data.

Implement GELU with explicit FP32 temporaries matching the Python expression:

```cpp
float gelu_scalar(float value) {
  constexpr float inv_sqrt2 = 0.70710678118654752440f;
  const float x = value * inv_sqrt2;
  const float sign = (x > 0.0f) - (x < 0.0f);
  const float a = std::fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * a);
  const float polynomial = (((((1.061405429f * t - 1.453152027f) * t
      + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
  const float erf = sign * (1.0f - polynomial * std::exp(-(a * a)));
  return 0.5f * value * (1.0f + erf);
}
```

Compile with `-ffp-contract=off` so multiply-add contraction cannot change
the quantized boundary. Return C-contiguous arrays with the same rank and
shape as the input.

- [ ] **Step 4: Bind the class without changing `DmaBatch`**

```cpp
py::class_<HostGraphExecutor>(module, "HostGraphExecutor")
    .def(py::init<>())
    .def("quantize", &HostGraphExecutor::quantize,
         py::arg("input").noconvert(), py::arg("scale"))
    .def("gelu_quantize", &HostGraphExecutor::gelu_quantize,
         py::arg("input").noconvert(), py::arg("scale"))
    .def("add", &HostGraphExecutor::add,
         py::arg("left").noconvert(), py::arg("right").noconvert())
    .def("add_quantize", &HostGraphExecutor::add_quantize,
         py::arg("left").noconvert(), py::arg("right").noconvert(),
         py::arg("scale"))
    .def("concatenate", &HostGraphExecutor::concatenate,
         py::arg("inputs"), py::arg("axis"))
    .def("stats", &HostGraphExecutor::stats)
    .def("reset_stats", &HostGraphExecutor::reset_stats);
```

Add `HostGraphExecutor` to the expected extension surface without removing or
renaming any existing symbol.

- [ ] **Step 5: Verify GREEN in a device-free test build and optimized build**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_u250_host_graph_executor.py
make -f tools/Makefile.u250_runtime dma-batch
FPGA_DMA_BATCH_SO="$(find build/native_codec -name 'fpgaDmaBatch*.so' -print -quit)" \
  PYTHONPATH=. pytest -q tests/test_fpga_dma_batch_api.py \
  tests/test_u250_host_graph_executor.py tests/test_u250_native_codec.py
```

Expected: both runs pass with no device access and no API regression. The
first test compiles a constructor-neutral fixture and instantiates only
`HostGraphExecutor`; the optimized build also verifies the production module.

- [ ] **Step 6: Commit the C++ executor**

```bash
git add tools/u250_host_graph.hpp tools/fpga_dma_batch.cpp \
  tools/Makefile.u250_runtime tests/test_fpga_dma_batch_api.py \
  tests/test_u250_host_graph_executor.py
git commit -m "feat: add device-free U250 host graph executor"
```

---

### Task 3: Qualify and Select the Host Executor

**Files:**
- Create: `tools/u250_host_executor.py`
- Create: `tools/qualify_u250_host_executor.py`
- Create: `tests/test_u250_host_executor_selection.py`
- Modify: `tools/u250_cpp_mapped_runtime.py:318-400,573-590`
- Modify: `tools/run_u250_depthanything_hybrid.py:393-580`
- Modify: `tools/run_u250_native_codec_controlflow.py:277-359`

**Interfaces:**
- Produces: `HostExecutorSelection(mode: str, report: Path | None, extension_path: Path, extension_sha256: str)`.
- Produces: `HostExecutorSelection.create(extension) -> PythonHostExecutor | CppHostExecutor`.
- Produces: report schema `u250-host-executor-qualification-v1` containing extension/source digests and exact vector-suite results.
- Adds CLI `--host-executor {python,auto,cpp}` and `--host-executor-report PATH`.

- [ ] **Step 1: Write failing fail-closed selection tests**

```python
def test_cpp_mode_requires_matching_complete_report(tmp_path, extension_identity):
    with pytest.raises(RuntimeError, match="qualification report"):
        HostExecutorSelection("cpp", None, *extension_identity)


def test_auto_falls_back_with_reason_for_digest_mismatch(tmp_path, report, extension_identity):
    report["extension_sha256"] = "0" * 64
    path = tmp_path / "host_executor.json"
    path.write_text(json.dumps(report))
    selection = HostExecutorSelection("auto", path, *extension_identity)
    assert selection.backend == "python"
    assert selection.fallback_reason == "extension SHA-256 mismatch"


def test_cpp_accepts_only_all_exact_cases(report_path, extension_identity):
    selection = HostExecutorSelection("cpp", report_path, *extension_identity)
    assert selection.backend == "cpp"
```

Mutation tests must reject a missing operation, non-exact result, source
digest mismatch, unknown schema, changed extension path, or replaced loaded
module.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_u250_host_executor_selection.py`

Expected: collection fails because `tools.u250_host_executor` is absent.

- [ ] **Step 3: Implement Python and C++ adapters**

```python
class PythonHostExecutor:
    backend = "python"
    def quantize(self, value, scale):
        return np.clip(np.rint(np.asarray(value, np.float32) / scale),
                       -128, 127).astype(np.int8)
    def gelu_quantize(self, value, scale):
        return self.quantize(gelu_reference(value), scale)
    def add(self, left, right):
        return (left + right).astype(np.float32)
    def add_quantize(self, left, right, scale):
        return self.quantize(self.add(left, right), scale)
    def concatenate(self, values, axis):
        return np.ascontiguousarray(np.concatenate(values, axis=axis))


class CppHostExecutor:
    backend = "cpp"
    def __init__(self, extension):
        self.native = extension.HostGraphExecutor()
```

The C++ adapter delegates the same five methods and normalizes all returned
arrays to the declared dtype without adding copies when already contiguous.

- [ ] **Step 4: Implement report validation and qualification generation**

The qualifier runs deterministic edge vectors plus real r43 tensors from
`demo05_holdout_trace_r52.npz`. It records SHA-256 for every Python and C++
result. An operation is `exact: true` only when dtype, shape, C-contiguity,
and `np.array_equal` all match. For `gelu_quantize`, qualification compares
the INT8 FC2 boundary, not unquantized GELU floats.

The report must include:

```json
{
  "schema": "u250-host-executor-qualification-v1",
  "qualified": true,
  "source_sha256": "64 lowercase hex digits",
  "extension_sha256": "64 lowercase hex digits",
  "operations": {
    "quantize": {"exact": true, "cases": 1},
    "gelu_quantize": {"exact": true, "cases": 1},
    "add": {"exact": true, "cases": 1},
    "add_quantize": {"exact": true, "cases": 1},
    "concatenate": {"exact": true, "cases": 1}
  }
}
```

Counts are the real number of executed cases and therefore may exceed one;
the validator requires each count to be a positive integer.

- [ ] **Step 5: Wire independent runtime selection**

Load the extension once through `load_fpga_dma_batch()`. Create the host
executor before `DmaBatch` construction and cache it by extension digest plus
qualification digest. Reset only per-frame counters between resident requests.
Expose in summary:

```python
"host_executor": {
    "requested": selection.mode,
    "backend": executor.backend,
    "qualification_sha256": selection.report_sha256,
    "extension_sha256": extension_sha256,
    "fallback_reason": selection.fallback_reason,
    **executor.stats(),
}
```

For the CPU control-flow worker, preserve `extension.HostGraphExecutor` on the
fake extension while replacing only `DmaBatch`:

```python
fake_extension = SimpleNamespace(
    DmaBatch=ZeroOutputDma,
    HostGraphExecutor=extension.HostGraphExecutor,
    __file__=extension.__file__,
    _u250_extension_sha256=extension._u250_extension_sha256,
)
```

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_u250_host_executor_selection.py \
  tests/test_u250_cpp_mapped_runtime.py tests/test_u250_native_controlflow.py
```

Expected: all tests pass; `python` requires no report, `auto` reports an
explicit fallback, and `cpp` rejects every incomplete or mismatched report.

- [ ] **Step 7: Commit selection and qualification**

```bash
git add tools/u250_host_executor.py tools/qualify_u250_host_executor.py \
  tools/u250_cpp_mapped_runtime.py tools/run_u250_depthanything_hybrid.py \
  tools/run_u250_native_codec_controlflow.py \
  tests/test_u250_host_executor_selection.py tests/test_u250_cpp_mapped_runtime.py \
  tests/test_u250_native_controlflow.py
git commit -m "feat: qualify U250 host executor selection"
```

---

### Task 4: Replace Hot Python Boundaries

**Files:**
- Modify: `tools/run_u250_depthanything_hybrid.py:730-970`
- Modify: `tools/u250_host_executor.py`
- Create: `tests/test_u250_hybrid_host_boundaries.py`
- Modify: `tests/test_u250_native_controlflow.py`

**Interfaces:**
- Consumes the selected host executor from Task 3.
- Produces identical QKV codes, FC1 codes, FC2 codes, residual tensors,
  attention head assemblies, and decoder tile assemblies.
- Produces per-operation backend/call counters that reconcile with
  `host_profile`.

- [ ] **Step 1: Write failing boundary-equivalence tests**

Use small deterministic arrays plus extracted demo05 tensors. Exercise the
same call shapes as the runner:

```python
@pytest.mark.parametrize("shape,scale", [
    ((1, 1370, 384), 0.10580708831548691),
    ((1, 1370, 1536), 0.03993530943989754),
])
def test_encoder_quantized_boundaries_are_exact(executors, shape, scale):
    value = deterministic_fp32(shape)
    assert np.array_equal(
        executors.cpp.quantize(value, scale),
        executors.python.quantize(value, scale),
    )


def test_gelu_fc2_boundary_is_exact(executors):
    hidden = deterministic_fp32((1, 1, 1370, 1536))
    assert np.array_equal(
        executors.cpp.gelu_quantize(hidden, 0.03993530943989754),
        executors.python.gelu_quantize(hidden, 0.03993530943989754),
    )
```

Test all 12 real QKV/FC1/FC2 scales from the runtime contract. Test attention
assembly with six heads and three query pieces, and decoder row tiles for
first/middle/last halo cases.

- [ ] **Step 2: Verify RED at the integration boundary**

Run: `PYTHONPATH=. pytest -q tests/test_u250_hybrid_host_boundaries.py`

Expected: tests fail because the runner still calls module-level NumPy helpers
instead of the selected executor and exposes no boundary helpers.

- [ ] **Step 3: Route quantization and fused GELU boundary through the executor**

Replace direct `quantize()` calls with `host_executor.quantize()`. In the
current `host_activation` branch replace:

```python
activated = gelu(hidden)
fc2_input = quantize(activated, scale)
```

with:

```python
fc2_input = host_executor.gelu_quantize(hidden, scale)
if args.depth_only:
    activated = None
else:
    activated = gelu(hidden)
```

This avoids constructing the large FP32 GELU result during production depth
inference while preserving trace capture behavior. The detailed profile must
record `encoder.gelu_quantize`; it must not label FC1 NPU time as GELU.

- [ ] **Step 4: Route residual and concatenation boundaries**

Use `host_executor.add()` for encoder residuals and
`host_executor.concatenate()` for patch projection, attention heads/query
chunks, FC1 chunks, channel slices, and decoder tiles. Do not replace view-only
reshape/transpose operations. Preserve `np.ascontiguousarray` only at ABI
boundaries that require it.

- [ ] **Step 5: Add counter and fallback assertions**

The full CPU control-flow summary must show:

```python
assert summary["host_executor"]["backend"] == "cpp"
assert summary["host_executor"]["fallback_reason"] is None
assert summary["host_executor"]["gelu_quantize_calls"] == 12
assert summary["npu_calls"] == 443
assert summary["submission_groups"] == 248
assert summary["vendor_pack_calls"] == 0
assert summary["vendor_unpack_calls"] == 0
```

Record actual quantize/add/concatenate counts and pin them in the generated
r61 CPU qualification rather than copying guessed values into the validator.

- [ ] **Step 6: Verify GREEN and full local regression**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_u250_hybrid_host_boundaries.py \
  tests/test_u250_host_profile.py tests/test_u250_host_executor_selection.py \
  tests/test_u250_native_controlflow.py
PYTHONPATH=. pytest -q tests/test_fpga_dma_batch_api.py \
  tests/test_u250_codec_selection.py tests/test_u250_cpp_mapped_runtime.py \
  tests/test_u250_layout_descriptors.py tests/test_u250_native_board_gate.py \
  tests/test_u250_native_codec.py tests/test_u250_native_controlflow.py \
  tests/test_u250_host_profile.py tests/test_u250_host_graph_executor.py \
  tests/test_u250_host_executor_selection.py tests/test_u250_hybrid_host_boundaries.py
```

Expected: all non-environment-dependent tests pass; extension-dependent tests
run when `FPGA_DMA_BATCH_SO` is supplied and otherwise report explicit skips.

- [ ] **Step 7: Commit integration**

```bash
git add tools/run_u250_depthanything_hybrid.py tools/u250_host_executor.py \
  tests/test_u250_hybrid_host_boundaries.py tests/test_u250_native_controlflow.py
git commit -m "perf: execute DepthAnything host boundaries in C++"
```

---

### Task 5: CPU-Only r61 Qualification

**Files:**
- Modify: `tools/run_u250_native_codec_controlflow.py`
- Modify: `tools/run_u250_mapped_r58_gate.sh`
- Modify: `tools/check_u250_native_board_gate.py`
- Create: `artifacts/u250_host_graph_r61/host_executor_qualification.json`
- Create: `artifacts/u250_host_graph_r61/full_controlflow.summary.json`
- Create: `artifacts/u250_host_graph_r61/controlflow_input_inventory.json`
- Modify: `docs/U250_MAPPED_RUNTIME_R58.md`

**Interfaces:**
- Consumes the DS Python 3.13 extension built from current source.
- Produces a fail-closed, CPU-only r61 evidence set with zero device/lock opens.
- Produces an immutable source/input/extension inventory for board deployment.

- [ ] **Step 1: Write failing r61 gate tests**

Add fixtures requiring schema 2 host profiling and the qualified C++ backend:

```python
def test_r61_gate_rejects_python_host_executor():
    summary = qualified_r61_summary()
    summary["host_executor"]["backend"] = "python"
    with pytest.raises(AssertionError, match="host executor"):
        controlflow().assert_native_controlflow(summary)


def test_r61_gate_rejects_unreconciled_host_timing():
    summary = qualified_r61_summary()
    summary["latency_breakdown"]["unattributed_host_residual_ms"] += 2.0
    with pytest.raises(AssertionError, match="reconcile"):
        controlflow().assert_native_controlflow(summary)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_u250_native_controlflow.py tests/test_u250_native_board_gate.py`

Expected: new r61 tests fail because the gates do not require host-executor
evidence.

- [ ] **Step 3: Extend the gates and deployment inventory**

Require source hashes for `u250_host_profile.py`, `u250_host_executor.py`, and
`u250_host_graph.hpp`; require the qualification report digest and loaded
extension digest to agree. The CPU trace must still prove zero XDMA, UIO, and
runtime-lock opens and zero writes into the source package or runtime tree.

- [ ] **Step 4: Build and qualify on the U250 host without taking the board lock**

Create a fresh package at:

```text
/home/visitor/Documents/depthanything_u250_host_graph_r61
```

Copy the unchanged r60 bank/cfg/inputs plus current sources, build with:

```bash
/home/visitor/anaconda3/envs/ds/bin/python -m pybind11 --includes
make -f tools/Makefile.u250_runtime dma-batch \
  PYTHON=/home/visitor/anaconda3/envs/ds/bin/python
```

Run the host qualifier, then the CPU control-flow worker using the real native
codec and `HostGraphExecutor` but the zero-output DMA transport. Expected:
443 dispatches, 248 groups, 1103 native packs, 683 native unpacks, zero vendor
calls, zero device opens, and 12 C++ `gelu_quantize` calls.

- [ ] **Step 5: Pin evidence and re-run fail-closed checks**

Generate the deployment inventory only after all files are final. Rehash every
entry independently, then run:

```bash
/home/visitor/anaconda3/envs/ds/bin/python \
  tools/run_u250_native_codec_controlflow.py \
  --check-summary artifacts/u250_host_graph_r61/full_controlflow.summary.json \
  --layout-codec-report artifacts/u250_native_codec/all_oracle.json
```

Expected: `native CPU control-flow gate passed` and no package mutation after
inventory generation.

- [ ] **Step 6: Commit CPU qualification**

```bash
git add tools/run_u250_native_codec_controlflow.py \
  tools/run_u250_mapped_r58_gate.sh tools/check_u250_native_board_gate.py \
  artifacts/u250_host_graph_r61 docs/U250_MAPPED_RUNTIME_R58.md
git commit -m "test: qualify r61 U250 host graph execution"
```

---

### Task 6: U250 Board Qualification and Stage 2 Decision

**Files:**
- Create: `artifacts/u250_host_graph_r61/demo05_decoder_only.summary.json`
- Create: `artifacts/u250_host_graph_r61/demo05_resume_l11.summary.json`
- Create: `artifacts/u250_host_graph_r61/demo05_full_first.summary.json`
- Create: `artifacts/u250_host_graph_r61/demo05_full_resident.summary.json`
- Create: `artifacts/u250_host_graph_r61/gate_summary.json`
- Create: `artifacts/u250_host_graph_r61/host_profile_ranked.json`
- Modify: `docs/U250_MAPPED_RUNTIME_R58.md`

**Interfaces:**
- Produces four safe-DMA board summaries and one aggregate r61 gate.
- Produces a ranked Stage 2 input containing operation time, calls, bytes,
  percentage of process wall, and observed improvement versus r60.

- [ ] **Step 1: Write failing board-gate tests for r61 evidence**

```python
def test_r61_full_resident_requires_exact_output_and_cpp_host_executor(r61_frame):
    check_frame("demo05_full_resident", r61_frame)
    assert r61_frame["output_sha256"] == EXPECTED
    assert r61_frame["host_executor"]["backend"] == "cpp"
    assert r61_frame["host_executor"]["gelu_quantize_calls"] == 12


def test_ranked_profile_is_descending_and_reconciled(ranked_profile):
    values = [item["ms"] for item in ranked_profile["operations"]]
    assert values == sorted(values, reverse=True)
    assert abs(ranked_profile["process_wall_ms"]
               - ranked_profile["reconciled_ms"]) <= 0.5
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_u250_native_board_gate.py`

Expected: the r61 fixture fails until new board evidence exists and validates.

- [ ] **Step 3: Acquire the board lock and verify the deployed package**

On U250, run the inventory verifier before opening a device, then acquire:

```bash
flock -n /tmp/ds-u250-runtime.lock \
  bash tools/run_u250_mapped_r58_gate.sh
```

If `flock` cannot acquire the lock, exit 75 and leave the package and retained
evidence untouched.

- [ ] **Step 4: Run four board modes into a fresh directory**

Execute decoder-only, layer-11 resume, full first frame, and a second full
frame in the same resident PID with `--layout-codec native`,
`--host-executor cpp`, safe DMA, attention launch group 3, and decoder launch
group 32. Do not use persistent DMA descriptors.

Each result must report:

```text
output_sha256 = 2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725
relative_l2 = 0
rmse = 0
stale_events = 0
vendor_pack_calls = 0
vendor_unpack_calls = 0
```

The resident frame must additionally report `resident_bank_reused=true`,
`cpp_runtime_reused=true`, `codec_yaml_reused=true`, and `load_ms=0`.

- [ ] **Step 5: Compare repeated resident measurements**

After the correctness gate passes, run five additional same-process frames.
Record every sample and report median, minimum, maximum, and median absolute
deviation. Enable C++ host execution by default only when its median process
wall is below the r60 Python-host median measured in the same session. Do not
claim a distribution from the historical single r60 sample.

- [ ] **Step 6: Generate the ranked Stage 2 input**

Sort explicit host-profile families by median time. Include decoder operator
timings separately and identify Resize as a Stage 2 target only if it remains
the largest decoder host operator. The output schema is:

```json
{
  "schema": "u250-host-profile-ranked-v1",
  "process_wall_ms": 1.0,
  "reconciled_ms": 1.0,
  "operations": [
    {"name": "encoder.gelu_quantize", "calls": 12,
     "ms": 1.0, "percent": 1.0, "bytes": 1}
  ]
}
```

All numeric values are populated from the measured resident median, not from
the illustrative values above.

- [ ] **Step 7: Verify retained evidence and full regression**

Run:

```bash
PYTHONPATH=. python tools/check_u250_native_board_gate.py \
  --run-dir artifacts/u250_host_graph_r61
PYTHONPATH=. pytest -q tests/test_fpga_dma_batch_api.py \
  tests/test_u250_codec_selection.py tests/test_u250_cpp_mapped_runtime.py \
  tests/test_u250_layout_descriptors.py tests/test_u250_native_board_gate.py \
  tests/test_u250_native_codec.py tests/test_u250_native_controlflow.py \
  tests/test_u250_host_profile.py tests/test_u250_host_graph_executor.py \
  tests/test_u250_host_executor_selection.py tests/test_u250_hybrid_host_boundaries.py
git diff --check
```

Expected: the gate and all available tests pass, the board evidence is
immutable after collection, and `git diff --check` prints nothing.

- [ ] **Step 8: Document and commit Stage 1**

Document the corrected CPU/NPU partition, per-family host profile, repeated
latency samples, exact output result, extension/source/inventory hashes, and
the measured go/no-go decision for Stage 2.

```bash
git add artifacts/u250_host_graph_r61 docs/U250_MAPPED_RUNTIME_R58.md \
  tests/test_u250_native_board_gate.py
git commit -m "perf: qualify DepthAnything U250 host graph r61"
```

---

## Stage Boundary

Do not begin decoder Resize implementation until Task 6 produces
`host_profile_ranked.json`. Stage 2 will use the five measured Resize shapes,
their real tensor ranges, and their measured share of resident wall time to
define exact C++ interpolation tests and an independent DS-Compiler
feasibility case. Stages 3 and 4 remain gated on the transfer and dispatch
profile produced after Stage 2 so their address-lifetime and fusion plans are
based on the retained runtime rather than the r60 baseline.
