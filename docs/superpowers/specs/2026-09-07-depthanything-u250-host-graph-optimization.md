# DepthAnything U250 Host Graph Optimization Design

## Objective

Reduce the steady-state U250 latency of the qualified DepthAnything-V2 r60
deployment without changing its numerical graph, golden depth output, resident
DDR weight bank, or safe-DMA behavior. Optimization proceeds in four ordered
stages: measured host execution, decoder resize acceleration, physical tensor
residency, and finally DS-Compiler launch-boundary fusion.

The committed r60 board result is the baseline:

- process wall: 1,621.344 ms;
- NPU: 441.203 ms;
- uninstrumented host graph and Python residual: 424.911 ms;
- native input pack: 244.830 ms;
- decoder host operators: 237.586 ms;
- native output unpack: 128.016 ms;
- H2C plus C2H: 144.798 ms;
- physical dispatches / C++ submission groups: 443 / 248;
- depth SHA-256: `2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.

## Corrected Runtime Partition

The r60 runtime-contract hash is
`a66dfd1ed6645d19569eb7158a0de0642c63cb4fdee8fb0aaa4a8f38a2236d86`.
It assigns patch projection, encoder LayerNorm cores, QKV projection,
attention, post-attention projection, MLP FC1, MLP FC2, and decoder
convolutions to NPU BINs.

MLP GELU is not currently an NPU operator. Every encoder block declares
`host_activation: {op: GELU, precision: FP32}`. The existing latency stage
name `encoder_mlp_fc1_gelu` describes the region around FC1; it must not be
interpreted as proof that GELU is in EPU. The optimized runtime must report
host GELU independently until an actual compiler-qualified EPU activation is
selected by the runtime contract.

The decoder host plan contains 102 host steps and 32 logical convolution
steps. Channel and row slicing expand the latter to 89 physical NPU
dispatches. The host steps are:

| Operator | Calls | r60 steady ms |
|---|---:|---:|
| Resize | 5 | 204.854 |
| ReLU | 17 | 14.294 |
| Add | 10 | 6.815 |
| LayerNormalization | 4 | 4.842 |
| DepthToSpace | 3 | 4.589 |
| Transpose | 4 | 1.864 |
| Slice | 8 | 0.155 |
| Constant | 38 | 0.036 |
| Reshape | 4 | 0.047 |
| Shape | 4 | 0.047 |
| Concat | 4 | 0.021 |
| Squeeze | 1 | 0.020 |

These placements are qualified-runtime decisions rather than a claim that
all listed operators are absent from the NPU. Metadata and view operations do
not justify a separate hardware launch. ReLU and Add require fusion with their
producer while preserving scale and layout. DepthToSpace and LayerNorm are
small enough that a standalone DMA/launch boundary may cost more than their
host implementation. Bilinear Resize with ONNX `align_corners` semantics is
the only dominant decoder host operator and currently has no selected,
board-qualified NPU equivalent.

## Measurement Model

`host_graph_and_python_residual_ms` is a subtraction-derived bucket, not a
Python-interpreter timer. It currently includes uninstrumented NumPy compute,
allocation, copies, and orchestration such as:

- input patchification, transpose, reshape, CLS concatenation and position
  embedding addition;
- encoder quantization, head/query slicing, grouped-call construction,
  attention/head concatenation, residual additions, FP32 GELU, and FC1/FC2
  tensor transformations;
- decoder row/channel tile construction and concatenation;
- Python dictionary tensor lookup, temporary-array creation, result assembly,
  hashing, and output serialization.

Before optimizing this bucket, the runner will record explicit timers for
each material family. The sum of explicit categories plus a final unexplained
residual must equal `process_wall_ms` within normal timer precision. Reports
will continue to distinguish NPU dispatch time, DMA, codec conversion, host
operators, and orchestration.

## Design

### Stage 1: Preparsed C++ host graph execution

First add fine-grained timers without changing computation. Timers cover
frontend transforms, quantization, GELU, residual/elementwise work,
head/chunk assembly, decoder tile assembly, and result serialization. A CPU
control-flow run and a board resident run determine the ranked targets.

Extend the existing pybind C++ runtime with a persistent host workspace. It
owns reusable, aligned arrays keyed by shape and dtype and exposes fused host
primitives for the measured hot paths. The first retained primitives are:

- symmetric FP32-to-INT8 quantization with the existing `rint`, saturation,
  and cast semantics;
- exact FP32 GELU using the same approximation and operation order as r60;
- residual add followed by optional quantization;
- attention head/query gather and output concatenation;
- decoder row/channel tile gather and concatenation.

The runtime parses an immutable execution schedule once per resident server
process. Python remains responsible for request handling and error reporting,
but invokes coarse C++ graph segments instead of constructing hundreds of
temporary lists and arrays. Each primitive retains an independently callable
Python fallback for bisecting and qualification.

Stage 1 does not alter DS BINs, card addresses, DMA safety, codec descriptors,
or model scales. An operation may switch to C++ only after exact array
comparison against the r60 implementation for representative and boundary
inputs.

### Stage 2: Exact decoder Resize acceleration

Implement a persistent resize plan for each of the five static decoder
shapes. The plan precomputes `y0`, `y1`, `x0`, `x1`, and FP32 interpolation
weights once. A C++ NCHW kernel performs separable bilinear interpolation with
ONNX `coordinate_transformation_mode=align_corners`. Output buffers come from
the Stage 1 workspace.

The C++ path is selected only when mode, rank, shape, and coordinate semantics
match the qualified plan. Any mismatch fails closed to the existing Python
implementation in development mode and is rejected in strict production
mode. Exact or explicitly bounded comparison is performed at all five nodes;
the complete final depth output must still match the r60 golden SHA.

In parallel, a separate DS-Compiler feasibility case may express each static
resize as a supported NPU sequence. It cannot replace C++ Resize unless it
passes compiler generation, instruction-level simulation, board execution,
intermediate numerical comparison, and end-to-end latency comparison. This
experiment must not require a new bitstream for Stages 1 or 2.

ReLU is fused into a producer Conv only when the generated BIN is qualified.
Add, DepthToSpace, and LayerNorm remain on host during Stage 2 because their
combined measured time is small relative to Resize.

### Stage 3: Physical tensor residency

Introduce a `DeviceTensorHandle` describing bank addresses, physical extent,
logical descriptor identity, dtype, scale, owner group, and lifetime. When a
producer output and consumer input have an identical qualified physical ABI,
the C++ executor passes the handle directly and suppresses C2H, unpack, pack,
and H2C. Incompatible layouts, required CPU consumers, or overlapping FM
lifetimes force the existing materialization path.

The first candidate chains are:

1. QKV projection to attention where head extraction can be represented by
   qualified physical offsets;
2. attention output to post-attention projection;
3. FC1 output through a compiler-qualified EPU GELU to FC2;
4. adjacent decoder Conv/ReLU/Conv regions that do not require a host skip
   merge.

An offline allocator validates every address interval against the two-bank
8 MiB shared FM workspace. The runtime rejects aliasing, stale handles,
descriptor mismatch, or a scale mismatch before device access. Every skipped
transfer is recorded by tensor boundary and byte count.

### Stage 4: DS-Compiler program-chain fusion

After Stages 1–3 establish which remaining boundaries are expensive, reduce
physical BIN fragmentation without changing the bitstream unless an explicit
hardware requirement is proven.

Compiler work proceeds in this order:

1. combine the three query slices of one attention head into a single
   scheduled program chain;
2. retain K/V and intermediate attention storage across compatible head
   work;
3. evaluate local QKV-to-attention-to-post-projection chaining;
4. combine decoder row tiles and channel slices within FM capacity;
5. fuse decoder Conv with immediately adjacent ReLU and compatible Add.

Each fusion candidate must pass compiler, simulator, board DMA/event, stale
event, address-overlap, intermediate precision, and final-depth gates. A
failed candidate is removed independently; it does not block the already
qualified earlier stages.

## Error Handling and Rollback

- Production mode rejects missing or mismatched extension, schedule,
  descriptor, manifest, or qualification digests before opening a device.
- Host primitive failures identify the exact operator, shape, dtype, and
  schedule index.
- Device-resident chaining rejects unqualified ABI transitions rather than
  silently converting them.
- Each stage has an explicit runtime switch so the previous qualified stage
  can be selected without rebuilding the bank.
- New board evidence is written to a fresh versioned directory; r60 evidence
  and deployment files are never overwritten.

## Verification and Acceptance

Every retained stage must pass:

1. unit tests for the C++/Python primitive API, error cases, descriptor
   lifetimes, and timing accounting;
2. red/green equivalence tests against the existing NumPy implementation;
3. the complete 443-dispatch CPU control-flow gate with zero protected device
   opens;
4. decoder-only, layer-11 resume, full first-frame, and full resident-frame
   U250 gates;
5. output SHA-256
   `2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`,
   `relative_l2=0`, `rmse=0`, no timeout, and zero stale events;
6. resident DDR bank reuse with `load_ms=0`, native codec use, and zero vendor
   pack/unpack calls;
7. a measured resident-frame latency improvement on repeated same-process
   requests. A stage with no repeatable improvement is not enabled by default.

Latency reports must include per-host-family timings, allocation/copy counts,
physical dispatches, C++ groups, direct device-tensor handoffs, skipped bytes,
and both process and inference wall intervals. Single-run measurements are
reported as samples rather than latency distributions.

## Non-goals

- changing model architecture or recalibrating scales during Stages 1–2;
- weakening safe-DMA, digest, address, event, or board-lock checks;
- replacing exact r60 depth parity with an unapproved accuracy tolerance;
- forcing metadata-only decoder operations onto NPU;
- overwriting any prior deployment or qualification evidence;
- force-pushing an unrelated remote `main` history.
