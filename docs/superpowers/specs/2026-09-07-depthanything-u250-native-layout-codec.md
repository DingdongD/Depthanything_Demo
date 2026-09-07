# DepthAnything U250 Native Layout Codec Design

## Purpose

Replace the vendor `npz2bin` tensor pack/unpack hot path in the r58 U250
runtime with native C++ codecs while preserving the existing r43 numerical
graph, compiler BINs, resident DDR image, and final-depth result bit for bit.

The measured r58 steady frame is 14,202.047 ms. Input packing consumes
7,497.623 ms and output unpacking consumes 5,343.998 ms, together accounting
for 90.42% of the frame. The NPU, H2C, and C2H paths are already below one
second combined, so this work is limited to physical tensor layout conversion
and its runtime selection.

## Scope

The native path covers every input/output tensor ABI used by the 262-case r43
resident bank:

- NDWC INT8 pack and unpack;
- NDWC BF16 pack and unpack;
- NCHW INT8 pack and unpack;
- NCHW BF16 pack and unpack.

The current graph primarily needs NDWC INT8 input packing, NDWC INT8/BF16
output unpacking, NCHW INT8 input packing, and NCHW BF16 output unpacking.
Both directions are included for symmetry and fixture generation, but an
untested direction cannot be selected in production.

This design does not change quantization scales, BF16 rounding, operators,
hardware dispatch count, instruction ranges, weights, or DDR-resident model
contents. Attention and decoder grouping remain host submission/DMA fusion;
the hardware continues to execute 443 independent compiler-qualified BINs.

## Selected Approach

Implement formula-driven codecs in the existing Python 3.13 `fpgaDmaBatch`
C++ extension. Each call receives a compact layout descriptor derived from
the cfg tensor record:

```text
layout: NCHW | NDWC
dims: [N, C-or-D, H-or-W, W-or-C]
bitdepth: 8 | 16
c_align: positive integer
w_align: positive integer
combined_bytes: exact two-bank physical extent
```

The implementation uses the DS architecture's documented NetIO/NetIOMM and
FM/MM tiling rules. It computes physical bank, address, and lane directly;
BF16 conversion uses IEEE BF16 round-to-nearest-even semantics matching the
existing PyTorch callback. The C++ API returns separate even/odd bank arrays,
so grouped DMA no longer needs Python combined-buffer split/merge operations.

The alternatives are rejected as follows:

- Full logical-to-physical index maps require large resident tables and
  indirect memory access for fixed mappings that have compact formulas.
- Thirty shape-specific kernels would be fast but couple runtime code to this
  exact manifest and would need source changes for every new geometry.

## Descriptor Generation and Validation

`CfgCodecRegistry` continues to pre-parse cfg files. It is extended to retain
all tensor fields needed by the native descriptor and to assign a stable
descriptor identity independent of relocated addresses.

At package validation time, every unique input and output descriptor is
checked against the official vendor codec with deterministic tensors chosen
to expose permutation, padding, signed INT8, and BF16 rounding errors:

- monotonic coordinate patterns across all logical axes;
- alternating negative and positive INT8 values, including -128 and 127;
- finite FP32 values around BF16 halfway boundaries and normal/subnormal
  boundaries;
- non-aligned logical tails in C and W;
- zero-filled physical padding verification.

Pack validation compares the complete combined physical byte buffer. Unpack
validation compares logical dtype, shape, and every value. Each descriptor is
marked `native_exact` only after both supported directions pass. Validation
produces a JSON report with descriptor, tensor users, byte counts, hashes,
and mismatch location when rejected.

## Runtime Selection and Fail-Closed Behavior

The production runner adds three codec modes:

- `vendor`: use only `npz2bin` for diagnostics;
- `native`: require every active boundary to be `native_exact`, otherwise
  fail before loading or accessing the board;
- `auto`: select native only for validated descriptors and record every
  fallback.

Board qualification uses `native`, not `auto`, so a successful full-frame
gate proves all 443 calls avoided the vendor pack/unpack path. `auto` exists
only to support incremental development and diagnosis.

Runtime counters report native/vendor pack and unpack calls, logical and
physical bytes, elapsed time by layout/dtype, and fallback reasons. The final
summary must distinguish codec time from DMA and NPU time and must assert zero
vendor calls for the production gate.

## C++ API Boundary

The extension adds focused methods rather than embedding model scheduling:

```text
pack_tensor(array, descriptor) -> (even_uint8, odd_uint8)
unpack_tensor(even_uint8, odd_uint8, descriptor) -> logical_array
validate_descriptor(descriptor) -> normalized descriptor metadata
```

Packing accepts contiguous NumPy INT8 or FP32 arrays. A bitdepth-16 pack
converts FP32 to BF16 inside C++. Unpacking returns INT8 for bitdepth 8 and
FP32 for bitdepth 16, matching the current runner contract. Unsupported
layouts, dimensions, byte extents, and dtypes raise descriptive exceptions
before touching a DMA device.

The existing `pack_int8_nchw` methods remain available to CompletionFormer.
The new APIs are additive and must not change their output or performance
counters.

## Data Flow

For each grouped call, the runner selects one validated descriptor per input,
packs directly to two bank arrays in C++, and submits those arrays to the
existing C++ batched H2C transport. After the NPU chain, C2H returns the two
exact bank arrays; C++ unpack produces the logical tensor directly. Python no
longer constructs a combined physical buffer for native boundaries.

The model schedule, host LayerNorm affine work, GELU representation, decoder
host operators, attention concatenation, and final `.npz` output remain
unchanged.

## Error Handling

Descriptor validation rejects:

- unsupported layout or bitdepth;
- dimensions other than rank four or non-positive dimensions;
- a combined extent that is not 256-byte aligned;
- an extent smaller than the formula's required padded storage;
- input dtype incompatible with bitdepth;
- bank buffers with unequal or incorrect exact sizes;
- any native/vendor oracle mismatch.

`native` mode aborts before resident bank upload if the manifest validation
report is incomplete, stale, or does not match the manifest SHA-256. Runtime
exceptions include case name, tensor direction/index, and descriptor identity.

## Testing and Qualification

Development follows test-driven increments:

1. Descriptor parser and validation tests fail before implementation and then
   pass for all manifest shapes and malformed descriptors.
2. NCHW INT8 native pack/unpack is compared byte for byte with existing
   CompletionFormer and vendor fixtures.
3. NCHW BF16 and NDWC INT8/BF16 codecs are each gated against official vendor
   fixtures for all unique descriptors.
4. Runner selection tests prove `native` fails closed, `auto` records
   fallback, and native paths bypass combined split/merge and vendor calls.
5. A CPU-only 443-call control-flow run requires zero vendor codec calls.
6. The non-blocking U250 gate runs decoder-only, layer-11 resume, a complete
   first frame, and a second same-process complete frame.

Every board output must equal the r43 reference SHA-256
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.
The second full frame must report `resident_bank_reused=true`, `load_ms=0`,
zero vendor codec calls, 443 hardware dispatches, and 248 C++ submission
groups. No CompletionFormer or unrelated board process may be interrupted;
the gate exits without device access when `/tmp/ds-u250-runtime.lock` is busy.

## Performance Reporting

Report measured first-frame and steady-frame wall latency and this breakdown:

- cfg/YAML initialization;
- native input pack;
- vendor input pack fallback;
- H2C;
- NPU;
- C2H;
- native output unpack;
- vendor output unpack fallback;
- decoder host operators;
- residual Python/graph work.

Compare against both the 60,561.193 ms legacy baseline and the 14,202.047 ms
r58 mapped-runtime steady baseline. No projected value is labeled as measured.
If native codec performance regresses for a descriptor, retain correctness,
record that result, and leave the descriptor disabled until its implementation
is faster than the vendor path on the U250 host.
