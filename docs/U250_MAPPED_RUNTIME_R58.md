# DepthAnything U250 mapped resident runtime r58

## Implemented runtime path

The r43 numerical graph and its 262 compiler-qualified BINs are unchanged.
The new runtime pre-parses all cfg tensor layouts into 30 ABI signatures,
loads the 42,340,352-byte bank only once per process, and reuses the
CompletionFormer-qualified `fpgaDmaBatch` C++ bridge.  The bridge retains its
mapped BAR, event descriptors, and page-locked DMA buffers across requests.

Attention's three query-pair calls per head use disjoint FM slots.  Their
inputs are sent by one batched H2C operation, the three existing BINs are
submitted by one C++ `run_npu_chain` call, and all outputs are returned by one
batched C2H operation.  Decoder row tiles and channel slices use the same
scheme, grouped by exact codec ABI and bounded by the 8 MiB per-bank shared FM
workspace.  This is host submission/DMA fusion, not unsupported instruction
fusion: the hardware still executes 443 separately qualified BINs.

`depthanything_u250_resident_server.py` accepts JSON-lines requests and calls
the runner repeatedly in one PID.  On request two and later, the C++ runtime,
mapped BAR, pinned buffers, cfg registry, and resident bank are reused.

## Corrected C2H extent

The cfg `Size` field describes the combined two-bank physical tensor.  The
legacy runtime incorrectly read that full byte count from each bank, merged a
2x buffer, and relied on `npz2bin` ignoring the tail.  An offline vendor-codec
gate confirmed that the exact combined output is 32,768 bytes for an
attention output (`outputSize` is 16,384 bytes per bank), while the old merged
buffer was 65,536 bytes; both decoded identically because the tail is ignored.

Across one complete frame, exact-span C2H changes 421,942,272 bytes to
210,971,136 bytes, a 50% reduction.  Relative to the recorded demo05 r43
baseline C2H time of 191.526 ms, 95.763 ms is a conservative byte-linear
projection; it is not reported as a board measurement.

## Static launch analysis

The NPU dispatch count remains 443.  C++ submission groups fall to 248, a
44.018% reduction in Python-to-C++ launch transactions.  Attention changes
from 216 dispatch-level submissions to 72 head groups.  Decoder changes from
89 dispatches to 38 groups.  The remaining 138 encoder/front-end calls retain
their existing one-BIN boundaries.

The existing full-frame demo05 baseline is 60,561.193 ms wall, 447.733 ms NPU,
428.594 ms H2C, and 191.526 ms C2H.  The new report separates cfg preparse,
vendor cfg activation, input pack, H2C, NPU, C2H, output unpack, and residual
host graph/Python time, and records both physical dispatches and C++ groups.

A complete 443-call CPU control-flow gate was also executed on the U250 host
with NPU/DMA replaced by a zero-output transport.  It exercises all real
vendor cfg pack/unpack calls and every host graph operation without accessing
the locked card.  It completed in 10,266.248 ms: input pack 5,839.995 ms,
output unpack 3,859.465 ms, decoder host operators 225.094 ms, cfg activation
9.318 ms (107 activations), and residual Python/graph work 383.332 ms.  This
confirms that cfg parsing is not the remaining bottleneck; physical layout
conversion now accounts for 94.48% of CPU-only runtime.  It also confirms
that disabling calibration/trace collection in production removes the former
dominant percentile/copy overhead.  These are host-only measurements, not an
end-to-end board latency claim.

## C++ host graph r61 CPU qualification (2026-09-08)

The r61 package at
`/home/visitor/Documents/depthanything_u250_host_graph_r61` uses the qualified
C++ `HostGraphExecutor` for FP32/INT8 quantize, GELU-to-INT8, residual add, and
concatenation boundaries. Shape and metadata operations with integer dtypes
remain on NumPy so they cannot be misrouted through the tensor executor.

The complete CPU-only control-flow qualification passed against the real r52
trace and the DS Python 3.13 extension. It exercised 443 fake NPU dispatches in
248 groups, 1103 native packs, 683 native unpacks, 218 C++ host operations,
and all 12 encoder GELU-to-INT8 boundaries. Vendor codec calls, device opens,
runtime-lock opens, and protected package/runtime writes were all zero. The
process took 1,358.006 ms, including 545.254 ms of native layout conversion;
this is a host-only structural measurement and not a U250 latency result.

The retained evidence is portable: callers can explicitly provide the layout
oracle, host-executor qualification report, and canonical input inventory,
without relying on paths from the remote invocation. The r61 deployment
inventory contains 293 independently rehashed files and passed the CPU-only
package verifier before any board lock or device access.

Reproducibility hashes for this CPU gate:

- DS extension: `fd5d8d8059cda7bb4dd9ac2f9bbd60b53880b0fc319b39da6455c71a2c4a9702`.
- Host-executor report: `1bcabdfc191a0e301abd2f1069bae1ee3b69423350421c6f4dd7258173a1a391`.
- Native ALL oracle: `f7b83e012fe0d93f706e8b0eee64dcfe3a62a4bb6be0f36186864153db04979e`.
- Canonical r61 input inventory: `820483131391f4acc5842ea632c15bab94b3f026c4e1934797072ec0096745aa`.
- CPU control-flow summary: `360d9ad160cf3746aa59cae409d05d07e4262819c8c8817d4fc54dc64daa1813`.
- Deployment inventory: `54d0333aa43c0b90d3a4e16e915747ea0ace425ce7b10996a30b45a515fb8104`.

## C++ host graph r61 board qualification (2026-09-08)

The first r61 full-frame attempt exposed a real-input qualification gap in the
scalar NumPy-compatible `expf` transcription: negative GELU inputs from
`-16.875` to `-13.3125` entered the FP32 subnormal exponent range, where the
normal-only bit construction wrapped and produced saturated INT8 `127`
instead of `0`. Decoder-only and layer-11 resume remained exact, which isolated
the fault to earlier encoder GELU boundaries rather than the bitstream, BINs,
DDR bank, or DMA. The failed run was retained on the board as
`board_r61_gate_failed_pre_expfix_20260908`.

The fix returns zero below the normal `expf` range, where the exponential can
no longer affect FP32 GELU, and extends qualification across the observed
large-negative band. A diagnostic full frame compared every one of the 218
C++ host calls with its NumPy oracle and passed bit for bit. The rebuilt
extension then passed all 41 native layout descriptors, the complete CPU gate,
and all four official safe-DMA board modes:

| Gate | Dispatches / groups | Wall ms | Process wall ms | Relative L2 / RMSE |
|---|---:|---:|---:|---:|
| Decoder only | 89 / 38 | 568.004 | 626.134 | 0 / 0 |
| Resume layer 11 | 118 / 55 | 628.922 | 687.996 | 0 / 0 |
| Full first frame | 443 / 248 | 1,663.927 | 1,727.756 | 0 / 0 |
| Full resident frame | 443 / 248 | 1,636.434 | 1,667.766 | 0 / 0 |

Every frame produced depth SHA-256
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`,
with safe DMA, zero stale events, zero codec fallback, and the expected native
pack/unpack counts. The resident frame reused the bank, C++ runtime, and codec
YAML and recorded `load_ms=0`. Independently reopening all four saved NPZ files
and hashing their contiguous FP32 arrays reproduced the same expected digest.

A separate same-PID run used one warm-up plus five measured resident frames.
All five remained bit exact. Wall latency was 1,602.399--1,730.097 ms, with
mean 1,663.080 ms, median 1,671.527 ms, p95 1,722.328 ms, and sample standard
deviation 52.161 ms. Median component times were NPU 443.231 ms, input pack
261.585 ms, decoder host operators 224.264 ms, output unpack 130.978 ms, H2C
80.299 ms, and C2H 66.577 ms. Within the instrumented host profile,
`encoder.gelu_quantize` is now the largest remaining operation at a 281.688 ms
median over 12 calls.

## BF16 GELU LUT r62 qualification (2026-09-08)

The first priority was refined using the r61 profile. FC2 native packing costs
only about 1.3 ms per layer, while scalar GELU-to-INT8 consumed 281.688 ms per
resident frame. FC1 outputs arrive as exact BF16 values expanded to FP32, so
r62 caches an exact 65,536-entry BF16-to-INT8 GELU table for each quantization
scale. Arbitrary non-BF16 FP32 input still uses the qualified scalar path. The
cache is owned by the extension module rather than one executor instance, so
the first full frame records 12 misses and later resident frames record 12
hits even though the request server constructs a new executor facade.

Qualification covers the complete finite BF16 bit domain for all three test
scales, the real r52 trace, all 41 native layout descriptors, and the complete
443-dispatch CPU control flow. Four official safe-DMA board modes and five
additional same-PID resident frames remained bit exact with zero stale events.

| Measurement | r61 median | r62 median | Change |
|---|---:|---:|---:|
| Resident wall | 1,671.527 ms | 1,517.618 ms | -153.909 ms (-9.21%) |
| GELU-to-INT8 | 281.688 ms | 39.566 ms | -242.122 ms (-85.95%) |
| Host profile total | 357.424 ms | 126.673 ms | -230.751 ms (-64.56%) |
| NPU | 443.231 ms | 441.366 ms | -1.865 ms |

The r62 wall distribution is 1,473.650--1,628.159 ms, mean 1,535.459 ms,
median 1,517.618 ms, and p95 1,615.755 ms. The difference between the GELU
gain and end-to-end gain is primarily run-to-run movement in native pack and
unpack: their r62 medians are 306.239 and 154.207 ms. Decoder host operators
remain 226.954 ms, dominated by bilinear Resize. Consequently the next work is
ordered as follows:

1. Benchmark direct GELU-LUT-to-NDWC packing; retain it only if it improves
   beyond the approximately 16 ms/frame standalone FC2 packing cost.
2. Implement and qualify a C++ align-corners bilinear Resize path, currently
   about 189 ms/frame on NumPy.
3. Fuse the general quantize/pack and unpack/consumer boundaries, now the
   largest combined host cost.
4. Retain more intermediate encoder tensors on the card where compatible BIN
   address contracts permit it.
5. Evaluate persistent DMA descriptors last, because H2C+C2H is much smaller
   than layout conversion and host operators and safe DMA is the proven mode.

r62 reproducibility hashes:

- DS extension: `a1efba57f43cc2f168aedc44a46c5e3a93a6e879a13f3bccfa145e37d951183d`.
- Host-executor report: `c8e328615658113c6c3c5b23d95f85166955917fbe6ac2a639413a8ee8b0ad35`.
- Native ALL oracle: `e3b7ee5a1865fed7c74c8842b473f1b52715442e0db933106662c07b19f19f93`.
- Canonical input inventory: `3787b722125b4a766b819f06ecc1fb5a117dcab5f6c10d93b9cb85948b4da1c2`.
- CPU control-flow summary: `4dfd3b56b7881c26d369497531d3fd05367a57274412052b35a6edeac986f12b`.
- Deployment inventory: `75b00cdbb28bedf0da6344ee8631f960ddc67386793f5d4b8be3ec9cf3a5d640`.

## C++ align-corners Resize r63 qualification (2026-09-08)

Priority 2 moves all five decoder `Resize` nodes from NumPy to the qualified
C++ host executor. The implementation preserves the former operation order:
coordinates are rounded to FP32, interpolation weights and the vertical and
horizontal products are evaluated as FP64, and the result is cast to FP32.
Compilation retains `-ffp-contract=off`. Qualification covers output-dimension
one, every decoder spatial transition, non-square 75x518 output, and the real
r52 trace. All cases are byte-exact against the original NumPy implementation.

The summary schema is now version 3. Both the CPU control-flow and board gates
require exactly five native Resize calls and reconcile them into the total
host-call count. The complete fake-transport run recorded 443 dispatches,
1,103/683 native pack/unpack calls, zero device or lock opens, and five C++
Resize calls. All four safe-DMA board modes and five additional resident
frames retained the expected depth SHA-256 with zero RMSE and relative L2.

| Measurement | r62 median | r63 median | Change |
|---|---:|---:|---:|
| Resident wall | 1,517.618 ms | 1,387.169 ms | -130.449 ms (-8.60%) |
| Decoder host operators | 226.954 ms | 46.249 ms | -180.706 ms (-79.62%) |
| Align-corners Resize | 192.540 ms | 10.530 ms | -182.011 ms (-94.53%) |
| NPU | 441.366 ms | 441.718 ms | +0.352 ms (+0.08%) |
| Native input pack | 306.239 ms | 318.288 ms | +12.049 ms |
| Native output unpack | 154.207 ms | 173.944 ms | +19.737 ms |

The five r63 resident wall samples span 1,377.000--1,455.104 ms, with mean
1,403.013 ms, median 1,387.169 ms, p95 1,447.244 ms, and sample standard
deviation 32.921 ms. Resize is no longer a primary bottleneck. Priority 3 is
therefore quantize/pack and unpack/consumer fusion; those two layout boundaries
now total about 492.232 ms at the median, versus 441.718 ms on the NPU itself.

r63 reproducibility hashes:

- DS extension: `54832ff34e66deae0608c90446759ef767ec5863d99be8810d904a4557b577b5`.
- Host-executor report: `0fb15b0f16194a405130a44b09a8d774113fd04120a65b951c4f9a454c9d5bfb`.
- Native ALL oracle: `0a3abc48353a5f003454bfe6fc77f276e26e8b4c39ff96d3dddd0754052a471b`.
- Canonical input inventory: `928758f8b1a8244771d85ffe077a5519a949fd4c704423455cd9f0296deeedcd`.
- CPU control-flow summary: `0a1676b85431e222b2af6a3fa2b39b397bb46a42a8c75abfc5b44581597a98ef`.
- Deployment inventory: `798e095157b45e4eb9d646d1b0a1b7cc8077e1b1ff98b4caef29b38ebf6bae99`.
- Repeated-latency report: `6e917227a7604848c890bcf0bad8175bd346807c1881deb533b75ee4cfd35df7`.

## Explicit physical-pack reuse r64 qualification (2026-09-08)

Priority 3a removes repeated layout conversion when one immutable logical
tensor feeds several ABI-identical BIN inputs. Reuse is explicit: the runner
wraps only tensors whose lifetime is known to be read-only, and caches by the
complete address-independent descriptor identity. It does not infer
immutability from a NumPy object address. The covered fan-outs are attention
K/V across query slices, six-way MLP FC1, six-way patch projection, and
channel-sliced decoder convolutions.

The schema-4 gates require the exact full-frame reduction from 1,103 to 721
native pack calls, 382 cache hits, 69,055,500 skipped logical bytes, and
72,978,432 skipped physical-layout bytes. The formal strace control-flow gate
reproduced all four values with zero device/lock opens. Four board modes and
five resident measurements remained bit-exact with zero stale events.

| Measurement | r63 median | r64 median | Change |
|---|---:|---:|---:|
| Resident wall | 1,387.169 ms | 1,324.204 ms | -62.966 ms (-4.54%) |
| Native input pack | 318.288 ms | 204.531 ms | -113.757 ms (-35.74%) |
| Native output unpack | 173.944 ms | 190.076 ms | +16.132 ms |
| NPU | 441.718 ms | 441.886 ms | +0.168 ms (+0.04%) |
| Decoder host operators | 46.249 ms | 49.546 ms | +3.297 ms |

The five r64 wall samples span 1,301.958--1,376.075 ms, with mean
1,333.294 ms, median 1,324.204 ms, p95 1,369.630 ms, and sample standard
deviation 28.165 ms. Input pack is now 204.531 ms, and output unpack is the
next-largest layout boundary at 190.076 ms.
Priority 3b should fuse the two high-fan-out output consumers: attention BF16
chunks into post-attention INT8 input, and FC1 BF16 chunks into GELU/FC2 INT8
input. That also avoids intermediate concatenation and quantization passes.

r64 reproducibility hashes:

- DS extension: `54832ff34e66deae0608c90446759ef767ec5863d99be8810d904a4557b577b5`.
- Host-executor report: `ee2855208b6fc3ce8dc25ffa4a3e2f10c0149cfc2fa36de9233b70b8ce63b8d1`.
- Native ALL oracle: `0a3abc48353a5f003454bfe6fc77f276e26e8b4c39ff96d3dddd0754052a471b`.
- Canonical input inventory: `1fd2279f1b13698344150c89363587b2193e4a1ca8c228baecff3fb45f885393`.
- CPU control-flow summary: `13e971783c9794eb229d624fc3f873f654d513d1e6bff7b791253f0c2db744e7`.
- Deployment inventory: `f3cad40eaf98f97393334ee244bbd791a4e60c388a69a2be4a5da3cf1cc58827`.
- Repeated-latency report: `a7182bb2f54b69b166cf6514c478aea82d87544a1860f47f987be5c0c988a079`.

## Board gate result

The updated runner, resident server, and Python-3.13 C++ extension are deployed
under
`/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_lnfold_decoderretuned`.
The complete free-lock U250 gate passed on demo05.  Decoder-only, layer-11
resume, the first complete frame, and the second same-process complete frame
all matched the r43 depth SHA-256
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`
bit for bit (relative L2 and RMSE zero).

| Gate | Dispatches / C++ groups | Wall ms | NPU ms | H2C ms | C2H ms |
|---|---:|---:|---:|---:|---:|
| Decoder only | 89 / 38 | 4,063.556 | 79.437 | 28.093 | 28.555 |
| Resume layer 11 | 118 / 55 | 4,959.144 | 111.189 | 35.490 | 36.245 |
| Full first frame | 443 / 248 | 14,323.313 | 477.308 | 138.988 | 120.631 |
| Full resident frame | 443 / 248 | 14,202.047 | 479.567 | 135.801 | 110.968 |

The second full request reports `resident_bank_reused=true`, `load_ms=0`, and
`codec_yaml_reused=true`.  It reuses four persistent XDMA descriptors, the
mapped BAR, 42,344,448 bytes of pinned H2C capacity, and 10,084,352 bytes of
pinned C2H capacity.  Relative to the recorded 60,561.193 ms baseline, steady
wall latency fell by 76.55%, a 4.264x speedup.

The steady full-frame breakdown is input pack 7,497.623 ms (52.79%), output
unpack 5,343.998 ms (37.63%), NPU 479.567 ms (3.38%), residual graph/Python
419.871 ms (2.96%), decoder host operators 218.597 ms (1.54%), H2C 135.801 ms
(0.96%), and C2H 110.968 ms (0.78%).  Physical codec conversion is therefore
the clear next optimization target; DMA and NPU together are no longer the
end-to-end bottleneck.

## Native codec r60 board qualification (2026-09-07)

The r60 package is deployed separately at
`/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_nativecodec_r60`.
The existing r43/r58 and r59 packages and evidence were preserved. The numerical graph and resident
bank remain unchanged; the bank SHA-256 is
`9d01d1fd4ecae67755a4314e98a2f9d4cbe7182f3985d7573113de579b7f9577`.

All four executions passed with `--layout-codec native`, the singular ALL
qualification report, and safe DMA (open/seek/transfer/close per coalesced
segment). The full frames were two requests in one resident process. Every
output matched SHA-256
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`,
with `relative_l2=0`, `rmse=0`, no timeout, and `stale_events=0`. Independently
rehashing the saved output arrays and comparing them with the golden array
also passed. No persistent-descriptor diagnostic run was performed.

| Gate | Dispatches / groups | Wall ms | Process wall ms |
|---|---:|---:|---:|
| Decoder only | 89 / 38 | 583.275 | 641.512 |
| Resume layer 11 | 118 / 55 | 652.910 | 700.591 |
| Full first frame | 443 / 248 | 1,732.278 | 1,798.480 |
| Full resident frame | 443 / 248 | 1,590.284 | 1,621.344 |

The second full frame reports `resident_bank_reused=true`, `load_ms=0`,
`cpp_runtime_reused=true`, and `codec_yaml_reused=true`. Its native tensor
pack/unpack counters are 1103/683; vendor counters are 0/0, with no fallback.
The steady wall measurement is 38.082x faster than 60,561.193 ms (97.374%
lower), and 8.931x faster than 14,202.047 ms (88.802% lower). These compare one
measured r60 resident frame with historical recorded baselines; they are not
projections or a latency-distribution claim. The resident full frame was
141.994 ms faster than the first in this sample.

| Measured component | First full ms | Resident full ms |
|---|---:|---:|
| Bank load | 22.483 | 0.000 |
| Cfg preparse | 15.949 | 0.000 |
| Vendor cfg activation | 0.000 | 0.000 |
| Input pack | 281.935 | 244.830 |
| H2C | 80.805 | 80.015 |
| NPU | 442.154 | 441.203 |
| C2H | 72.924 | 64.784 |
| Output unpack | 136.090 | 128.016 |
| Decoder host operators | 246.078 | 237.586 |
| Host graph/Python residual | 500.063 | 424.911 |

The breakdown sums to `process_wall_ms`. The historical-comparison field
`wall_ms` starts after host input/setup and before runtime/bank setup; the
residual uses the broader process interval. Neither interval includes Python
interpreter startup. C++ group DMA time is charged to the first physical
dispatch; NPU time remains per BIN.

The runtime now requires the qualification report's `extension_sha256` to
match the exact extension file before importing it or constructing DMA.
It records the digest captured at module load and retains that identity with
the cached runtime. Native reuse rejects changed requested paths, replaced
files, and untracked loaded modules, including vendor-to-native transitions.
Auto mode falls back every affected case direction with explicit extension
provenance reasons and vendor tensor accounting. Each frame records the loaded
path/digest and qualification digest in `cpp_runtime`.

The DS Python 3.13 extension and validator bytes match r59, so the existing
41-descriptor ALL oracle remains applicable (41 exact and enabled). Its
extension hash matches the deployed r60 file and every frame's recorded
loaded digest. The Python change received a fresh full CPU control-flow
qualification before board access: 443/248 calls/groups, 1103/683 native tensor
calls, zero vendor calls, zero device/lock opens or protected writes, 2803
traced opens, and 520.806 ms total native codec work. The canonical inventory
was independently rehashed and repinned. The stale-event counter is cumulative
within each frame and is reset by `reset_frame_stats()`.

Reproducibility hashes:

- Native C++ source: `c9e3e1410263a0f3177213920706b18d699659bac6e0ecb92ff1e07df3b8ee0a`.
- DS extension: `2cd20094a27e557568ea9c06d816f0eba7cc35c1ded566d9506958205b296e65`.
- Manifest: `8ee71ef3e53b14aa3a2b078c1ee5d3eedc230768ab5e40c518db17e668936877`.
- ALL report: `139f224a3d725b20facfeca7946ab6b909f8e2c033c98e79541b9bf009c52fb5`.
- Python runtime: `4e86319db9df2a710708480d4f3dc282b902711168dae6b8786c01ed4f66d4c1`.
- Canonical input inventory: `7c2432aa0890c6de104294c0fccea0260d1d027275d5495d0360dacfcc9f72fa`.
- CPU control-flow summary: `00102299541f2176fdb04bd052f96b30298484edb60c3bfc3d37e45e418f2330`.
- Deployment inventory: `2dcfffd749e121722ea509111f837e143da0f746a152af61f599d23ac3066982`.

The gate verifies the independently pinned deployment inventory, including
all source, extension, input, cfg, helper, manifest, report, and runtime
hashes, before importing the extension. Run from the deployed package with:

```bash
U250_DEPLOYMENT_SHA256=2dcfffd749e121722ea509111f837e143da0f746a152af61f599d23ac3066982 \
  bash tools/run_u250_mapped_r58_gate.sh
```

It obtains `/tmp/ds-u250-runtime.lock` once with `flock -n`; contention exits
75 before package writes or device access. It requires a fresh
`native_r60_gate` output directory and will refuse to overwrite retained
evidence. The successful run is preserved there. Committed evidence is in
`artifacts/u250_native_codec/gate_summary.json`, with all four individual
summaries, execution logs, and the verified deployment inventory alongside.
`r60_output_verification.json` records the independent saved-array rehashes
and preservation checks for the original r43/r59 bank and r59 evidence.
The CPU-only `--check-native-board RUN_DIR` entry point revalidates retained
summaries/logs and regenerates the aggregate summary without taking a board
lock or accessing a device.
