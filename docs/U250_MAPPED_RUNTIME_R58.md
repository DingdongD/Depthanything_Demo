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
process took 1,212.561 ms, including 529.310 ms of native layout conversion;
this is a host-only structural measurement and not a U250 latency result.

The retained evidence is portable: callers can explicitly provide the layout
oracle, host-executor qualification report, and canonical input inventory,
without relying on paths from the remote invocation. The r61 deployment
inventory contains 293 independently rehashed files and passed the CPU-only
package verifier before any board lock or device access.

Reproducibility hashes for this CPU gate:

- DS extension: `ab9f0d92a91adf16cc6f2c632cf05d6a6d783141b8231cafeda5b80cb65f023f`.
- Host-executor report: `802257bb1e49124d6ea5a7d66a499863bb3c66e600a24fa00cf86905e971bac3`.
- Native ALL oracle: `7d2070de670449c8f7252af12a36af459b8025eb1a6b414e84a4c17ace46b6d5`.
- Canonical r61 input inventory: `b5be2eb3e49bd8027e2dacfc3a09daf665f273ed11e6fce8493c7cf378b6f26c`.
- CPU control-flow summary: `89e971eabc54677f5b4b3363f882a6addf577827197f4429a3d42934601086ca`.
- Deployment inventory: `d033c4e57e39d860d8506f56db6fe7cc7e470759b02e11b8578076f10cf66ed0`.

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
