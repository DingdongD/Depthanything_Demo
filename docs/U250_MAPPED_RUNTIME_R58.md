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

## Native codec r59 board qualification (2026-09-07)

The r59 package is deployed separately at
`/home/visitor/Documents/depthanything_u250_resident_kernel_bank_r43_nativecodec_r59`.
The existing r43/r58 package was preserved. The numerical graph and resident
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
| Decoder only | 89 / 38 | 590.745 | 654.610 |
| Resume layer 11 | 118 / 55 | 641.210 | 700.707 |
| Full first frame | 443 / 248 | 1,682.535 | 1,745.679 |
| Full resident frame | 443 / 248 | 1,697.086 | 1,728.990 |

The second full frame reports `resident_bank_reused=true`, `load_ms=0`,
`cpp_runtime_reused=true`, and `codec_yaml_reused=true`. Its native tensor
pack/unpack counters are 1103/683; vendor counters are 0/0, with no fallback.
The steady wall measurement is 35.685x faster than 60,561.193 ms (97.198%
lower), and 8.368x faster than 14,202.047 ms (88.050% lower). These compare one
measured r59 resident frame with historical recorded baselines; they are not
projections or a latency-distribution claim. The first full frame happened
to be 14.551 ms faster than the second in this sample.

| Measured component | First full ms | Resident full ms |
|---|---:|---:|
| Bank load | 22.418 | 0.000 |
| Cfg preparse | 15.815 | 0.000 |
| Vendor cfg activation | 0.000 | 0.000 |
| Input pack | 277.353 | 300.618 |
| H2C | 78.115 | 83.083 |
| NPU | 441.948 | 442.277 |
| C2H | 72.687 | 74.084 |
| Output unpack | 137.834 | 152.218 |
| Decoder host operators | 216.541 | 223.014 |
| Host graph/Python residual | 482.968 | 453.696 |

The breakdown sums to `process_wall_ms`. The historical-comparison field
`wall_ms` starts after host input/setup and before runtime/bank setup; the
residual uses the broader process interval. Neither interval includes Python
interpreter startup. C++ group DMA time is charged to the first physical
dispatch; NPU time remains per BIN.

The existing C++ stale-event counter was exposed through `stats()` so this
gate can reject nonzero or missing evidence. Scheduling and event-wait code
were unchanged; the counter is cumulative within a frame and is reset by
the existing `reset_frame_stats()` path. This one-line observability change
required a fresh DS Python 3.13 build, a fresh 41-descriptor ALL oracle
(41 exact and enabled), and a fresh complete CPU control-flow qualification
before board access. That CPU replay recorded 443/248 calls/groups,
1103/683 native tensor calls, zero vendor calls, zero device/lock opens, and
557.154 ms total native codec work.

Reproducibility hashes:

- Native C++ source: `c9e3e1410263a0f3177213920706b18d699659bac6e0ecb92ff1e07df3b8ee0a`.
- DS extension: `2cd20094a27e557568ea9c06d816f0eba7cc35c1ded566d9506958205b296e65`.
- Manifest: `8ee71ef3e53b14aa3a2b078c1ee5d3eedc230768ab5e40c518db17e668936877`.
- ALL report: `139f224a3d725b20facfeca7946ab6b909f8e2c033c98e79541b9bf009c52fb5`.
- Deployment inventory: `e82b5f546e22ab60bf03770756fa76be749ae4788c826aba14b355f398cb01c3`.

The gate verifies the independently pinned deployment inventory, including
all source, extension, input, cfg, helper, manifest, report, and runtime
hashes, before importing the extension. Run from the deployed package with:

```bash
U250_DEPLOYMENT_SHA256=e82b5f546e22ab60bf03770756fa76be749ae4788c826aba14b355f398cb01c3 \
  bash tools/run_u250_mapped_r58_gate.sh
```

It obtains `/tmp/ds-u250-runtime.lock` once with `flock -n`; contention exits
75 before package writes or device access. It requires a fresh
`native_r59_gate` output directory and will refuse to overwrite retained
evidence. The successful run is preserved there. Committed evidence is in
`artifacts/u250_native_codec/gate_summary.json`, with all four individual
summaries, execution logs, and the verified deployment inventory alongside.
The CPU-only `--check-native-board RUN_DIR` entry point revalidates retained
summaries/logs and regenerates the aggregate summary without taking a board
lock or accessing a device.
