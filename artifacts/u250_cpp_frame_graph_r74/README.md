# U250 C++ frame graph / Decoder native boundary (r74)

## Result

- The mapped runtime now exposes a typed `run_frame_graph` interpreter. It
  validates the complete dataflow before board access, owns a tensor table,
  releases intermediates after their final use, and holds one schedule lock
  across host transforms, DMA, NPU chains, and C2H.
- Supported opcodes cover `device_read`, `device_write`, `npu_chain`, native
  pack/unpack, quantize/GELU/add/concatenate/resize, the physical FC1 and
  attention bridges, and the Decoder capture bridge.
- Every H2C/NPU/C2H transaction is now compiled to this interpreter; the
  legacy resident-transaction scheduler records zero calls. A full frame runs
  602 typed nodes and all 407 physical NPU programs through 196 interpreter
  invocations on one persistent mapped runtime.
- The production Decoder boundary itself is compiled to 17 nodes. Four
  BF16 captures are snapshotted before FM reuse, transformed by four exact
  LayerNorm/CLS-drop/NDWC-to-NCHW/INT8 bridge operations, and consumed by 12
  already-qualified project Conv programs.

The old compiler-generated `LayerNorm -> Mat2Img -> Conv` and
`Mat2Img -> Conv` stems are not submitted. They compile, but the deployed U250
bitstream does not complete the cross-unit Mat2Img/Copy -> DDRSWAP -> CTC
schedule (status remains `0x0`). Standalone LayerNorm and project Conv programs
do complete. The r74 bridge therefore removes Mat2Img from the hardware
boundary and uses only the qualified Conv BIN ABI; no bitstream rebuild is
required.

## Board evidence

Board: `visitor@192.168.115.178`, safe DMA, current installed bitstream.

- 3/3 full frames completed with identical output SHA-256
  `1ece41eeb9fb46f98b530d131fe4798ce59915d40e4f7fa8ea1e23dd2f3e8967`.
- 407 physical NPU dispatches; 199 logical submission groups.
- Full interpreter: 196 calls, 602 nodes, 407 programs, 0 legacy
  resident-transaction calls.
- Decoder interpreter: 17 nodes, 12 NPU programs, peak 12 tensor-table
  entries, 4 physical bridge calls.
- Full-frame wall: 1305.488, 1327.032, 1280.332 ms; median 1305.488 ms.
- NPU time is about 440 ms.
- All 60 device handles were invalidated; live handle count is zero.
- No stale interrupt events and no vendor codec fallback.

Against the previous exact host-capture golden, cosine similarity is
0.9999744067 and relative L2 is 0.0071544453. This deterministic delta is from
the intended encoder-to-decoder residency boundary: captures are rounded to
the NPU BF16 physical representation before Decoder LayerNorm. The C++ bridge
itself was checked byte-for-byte against the same BF16 physical input contract;
the three board runs are identical.

## Evidence files

- `board_summary.json`: final full-frame board summary and timing breakdown.
- `board_depth.npz`: final board depth output.
- `repeat_summary.json`: compact 3-run determinism/latency report.
- `native_codec_all_oracle.json`: native codec qualification bound to extension
  SHA-256 `caf7228e...762f3b`.
- `host_executor_qualification.json`: exact host/physical-fusion qualification.
- `full_controlflow.summary.json`: CPU-only, zero-output full control-flow gate.
- `full_controlflow.opens.log`: strace evidence showing no board device or lock
  access during the CPU-only gate.
- `controlflow_input_inventory.json`: pinned input/cfg/runtime digests.

The interpreter is now the sole physical DMA/NPU scheduler for the full frame.
Data-dependent host boundaries still divide the frame into 196 checked graph
fragments, so this evidence does not claim a single Python-to-C++ call for the
entire model. The Decoder capture/project boundary is one 17-node fragment that
contains all four project groups.
