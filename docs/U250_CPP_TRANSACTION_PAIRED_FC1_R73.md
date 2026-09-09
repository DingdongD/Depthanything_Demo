# U250 r73: C++ transactions and paired FC1 dispatches

## Qualified result

The r73 package keeps one mapped `DmaBatch` and one C++ host-graph executor
alive for the server process.  Each dependency group now crosses Python once:
the C++ `run_resident_transaction` method holds the schedule lock across H2C,
all program launches in the group, and C2H.

The 1536-channel encoder FC1 is compiled as three programs per layer.  Each
program shares one 384-channel INT8 input and emits two ordered 256-channel
BF16 outputs.  The runtime flattens outputs in `(program, output-index)` order
before the existing C++ GELU/requantize/FC2 path.

| Metric | r69 | r73 | Change |
| --- | ---: | ---: | ---: |
| Physical NPU dispatches | 443 | 407 | -36 |
| Python submission / C++ transaction calls | 236 | 200 | -36 |
| Warm wall median (5 frames) | 1289.175 ms | 1248.983 ms | -40.192 ms (-3.12%) |
| Warm process median | 1320.762 ms | 1280.614 ms | -40.147 ms |
| Warm NPU median | 442.742 ms | 439.990 ms | -2.753 ms |

All six resident-server frames produced SHA-256
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.
The 268,324 final-depth values are bit exact against the retained r43 golden:
zero mismatches, zero max/mean absolute error, and zero relative L2 error.
Every frame reported 407 dispatches, 200 C++ transactions, and zero stale
events.

## Address compatibility finding

Relinking a smaller bank moved later static program/parameter addresses and
caused the first observable error at `attention_l11`, even though layer-11 Q,
K, and V logical tensors were exact.  Pinning only the shared FM base did not
fix it.  The released image therefore uses an in-place replacement policy:

- each pair of original FC1 images owns a 204,800-byte aligned slot;
- the paired image occupies 199,168 bytes of that slot;
- the remaining bytes are zero guard space;
- every byte outside the 36 original FC1 slot pairs is unchanged from r69;
- all later static bases and the qualified FM base `99856` remain unchanged.

The byte audit found 3,688,696 changed bytes inside FC1 slots and zero changed
bytes outside them.  The full bank remains 42,340,352 bytes.

## Device-native residency boundary

The encoder already retains two compatible BF16 NDWC boundaries per block:
the residual input from norm1 to post-attention, and post-attention to norm2.
Twelve device connections skip 25,362,432 H2C bytes per frame and all handles
are invalidated at frame end.

The four encoder captures cannot yet be aliased directly into the released
decoder project convolutions.  A capture is BF16 NDWC `[1,1,1370,384]`; the
decoder needs LayerNorm, class-token removal, transpose/reshape to
`[1,384,37,37]`, and fixed-scale INT8 packing before CTC.  Those are different
storage and quantization ABIs.  The tested fused BF16-NDWC LayerNorm/layout to
CTC route is not stable on the current bitstream, so r73 deliberately retains
the qualified C++ host materialization rather than claiming unsafe device
residency.

This is a single resident C++ transport/runtime object, but not one hardware
launch per frame: 200 data-dependent groups remain.  Reducing that further
requires either compiler-generated multi-op programs across the incompatible
quantization/layout boundaries or a C++ frame-schedule interpreter that owns
the remaining host graph; merely forwarding a `DeviceTensorHandle` cannot
change tensor semantics.

## Evidence

- `artifacts/u250_cpp_transaction_paired_fc1_r73/board_full_summary.json`
- `artifacts/u250_cpp_transaction_paired_fc1_r73/repeated_latency.json`
- `artifacts/u250_cpp_transaction_paired_fc1_r73/fc1_pair_l00_p00_summary.json`
- `artifacts/u250_cpp_transaction_paired_fc1_r73/full_controlflow.summary.json`
- `artifacts/u250_cpp_transaction_paired_fc1_r73/controlflow_input_inventory.json`
