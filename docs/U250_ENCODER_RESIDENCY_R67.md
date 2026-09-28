# U250 encoder intermediate residency: r67 address qualification

The r66 resident bank can safely retain two encoder boundaries without changing
the BIN or bitstream:

1. the block residual `x` uploaded for norm1 can be reused as the residual input
   of post-attention; and
2. the BF16 post-attention output can be consumed directly by norm2.

All 12 encoder blocks have the same FM tensor ABI and address pattern. The shared
workspace is 8 MiB per bank (65,536 address units, 128 bytes per unit per bank).
The qualified plan uses only 20,640 units:

| Lifetime/region | Unit interval per bank | Bytes per bank |
|---|---:|---:|
| Existing QKV and attention scratch | `[0, 8256)` | 1,056,768 |
| Retained residual `x` | `[8256, 12384)` | 528,384 |
| Resident post-attention output | `[12384, 16512)` | 528,384 |
| Resident norm2 output | `[16512, 20640)` | 528,384 |

This leaves 5,746,688 bytes per bank unused. The required FM base offsets are
8,256 units for norm1, 6,192 for post-attention, and 12,384 for norm2. QKV and
attention remain at the compiled base. Post-attention input 0 temporarily reuses
`[6192, 8256)` only after QKV output has been materialized.

The U250 probe used the real r66 resident bank. It ran norm1 at the relocated
address, then an ordinary QKV and attention kernel at the default address, and
verified that retained `x` was byte-exact. Relocated post-attention produced the
same valid BF16 elements (`max_abs=0`), and norm2 consuming that output in place
produced a byte-exact physical output.

Whole-buffer equality is not a valid compatibility test for a reused output
region: DS kernels need not overwrite padding lanes, so padding can retain bytes
from the prior owner. Runtime qualification must compare the storage ABI and
valid descriptor lanes, and must never expose padding as tensor data.

The remaining encoder boundaries still require host materialization:

| Boundary | Reason |
|---|---|
| norm1 to QKV | fixed-scale BF16-to-INT8 quantization |
| QKV to attention | head slicing, K transpose, and query slicing |
| attention to post | head assembly and BF16-to-INT8 quantization |
| norm2 to FC1 | fixed-scale BF16-to-INT8 quantization |
| FC1 to FC2 | GELU, six-slice assembly, and quantization |
| FC2 to next norm1 | residual add with post is still on host |

The r67 runtime integration removes 25,362,432 H2C bytes, 24 native BF16 pack
calls (50,503,680 logical bytes), and 12 Python/C++ submission groups per full
frame. It does not reduce C2H yet because post is still needed by the host
residual add, and it does not reduce the 443 NPU dispatches.

The implementation gate is fail-closed: a device handle must carry its physical
interval, valid-lane storage ABI, generation, and lifetime; an overlapping write,
ABI mismatch, stale generation, or out-of-workspace relocation must be rejected
before device access.

The four-mode U250 gate passed with exact final depth in decoder-only, L11
resume, full-first, and full-resident execution. Handle creation/invalidation
counts were respectively 0/0, 5/5, and 60/60 for both full runs, with zero live
handles at every frame boundary. `post -> norm2` device connections were
respectively 0, 1, and 12. The final reboot-clean steady resident frame measured
1,357.16 ms, including 449.92 ms NPU time; the result SHA-256 remained
`2ec1dbc8f769d319067e113a3139188556bd7e0b145ebe38291f5ed6b8617725`.
