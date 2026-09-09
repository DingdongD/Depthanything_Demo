# U250 DepthAnything V2 accuracy correction r79

Date: 2026-09-09. Board: `visitor@192.168.115.178`.

## Result

r79 promotes the r77 decoder width-tail fix and selectively executes encoder
Layer 0 / attention Head 3 in host FP32.  The other 71 attention heads remain
on the U250 as fixed-scale INT8 QK/AV programs.  The host head is converted to
the exact post-attention INT8 code before it enters the BF16 physical fusion
container, so enabling fusion does not add an unintended BF16 rounding step.

On `demo05`, against the PyTorch FP32 output:

| Version | relL2 | cosine | MAE | Pearson | SSIM |
|---|---:|---:|---:|---:|---:|
| r74 baseline | 0.418367 | 0.913680 | 0.774932 | 0.640012 | 0.786443 |
| r77 decoder width fix | 0.329599 | 0.944142 | 0.635701 | 0.845844 | 0.830295 |
| r79 Layer0/Head3 correction | 0.315790 | 0.951336 | 0.601928 | 0.873982 | 0.847782 |

The twelve-sample board calibration set gives mean relL2 `0.429587`, cosine
`0.941188`, MAE `0.910681`, and Pearson `0.860231`.  This is a real improvement
over the original deployment, but it is not FP32-equivalent accuracy.  The
remaining dominant error is accumulated encoder quantization: the measured
attention surrogate-vs-FP32 relL2 is `0.1734` at layer 0, `0.4016` at layer 1,
and `0.7891` at layer 2.  Decoder-only execution from FP32 encoder captures is
substantially closer, and the former `resize_layers.3/Conv` width-tail
corruption has already been removed by the width-64/crop-19 kernel contract.

## Board gate

The clean remote deployment is:

`/home/visitor/Documents/depthanything_u250_accuracy_r79_l00h3`

Its default runtime contract enables the correction.  The final smoke test
completed with output SHA-256 `71185fff67b8cb254f42ee51358624949b5139be7b232425a54217de3b7e54ff`,
404 NPU dispatches, 12 physical attention fusions, zero static weight reloads,
and zero vendor codec fallbacks.  NPU time was `441.734 ms`; full wall time was
`1465.506 ms` in the clean-package run.  Three independent runs in the source
package were bit-identical; their wall times were `1409.765`, `1640.709`, and
`1499.612 ms`, showing that host-side latency still has material variance.

## Evidence

- `demo05_depth_comparison.png`: shared-range FP32/r74/r77/r79 depth and error
  visualization.
- `accuracy_evidence.json`: demo05 metrics, per-sample calibration metrics,
  aggregates, board gate counters, and immutable hashes.
- `demo05.npz` and `demo05.summary.json`: clean r79 board output and runtime
  summary.
