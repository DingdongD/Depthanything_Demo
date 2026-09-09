# U250 DepthAnything V2 DA-2K calibration r80

Date: 2026-09-09. Board: `visitor@192.168.115.178`.

The calibration set contains 512 scene-stratified DA-2K images across eight
domains.  A separate 128-image tuning split selected the candidate, and a
393-image holdout split was not used for selection.  The board gate uses a
balanced 32-image subset (16 tuning and 16 holdout images, 65 annotated
relative-depth pairs).

r80 changes only the A8 input scale of the final 1x1 Decoder Conv
`/depth_head/output_conv2/output_conv2.2/Conv`, from `0.1427165354` to `0.35`.
The original boundary clipped `2.435%` of DA-2K activation samples.  Pure
histogram MSE suggested `0.945742`, but end-to-end testing rejected that value;
`0.35` is the conservative Pareto point that improves DA-2K continuous and
pair metrics without a material demo05 regression.

| Gate | Metric | r79 | r80 |
|---|---|---:|---:|
| DA-2K 32 | relL2 | 0.376522 | 0.367132 |
| DA-2K 32 | cosine | 0.972968 | 0.975635 |
| DA-2K 32 | Pearson | 0.942605 | 0.949430 |
| DA-2K 32 | pair accuracy | 0.938462 | 0.953846 |
| Holdout 16 | relL2 | 0.382578 | 0.371828 |
| Holdout 16 | pair accuracy | 0.941176 | 0.970588 |
| demo05 | relL2 | 0.315790 | 0.315956 |
| demo05 | MAE | 0.601928 | 0.601295 |

The corrected resident manifest was re-qualified against all 42 active native
layout descriptors.  After reboot and XDMA reload, three formal demo05 runs
were bit-identical with output SHA-256
`c63a264ddb574d413f90d2264c66efb17c8293f7ee72728a52cc5452cc9e8128`.
The formal gate uses 404 NPU dispatches, zero static reloads, zero vendor codec fallbacks,
median NPU time `440.891 ms`, and median wall time `1441.722 ms`.

`accuracy_evidence.json` is the compact machine-readable result.  The
activation histogram, deterministic manifests, scale analysis, board outputs,
formal summaries, and native-codec qualification are retained in this
directory for reproduction.
