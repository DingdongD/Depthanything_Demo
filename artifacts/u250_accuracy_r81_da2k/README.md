# U250 r81 DA-2K closed-loop calibration

This experiment keeps r80 (`decoder_conv_31_tile74` input scale `0.35`) as
the deployment baseline and tests one variable at a time against the same
16-image tuning and 16-image holdout subsets.

## Result

No r81 candidate is promoted.  The deployed package remains r80 because all
otherwise promising candidates fail at least one cross-domain or ordinal-pair
gate.

| Candidate | 32-image rel-L2 | MAE | cosine | Pearson | affine rel-L2 | pair accuracy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r80 baseline | 0.367132 | 0.770731 | 0.975635 | 0.949430 | 0.177302 | 62/65 | keep |
| layer-3 QKV A8 scale 0.11 | 0.367298 | 0.766825 | 0.976056 | 0.950118 | 0.176147 | 63/65 | reject: demo05 Pearson regression |
| layer-3 QKV A8 scale 0.146016766 | 0.368149 | 0.767679 | 0.975956 | 0.950290 | 0.176316 | 62/65 | reject: rel-L2 and demo05 regression |
| decoder conv18 A8 scale 0.8 | 0.361896 | 0.760770 | 0.976719 | 0.952258 | 0.172708 | 61/65 | reject: one ordinal-pair regression |
| decoder conv18 A8 scale 0.55 | 0.362901 | 0.762524 | 0.976434 | 0.951561 | 0.173844 | 61/65 | reject: one ordinal-pair regression |

Layer-3 per-head K-scale replacement also regressed all continuous metrics on
the eight-scene screen and was rejected before the full gate.  Histogram MSE
is therefore useful for candidate generation, but not sufficient for selecting
Q/K scales because it changes the effective QK-logit temperature.

The conv18 scale sweep exposed two independent pair boundaries.  Scale `0.55`
preserves `tuning_0020` but makes one `holdout_0227` pair exactly tied; scale
`0.54` restores both `holdout_0227` margins but flips `tuning_0020`.  A single
per-tensor input scale cannot satisfy both strict gates.  The next correction
should jointly optimize conv18 and a downstream scale, or use channel/group
compensation, with the two samples as hard constraints.

## Hardware notes

All successful runs used 404 physical NPU dispatches, zero static reloads, and
zero vendor pack/unpack calls.  Repeated process-level loading of different
50.6 MB resident banks can eventually leave the current U250 runtime in a bad
state (non-finite encoder output or an attention timeout).  The r80 health gate
returned its bit-exact output SHA-256
`c63a264ddb574d413f90d2264c66efb17c8293f7ee72728a52cc5452cc9e8128`
after reboot.  Formal calibration was consequently split into at most eight
frames per reboot; this is a sweep-only reload issue, not the normal one-bank
resident deployment path.
