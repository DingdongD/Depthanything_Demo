# U250 decoder residency priority 1-5 (r68)

Date: 2026-09-09. Board: `visitor@192.168.115.178`, U250 bitstream PCI ID
`10ee:902f`, XDMA interrupt mode 0. All experimental paths are fail-closed and
the qualified production default remains r67.

## Results

1. Encoder captures: **qualified**. Blocks 2/5/8/11 use fixed per-bank FM
   offsets 20640/33024/45408/53664 (256-byte units). The pure-SPU run exercised
   all four captures with 15 forwarded device inputs, 65 handle creations, 65
   invalidations, and zero live handles at frame completion. It skipped
   28,532,736 H2C bytes, 3,170,304 more than r67.
2. Decoder LayerNorm: **functionally qualified, not selected for performance**.
   Four `tail_norm1_from_tokens` SPU calls completed on hardware. Final-depth
   relative-L2 versus r43 is 0.0075011, cosine 0.9999721, max-abs 0.21875.
   The four calls add 12.234 ms NPU, 6.014 ms H2C and 8.288 ms C2H while r67's
   four host LayerNorms cost 8.213 ms. Full cold wall time was 1448.391 ms and
   NPU time 454.674 ms, so standalone offload is slower.
3. Slice/Transpose/Reshape device view: **compiler issue partially fixed, board
   incompatible**. A missing DDRSwap output bit depth was safely inherited from
   its input, allowing liveness to advance. Keeping Slice in the device graph
   then exceeds FM0 because the 1370-token input and 1369-token ROI overlap.
   Moving the zero-copy Slice to host lets all 12 `Mat2Img + Conv` kernels
   compile, but the first physical kernel times out on the current bitstream
   with status `0x0`.
4. Direct project Conv connection: **offline-qualified, board incompatible**.
   The normalized-token BF16 path has worst holdout degradation ratio 1.00318
   and worst relative-L2 0.050548. Capture-input variants also compile and pass
   offline gates, but both EPU-affine and folded-affine variants time out at the
   first stem. The common unsupported boundary is the cross-unit layout/Conv
   program, not standalone Conv or standalone SPU LayerNorm.
5. Conv + DepthToSpace + Resize: **evaluated and rejected for production**.
   The exact factor-4 polyphase D2S/Conv probe is bit-exact but requires 16
   kernels and 65.115 ms NPU. In r67 all three host DepthToSpace operations cost
   3.406 ms and all five Resize operations cost 15.793 ms. The probe is already
   slower than the entire host D2S+Resize budget and does not remove Resize, so
   integrating it would regress latency substantially.

## Evidence

- `decoder_layernorm_board.summary.json`: successful four-layer SPU run.
- `capture_stem_epu_timeout.log`: SPU + EPU affine + layout + Conv timeout.
- `capture_stem_fold_timeout.log`: SPU + folded affine + layout + Conv timeout.
- `patch_project_timeout.log`: host Slice + device Mat2Img/Conv timeout.
- `capture_stem_epu_holdout.json`, `capture_stem_fold_holdout.json`, and
  `layout_project_holdout.json`: independent demo05/demo06 precision gates.
- `layout_project_pre_liveness_fix.log` and
  `layout_project_post_liveness_fix.log`: compiler failure before/after the
  DDRSwap metadata fix.
- `d2s_polyphase_board.summary.json`: bit-exact 16-phase factor-4 board probe.

## Production decision

Keep r67 as the default full-model image. The r68 runner and exporters retain
the capture/LN/layout-stem implementations behind explicit flags for a future
bitstream/compiler revision. Do not ship any stem that merely compiles: the
board timeout gate is authoritative.
