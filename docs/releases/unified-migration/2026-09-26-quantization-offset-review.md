# B8 preparation: scalar quantization offset correction

Status: implemented and host-tested; board not-run. This is a shared H1 numerical
fix discovered while auditing B8 segmentation. B8 migration is still pending.

## Finding and decision

`samples/_shared/quantization.py` inherited the source S helper's behavior of
replacing a scalar zero-point with zero when the scale is per-channel. That
contradicts affine `(q - zero_point) * scale` decoding. For raw `[2, 4]`, scales
`[1, 3]`, and zero-point `7`, the old output `[2, 12]` chooses class 1; the
correct `[-5, -9]` chooses class 0. This can affect segmentation decisions.

Ruling: broadcast a scalar zero-point across the declared channel axis. Preserve
empty-offset symmetric decoding, vector offsets, scalar-scale decoding and
non-SCALE passthrough. Original platform snapshots remain byte-preserved.
This is an intentional correction to a source bug, not claimed legacy numerical
equivalence for that case. The old unit test explicitly expected the bug; it is
replaced with the affine arithmetic expectation and additional regression cases.

## Verification

- RED: 15 quantization tests, 3 expected failures (scalar offset, segmentation
  argmax, negative scalar offset on a non-last axis).
- GREEN: all 131 shared tests, DINOv2 18, FCOS 38, YOLOv5 79 passed: 266 total.
  Empty offset, per-channel vector offset and legacy non-SCALE cases remain covered.
- The three task families above are the direct production importers of this helper.
- A recursive scan of 269 existing evidence JSON files found no literal
  `scale`/`zero_point` object with vector scales and a scalar nonzero offset.
  This is limited evidence, not proof that every published model is unaffected.
- No board connection, SDK execution or model download was performed. Tests use
  host fixtures; their simulated download messages are not live transfers.

[Evidence and retained red/green logs](evidence/2026-09-26-quantization-offset.json).

## Documentation and next work

The shared README now explains scalar/vector/empty offsets and the deliberate
source difference. It also corrects its stale claim that raw F32 outputs reject
vestigial quant descriptors: the implemented `raw_f32` contract ignores them.

Continue B8 sample migration using the existing
[full source inventory](evidence/2026-09-23-b8-source-inventory.json). Do not infer
that UNet, UNetMobileNet or PP-LiteSeg share an output contract: source UNet
produces logits then an input-size mask; UNetMobileNet restores class IDs to
original geometry; PP-LiteSeg exports class IDs. The latter must not acquire
an extra argmax or affine transform. Source README instructions and current
artifact metadata must be checked per sample before selecting a binding.
