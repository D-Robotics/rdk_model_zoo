# YOLO26 Depth independent host review

Reviewer: Codex. Base `6377dc1a`. Accept the inspected host migration scope;
no blocking finding was identified in this review. This is the independent depth
capability retained by user decision, not a duplicate detection sample. Whole B8
and whole-branch delivery remain open. No product code was changed by the reviewer.

## Implementation and contracts

Reviewed Python task, binding, runner, geometry/restoration helpers and CLI;
native stage, tensor and resource implementations; offline metric formulas and
root/runtime/evaluator documentation. Inference files expose pre_process, forward,
post_process and predict, with file IO, rendering and SDK ownership separated.
Immutable per-call geometry prevents a later input from changing restoration.

The twenty manifest selections distinguish target and variant. All X5 and S n/s/m
profiles use letterboxed NV12 with calibrated log-depth. S l/x use scale-fill
RGB float32 input and clip/scale/bias raw-logit decoding. The API does not apply
those coefficients twice to calibrated outputs. Exponentiation precedes spatial
restoration. Output finiteness, profile/context agreement and padding geometry are
validated; results represent relative depth, not calibrated metric distance.

Native output reading validates rank, type, quantization, shape, checked byte
strides and allocation capacity before copying owned float data. Native task and
packed-model/buffer owners separate responsibilities and unwind injected errors.
Compact NV12 deliberately rejects padded input geometry; full vendor ABI support
is not inferred from fixture compilation. Custom artifacts require explicit model
path and contract reference while retaining board identity checks and separating
their hashes from publisher provenance.

## README and offline evaluation

Documentation explains both profiles, current commands and output arrays, X5-only
native scope, source results and exact verification limits. Chinese and English
texts preserve the source s=0.9984 versus claimed 0.999 threshold conflict, and
avoid merging differently bound latency tables into a new benchmark. Archived
sources remain provenance; the native sample guides provide the current workflow.

Evaluator documentation separates prepared RGB tensors from packed runtime NV12,
raw/log outputs from restored depth, and three preparation/restoration protocols.
Saved outputs need matching record identities and producer evidence. Metric code
and prose agree on pixel pooling, lower-median dataset alignment, average-middle
single-image alignment, invalid-GT handling and null zero-norm cosine. Optional
Torch interpolation and source float32 reductions are not advertised as newly
validated or bit-identical to default OpenCV/float64 behavior.

## Independent verification and limits

[verification.json](evidence/2026-09-28-yolo26-depth-independent-review/verification.json)
retains commands, outputs, timestamp, cwd and candidate hashes. All recorded file
hashes were checked again before this report.

- 38 host tests passed, covering profile/source behavior, target/custom-artifact
  gates, CLI outputs, offline evaluation fixtures and native contracts/resources.
- Sample checker: zero violations, one CLI policy skip, zero exemptions.
- Production yolo26_depth.cpp, image_io.cpp and main.cpp compile individually
  with real existing host OpenCV headers and C++17 -Wall -Wextra -Werror.
  These are compile-only checks, not a linked vendor executable.

No board, full vendor SDK build, optional Torch backend, dataset measurement,
weights download, export/calibration/quantization or toolchain installation was
performed. Synthetic host preparation/evaluation tests do not verify the user's
trusted quantization recipes. Roll this scoped disposition into the next
coordinated ledger/plan update after the active shared-file writer finishes;
keep those broader acceptance items and board not-run status separate.
