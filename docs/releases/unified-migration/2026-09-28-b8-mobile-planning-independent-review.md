# UNetMobileNet and DiffusionDrive independent host review

Reviewer: Codex. Candidate base `0addb7a5`; the two sample trees are unchanged
from the recorded test snapshots. Accept these two samples within the reviewed
host migration scope. This does not close the whole B8 batch or certify a board.
The reviewer made no product implementation changes.

## UNetMobileNet

Inspection covered the Python three-stage task, immutable per-call geometry,
SDK binding/runner, native tensor/resource contracts, CLI/visualization separation
and the bilingual root/runtime/evaluator guides. Split NV12 input geometry is
Y [1,1024,2048,1] and UV [1,512,1024,2]. Source INTER_AREA preprocessing and
19-class decoding are explicit. Integer SCALE scores use affine float64 decoding
before argmax; explicit NONE integer values retain their ordering. The API returns
original-resolution int32 IDs with nearest restoration, separately from rendering.
Native storage/stride and failure cleanup fixtures are scoped as fake SDK evidence.
Target selection distinguishes S100/S600 and rejects unsupported identities.

The guide distinguishes this original-resolution Cityscapes sample from the
fixed-grid X5 UNet sample, explains palette and overlay weights, and preserves
historical figures. Python NPY and native lossless PNG are documented separately.
No dataset runner or source mIoU/performance result is fabricated. The evaluator's
statement that independent review is pending predates this report; update that
status in the next coordinated author documentation pass, retaining board not-run.

## DiffusionDrive

Inspection covered logical feature validation, quantization helpers, transport,
three-stage task, CLI/batch handling, offline evaluator and bilingual hierarchical
documentation. Camera/lidar/status/noise remain four explicit finite float32
logical inputs. Physical conversion is metadata-driven, with bounded integer
conversion and separate output decoding. Inference does no filesystem or rendering
work. Outputs own their storage. Source clipped sigmoid, inclusive score threshold
and BEV argmax semantics remain visible and exercised by the supplied six cases.

Documentation preserves the feature-level input boundary, fixed-noise requirement,
all case data and historical S100P/S600 results. It does not present model-reference
comparison as ground-truth accuracy or full NAVSIM evaluation. Evaluator schema
checks precede arithmetic; cosine's zero-norm result is undefined/null. Successful
metric calculation is not an acceptance threshold. Saved reference/candidate
hashes do not substitute for runtime/model provenance.

## Independent evidence

- [UNetMobileNet](evidence/2026-09-28-b8-mobile-planning-independent-review/unetmobilenet.json):
  19 host tests passed; sample checker has zero violations, one CLI policy skip,
  and zero exemptions. Evidence binds 38 candidate files.
- [DiffusionDrive](evidence/2026-09-28-b8-mobile-planning-independent-review/diffusiondrive.json):
  23 host tests passed; sample checker has zero violations, one CLI policy skip,
  and zero exemptions. Evidence binds 32 candidate files.

Full commands, output, timestamps and cwd are retained. The reviewer rechecked
all 70 recorded file hashes against the working tree before writing this report;
none changed. Tests cover meaningful source arithmetic, invalid metadata/schema,
CLI boundaries and native injected resource failures as indicated in their logs.
They do not establish real vendor SDK linking or board parity. No toolchain
installation, weights download, export, calibration, quantization run or hardware
connection was performed. Trusted conversion recipes were reviewed as documents.

Roll these scoped dispositions and the pending customer status wording into the
next coordinated plan/ledger/documentation update after the MiniCPM worker releases
shared files. Existing historical board results remain historical; new board,
performance and dataset checks remain not-run. Other B8 samples and the complete
README/source-depth audit retain their own open acceptance requirements.
