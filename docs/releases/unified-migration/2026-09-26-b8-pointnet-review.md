# B8 PointNet — host migration and author review

Status: host implementation complete for PointNet; independent review not-run,
Closed=no, board not-run. The other seven B8 sample families remain pending.

## Source and capability preservation

Source: `rdk_s @ 380e1a2bf42041af54be6f34935e50197cfadff9`.
Source runtime/main and chair file bytes were verified against that Git object.
All seven test-data files (chair cloud and six figures/screenshots) are copied
byte-for-byte. Original source implementation remains at `platforms/s/`; only
the two root READMEs gain a context banner before their unchanged bodies.

Canonical entry: `samples/vision/pointnet/runtime/python/main.py`.
Published asset remains `s:pointnet:s100/pointnet.hbm`; only the release manifest's
download entry now points to the new explicit `model/download.sh`. URL and unknown
publisher SHA-256 are unchanged. X5/S100P/S600 are explicitly unsupported here.
No new C++ implementation is claimed: the source has none.

## Code boundary and decisions

- Task exposes initialization plus pre_process/forward/post_process/predict.
  Normalization is in preprocessing, SDK and metadata in a lazy runner/binding,
  file IO in main, visualization in a separate module. Shared manifest reading,
  identity verification, metadata serialization and quantization are reused.
- Ruling: business input is raw XYZ; normalization no longer hides in a file
  loader. The README explicitly contrasts the old normalized-input API. No
  raw coordinates or loaded files are silently treated as normalized points.
- Per-call frozen context captures centroid/radius/count. N comes from metadata;
  no point sampling, padding or reordering. Output IDs retain point order.
- Strict input/output metadata and dtype checks precede task decoding. Float32
  outputs ignore vestigial descriptors; integer logits require valid SCALE
  metadata and are dequantized only in postprocessing before argmax.
- Real SDK loading is preceded by local target identity and artifact verification.
  Host injection is explicit. Asset hashes remain unknown where the publisher
  does not provide one; a locally observed digest is not publisher authentication.

## README review

Five levels, ten bilingual READMEs: root/model/runtime/python/conversion/evaluator.
Root provides an explicit prepare/run/result path; Python guide covers every
CLI flag/default, full executable API, stage shapes, point ordering and failures.
The source model README's false bundled-HBM claim is corrected to explicit download.

Architecture and ONNX figures, quantization screenshot, result images and all
four historical performance rows are retained. Missing recipe inputs are named:
training pin/checkpoint, exporter/environment, calibration set/config, compiler
version/march/config and metric definitions. No generic commands are offered as
reproducible export/compile recipes. Historical timing units were not supplied by
the source and are not invented. “trans/pred” values are not presented as mIoU.

46 local links resolve; 9 paired bash/Python blocks match between languages.
Both README API examples execute using the real runner with an injected SDK
fixture, not by importing a board SDK. The SDK-free CLI and both shell launchers
also work from a different working directory.

## Verification

- RED: new entry absent, the 16 initial tests fail before implementation.
- GREEN: PointNet 21 tests; actual delivered chair normalization/tensor bytes and
  postprocess match the real source helper with fixed raw output fixtures.
  Includes predict/stage equality, immutable interleaved contexts, no input
  mutation, raw-forward purity, owned runner output, target/shape/dtype failures,
  integer quantization, real CLI labels/report writing and explicit downloader.
- Repository required regressions: shared 131, ResNet 52, Ultralytics 78, OCR 44;
  combined with PointNet, 326 tests pass.
- Contract checker: PointNet 0 violations; full scope 37 samples / 0 violations /
  38 policy skips / 0 exemptions. Policy skips are CLI IO, not hidden README debt.
- Node 22 catalog build/typecheck passes: 57 families, 820 benchmark records.
- Board/model download/export/compile/current performance: not-run. The real HBM's
  tensor count, N and output dtype have not been observed; a mismatch will fail
  explicitly instead of being guessed. Synthetic metadata is not board evidence.

[Full evidence and logs](evidence/2026-09-26-b8-pointnet-evidence.json).

## Remaining work

Independent whole-branch review and later board execution remain separate gates.
Continue B8 UNet, UNetMobileNet, PP-LiteSeg, YOLO26 Depth, Depth Anything V2,
LaneNet and DiffusionDrive; do not mark B8 complete from this one sample.
