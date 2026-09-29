# B8 YOLO26 Depth: source contract review

Status: source audit complete; canonical implementation pending. This is an
implementation preparation record, not an independent acceptance review.
Board, real SDK build, export, OE compilation and dataset evaluation are not-run.

## Source identity and reproduction

All 81 files under the X5 and S source sample match the pinned source bytes:
X5 `ac115717197920355fc390bb04299b20e6436864`,
S `380e1a2bf42041af54be6f34935e50197cfadff9`.
The manifest supplies five X5 BINs with publisher hashes and fifteen S HBMs
without publisher hashes. S100P has five actual published entries in this family;
its support must not be copied from the unsupported S100P status of another sample.

Run from the repository root with NumPy, OpenCV and PyYAML installed:

```bash
python docs/releases/unified-migration/evidence/2026-09-26-b8-yolo26-depth-audit/capture.py
```

This checks source bytes against Git, reads manifests and all 29 YAMLs, and
executes selected source definitions without importing a board SDK. The evaluator
counterexample uses a constant depth map and an explicitly identified constant
resize stand-in, so it requires neither Torch nor an actual inference result.
The output is [the structured audit](evidence/2026-09-26-b8-yolo26-depth-audit.json).

## Findings that affect migration

| ID | Evidence | Required treatment |
|---|---|---|
| YD-A1 | S prose claims `exp → resize4x` inside the graph. Both exporter and runtime put those operations on CPU. | Use calibrated log-depth for X5 all variants and S n/s/m; use raw logits with CPU calibration for S l/x. Do not present uninspected HBM metadata as observed. |
| YD-A2 | S source table lists s cosine 0.9984, below its stated 0.999 acceptance threshold, while claiming all pass. Other source tables use different numbers and a lite-n perf example. | Keep historical tables and provenance, disclose conflicting profile/acceptance claims, and make no new pass claim. |
| YD-A3 | S exporter names log ONNX `_op11_log`, nine configs expect `-log`; all 24 configs point at `./calibration`, docs produce two differently named directories. Documented extraction flags do not exist. | Make emitted names and compile inputs agree; execute CLI examples and check every config. Keep nine experimental lite n/s/m recipes distinguished from fifteen published profiles. |
| YD-A4 | S evaluator consumes saved arrays, does not run HBM, and only implements lite scale-fill. For calibrated s log-depth zero, the source evaluator returns 5.555807 instead of 1.0. | Merge offline evaluation with explicit boundary and geometry selection. Retain X5 deployment-letterbox and validator protocols and its numerical comparison tool. |
| YD-A5 | Source letterbox rounds a dimension to zero for a 1×10000 image. C++ rounds 2.5 to 3 while Python rounds to 2. | Clear rejection of collapsed dimensions; define Python/native rounding consistently and test tie geometry. |
| YD-A6 | X5 native task combines SDK ownership with task logic; flat input copying and contiguous output reading do not validate physical layout/capacity. Native color percentiles differ from NumPy. | Separate owner, tensor validation, task stages and rendering; handle or reject padding explicitly; preserve lifecycle cleanup. |
| YD-A7 | S calibration comments assert direct RGB tensors exactly equal NV12 runtime conversion. | Describe intended range/geometry agreement, not byte equality across chroma subsampling and color conversion. |

## Implementation contract

- Default variant remains n; all five variants exist on X5, S100, S100P, S600.
  Only X5 has a source C++ implementation to migrate.
- X5 all variants and S n/s/m use 768-square letterbox, linear interpolation,
  padding 114, and one flat packed NV12 buffer. Do not replace this S input with
  the split Y/UV contract of UNetMobileNet.
- S l/x use 768-square scale-fill, RGB float32 NCHW `/255`; CPU applies clip
  `[-4,5]`, scale 1 and bias `-0.2498779296875` / `-0.316650390625`, then exp and
  direct linear restoration. Those constants are source checkpoint assumptions,
  not valid for arbitrary retrained weights.
- Export output shape is `[1,192,192,1]`; actual artifact descriptors must be
  checked at load. Output means relative depth, not calibrated metres.
- Keep only pre-process, forward, post-process and predict in the task class.
  Asset selection, metadata validation, SDK ownership, scheduling, warmup/timing,
  serialization and visualization belong in their respective modules.
- Consolidate the identical SUNRGBD archive extractor and image. Preserve all
  meaningful preparation, conversion, metric/alignment/fidelity and native
  capabilities, including experimental configurations with explicit labels.
- Deliver six levels of bilingual README: root, model, Python runtime, C++
  runtime, conversion and evaluator. Commands must match real parsers and paths;
  historical source performance must be separate from new verification.

The canonical sample, its regression tests, customer README files and migration
acceptance are still pending. No sample count or Closed flag changes in this audit.
