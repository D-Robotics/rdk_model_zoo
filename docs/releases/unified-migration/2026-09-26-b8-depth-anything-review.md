# B8 Depth Anything V2 — host migration and author review

Status: Python source capability and five-level bilingual documentation complete
for host review. Board=not-run; independent Review=not-run; Closed=no. There is no
source native implementation, executable conversion recipe or dataset evaluator.
Those gaps are documented, not filled with fabricated workflows. LaneNet and
DiffusionDrive remain in B8; the full H0–H9 plan remains active.

## Preserved source and implementation decisions

The [source audit](2026-09-26-b8-depth-anything-source-review.md) verifies 21 files
against S `380e1a2bf42041af54be6f34935e50197cfadff9`. All seven input/figure files
are byte-preserved. Exact source model identity is
`s:depth_anything_v2:s100/depth_any.hbm`; publisher SHA-256 remains unknown.
The manifest now points to explicit download.sh; download_model.sh forwards the
same flags. Source SoC positional arguments are replaced by explicit --target.

Only S100 has an asset. Source S100P prose does not establish compatibility;
S100P/S600/X5 selection explicitly fails. The SDK runner checks real board identity
before loading and validates one model/input/output, float32 RGB NCHW
[1,3,518,686] and float32 depth [1,518,686]. Source internal int16 quantization
is not evidence that its public output is integer. Real HBM metadata is still
unobserved; synthetic metadata does not prove that the artifact satisfies this
source-derived binding.

Actual input arithmetic is preserved: nearest stretch, BGR→RGB, per-pixel
three-channel mean/variance with epsilon1e-5, float64 intermediate statistics and
float32 NCHW output. Source comments incorrectly call this ImageNet normalization.
The canonical implementation and all guides disclose the actual transform.

Optional source letterbox retains floor-sized dimensions, linear resize and
fill127. Postprocessing now crops its padding before original-size restoration;
the source stretched the full padded output. Collapsed dimensions and mismatched
per-frame context fail explicitly. The default stretch path is unaffected by the
crop correction. A context travels with each prepared frame rather than being
stored as mutable last-frame state.

Source Torch was needed only for linear interpolation. Canonical OpenCV linear
uses the same half-pixel geometry, covered by an independent affine-plane analytic
fixture, but numerical bit identity with Torch is not claimed or tested. The task
returns owned float relative depth, replacing source display uint8 output;
normalization/colorization move to a separate module. Constant maps now have
zero-gray output instead of source NaNs. Float64 display normalization avoids
finite-F32 range overflow. No metric-depth assertion is made.

## Responsibilities and outputs

DepthAnythingV2Task has constructor plus pre_process/forward/post_process/predict.
Shared SingleArrayRunner owns SDK loading, scheduling and raw-array copying;
sample binding owns artifact and tensor identity; geometry and visualization are
separate helpers. The CLI owns image IO, scheduling and provenance records.
No shared code changed for this sample.

CLI outputs raw_depth.npy, depth_native.npy, depth_gray.png, depth_color.png and
report.json, including observed input/model hashes and unknown runtime values
when applicable. Optional --img-save-path retains an additional source-style
color image. Existing outputs are refused; an additional color path cannot
replace any canonical output (a discovered collision now has RED/GREEN coverage).
Priority0/core[0] are retained. No warmup, timing, install or download happens
implicitly. External model paths require exact contract identity; with no
publisher hash, that reference is explicitly not authentication of arbitrary bytes.

## README preservation and quality checks

Ten bilingual READMEs cover root/model/Python/conversion/evaluator. They preserve
source algorithm background, framework/ONNX/quantization/result/monitor figures,
all four HRT rows, approximately .999 quantization similarity and monitor values.
Each historical record is labeled, and absent dataset/software/digest/bandwidth
units are not invented. Source totals/averages/concurrency cannot be reconstructed
from available records; copied numbers are not silently corrected or called new
measurements.

The model guide explains exact asset/path/hash limitations and external copies.
The Python guide documents every flag/default, actual arithmetic, stage IO,
per-frame ownership, API change, source scheduling, output semantics and errors.
Conversion lists missing checkpoint, export, calibration, YAML and toolchain
identity. Evaluation distinguishes raw arrays, normalized displays, dataset
metrics and HRT timing without manufacturing a dataset evaluator.

[Reproducible README checks](evidence/2026-09-26-b8-depth-anything-host/check_readmes.py)
resolve 62 local links, check bilingual shell-command equality, validate 13 command
examples through actual parsers without side effects, and execute both Python
API snippets through the actual runner with an injected SDK fixture. Historical
HRT commands are not executed. Root/sample indexes now cover 42 samples.

## Verification and limits

- Sample15 tests: exact selection/rejection, metadata contracts, source arithmetic,
  letterbox cropping, raw ownership, per-call context, nonfinite/type errors,
  analytic half-pixel interpolation, constant/range visualization, stage purity,
  real identity gate, host inspection, full CLI image/NPY/report/scheduling,
  exact downloader and output-path collision prevention.
- Required regression: shared139, ResNet52, Ultralytics78, OCR44, checker27; with
  the final15 sample tests, 355 passed. Earlier full-suite log has14 sample tests;
  the subsequent output-green log is the final15 after the isolated CLI fix.
- Migration checker:42 samples,0 violations,43 CLI policy skips,0 exemptions.
- Node22 catalog build/typecheck:57 families,820 benchmark records. Existing
  source missing-dataset warnings remain.
- Shell syntax and all seven source image bytes verified. Source numerical audit,
  RED/GREEN logs, final regressions, README checks and hashes are retained in
  [host evidence](evidence/2026-09-26-b8-depth-anything-evidence.json).

No board/HP/SSH action, model download, real SDK execution, Torch comparison,
conversion or dataset measurement occurred. Host fixture success is not board
acceptance; author checks do not replace final independent whole-branch review.
