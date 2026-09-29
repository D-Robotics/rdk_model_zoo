# B8 Depth Anything V2 — source contract audit

Source: S `380e1a2bf42041af54be6f34935e50197cfadff9`. The canonical migration is
in progress; source audit is not completed implementation, independent review or
board acceptance. [Capture script and pinned bytes](evidence/2026-09-26-b8-depth-anything-audit/capture.py)
and [results](evidence/2026-09-26-b8-depth-anything-audit/result.json) are reproducible
without a model, SDK, Torch or hardware.

## Preserved scope

S-only Python source; no native implementation, conversion scripts/configs or
dataset evaluator are supplied. Ten bilingual READMEs, the original furseal image
and six explanatory/result figures form the source documentation. The manifest
contains exactly one HBM: `s:depth_anything_v2:s100/depth_any.hbm`, publisher digest
unknown. Root prose also names S100P, but there is no separate S100P asset or
compatibility evidence. The source shell reads SoC then unconditionally uses
S100, installs packages and downloads a model. Canonical inference must check the
actual target, refuse missing-asset targets and separate preparation.

Source ONNX documentation records input [1,3,518,686] and output [1,518,686].
Runtime input is float32 NCHW RGB. Internal int16 quantization is not proof that
public runtime output is integer; bind the float32 IO contract explicitly and
fail on incompatible metadata. Actual HBM metadata remains unobserved.

## Numerical findings and migration rulings

1. **Normalization differs from its docstring.** Runtime calls
   `zscore_normalize_lastdim`: each RGB pixel subtracts its three-channel mean
   and divides by sqrt(three-channel variance + 1e-5), then casts to float32.
   This is not ImageNet mean/std or `/255`. The source helper's extra squeeze
   does not affect fixed 518×686 input. Preserve actual arithmetic and disclose
   the misleading comment; do not guess the inaccessible model's calibration.
2. **Default resize uses nearest.** The source helper's default interpolation
   is INTER_NEAREST for stretch. Its optional letterbox branch uses default
   OpenCV linear, floor-sized geometry and gray127 padding. Retain both input
   modes. Source postprocess blindly stretches the entire output; optional
   letterbox migration must crop padding before restoration. Default stretch
   behavior is unaffected by this correction.
3. **Constant output divides by zero.** Source min/max display normalization
   produces all NaNs on a constant float output. Preserve relative float depth
   in task results and move display normalization to a visualization helper;
   an explicit all-zero grayscale policy fixes constant maps without claiming
   metric meaning. Float64 display arithmetic avoids range subtraction overflow.
4. **Torch is used only for bilinear restoration.** Canonical OpenCV linear
   interpolation removes this board dependency. Both follow half-pixel linear
   geometry, but numerical bit identity with Torch is not claimed. Nonuniform
   analytic interpolation fixtures and future raw-array board comparisons are
   required; historical images alone are not acceptance evidence.
5. **API responsibilities are separated.** The task exposes three stages and
   predict. SDK loading/scheduling belongs to the runner; file IO, hashes,
   reports and coloring belong to the CLI/helpers. The new result is float
   relative depth, not the source's display-normalized uint8 map. Document this
   deliberate API change and retain archived source entry points.

## Documentation to preserve

Keep source algorithm description and all framework/ONNX/quantization/result
figures, with historical attribution. Retain all four HRT rows: 1/2/4/8 threads,
100 frames, total 13738.43/26375.53/52214.07/102309.64 ms, reported averages
137.38/263.74/521.90/1020.35 ms and FPS 7.27/7.54/7.54/7.54. Do not recompute
conflicting totals/averages into invented consistency. Retain monitor references
95.4% BPU, about 300 MB ION, read15920/write11650 with unspecified bandwidth units
and missing software/artifact context. These are historical, not new benchmarks.

Conversion docs contain graph facts and approximate quantization similarity
(about .999), but no original weights, export script, ONNX checksum, calibration
recipe, exact toolchain version or compile YAML. They must identify those missing
prerequisites rather than fabricate a runnable conversion pipeline. Evaluation
has no dataset, GT protocol or dataset metrics. An illustrative command must not
be called a reproduced result; visual structure is not quantitative accuracy.

No board/HP/SSH/model download/SDK execution was performed. Full migration,
documentation validation, host regression and independent review remain open.
