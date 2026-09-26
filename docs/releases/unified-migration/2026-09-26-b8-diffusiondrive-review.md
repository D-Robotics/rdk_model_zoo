# B8 DiffusionDrive — host migration and author review

Status: host implementation and six-level bilingual documentation complete for
review. Board=not-run; real SDK/OE=not-run; independent Review=not-run; Closed=no.
B8 now has host implementations for all listed samples, but its acceptance and
the full H0–H9 objective remain open. This record is author verification only.

## Source and capability preservation

The [source audit](2026-09-26-b8-diffusiondrive-source-review.md) pins 39 files to
S `380e1a2bf42041af54be6f34935e50197cfadff9`, including twelve finite-float NPZ
archives. The sample is S-only/Python-only; no source C++ capability is discarded.
Both S100P/nash-m and S600/nash-p manifest identities and SHA-256 are preserved.
S100/X5 execution is rejected. Auto selection no longer falls back to S600 on an
unknown host; physical board and published-file gates precede real SDK loading.

Source capability maps to canonical responsibilities:

| Source capability | Canonical location / behavior |
| --- | --- |
| Four-input quantization and four-output decoding | Task stages plus quantization.py and model_binding.py |
| hbm_runtime loading/scheduling/transport | Thin local model_runner over shared NamedArrayRunner |
| Camera/BEV/LiDAR/trajectory/agent rendering | visualization.py returns canvas; CLI owns encoding/writes |
| Single-case flags and outputs | main.py, source aliases retained; new result-directory/provenance contract |
| Five-case script | run_cases.py/run_all_cases.sh reuse single-case CLI; stop-on-failure record |
| Offline float comparison | evaluator/compare_outputs.py with strict schemas and JSON evidence |
| Target HBM download/checksums | Explicit downloader + source SHA256SUMS; runtime never downloads |
| Two PTQ YAMLs | Exact source copies; missing export/calibration chain documented |
| Six input/reference pairs and six figures | Exact byte copies under test_data; bilingual case guide |
| Source README tables, references and caveats | Twelve bilingual root/model/runtime/conversion/evaluator/test_data READMEs |

Preservation checks cover 21 unchanged data/config/checksum files. Source
reference outputs are never replaced with newly computed values to pass tests.

## Binding, numeric semantics and shared changes

Logical input contract: finite float32 camera [1,3,256,1024], lidar [1,1,256,256],
status [1,8], noise [1,20,8,2]. Output roles: trajectory [1,8,3], agent_states
[1,30,5], agent_labels [1,30], bev_semantic_map [1,7,128,256]. Exact names and
shapes are required, independent of order. Supplied float references do not prove
physical HBM types; actual HBM metadata has not been observed in this migration.

Shared prerequisite e4578c4 adds input_quants without shifting old positional
metadata fields, preserves SDK descriptor identity and projects evidence without
deepcopy; U16/U32 spelling normalization does not widen other samples' allowlists.
Its 45-suite/1177-test regression is separately retained in
[evidence](evidence/2026-09-26-b8-diffusiondrive-metadata/result.json).

Physical source types remain supported, including unsigned16/32 and float16/32.
Integer tensors require validated positive finite SCALE metadata and integral
in-range zero points. Inputs retain per-tensor quantization; output per-axis
scales require matching axes. Scalar zero points broadcast correctly, fixing the
source reshape error. Rounding preserves source float32 arithmetic; final integer
clipping uses float64 bounds to prevent int32/uint32 upper-bound overflow. Floating
NONE descriptors may carry zero placeholders and pass through, preserving source
behavior; a RED test caught the initial overly strict rejection before delivery.
Nonempty floating SCALE descriptors retain the source affine behavior.

The task has constructor plus pre_process/forward/post_process/predict only.
It preserves caller noise, source clipped sigmoid [-60,60], probability >= threshold
(default0.5) and BEV argmax axis1. It returns owned arrays and does not download,
write, render, schedule or command hardware. Source __call__ alias is replaced
with explicit predict, documented with a runnable API example.

## Runtime and evaluator evidence

CLI outputs physical_inputs.npz, raw_outputs.npz, outputs.npz, result.png and
report.json with input/model/output digests, full actual metadata, scheduling,
threshold and UTC. Times are not called latency measurements. Optional source
output arguments produce extra copies/images while retaining canonical evidence.
New destinations are required. A discovered PNG-bytes-under-JPEG-name issue was
fixed: image extension is validated before SDK use and encoding matches extension;
RED/GREEN evidence checks JPEG magic and early rejection.

Batch dispatch validates all five feature files, reuses canonical CLI arguments,
loads per case as the source did, stops on first failure and records completed
return codes/available report hashes and remaining cases. The old positional
batch-output argument becomes --output; source environment overrides and implicit
downloads are removed. Inspection never imports the SDK or executes a model.

The evaluator rejects extra/missing names, wrong shapes/dtypes, nonfinite values,
invalid probabilities/classes and labels inconsistent with candidate argmax.
This prevents the source's wrong-rank broadcasting false agreement. Cosine with
either zero norm is null/undefined; valid metrics use float64 accumulation and
are descriptive without an inferred pass threshold. Agent masks are validated
but not scored without threshold context. Pixel agreement, class distributions,
per-class IoU and macro present-union IoU retain the source intent. Reference
predictions are not annotation truth, and no NAVSIM PDM Score is claimed.

## README and conversion quality

All twelve guides are written against current code and source content. The root
keeps algorithm context, upstream links, all source overview images and direct
links to historical tables. Model documents exact paths/hash gates and target
identity. Runtime lists every parser default, source aliases/changed defaults,
all outputs, stage IO, quantization details, API ownership and troubleshooting.
Test_data retains six cases, five-scene table, coordinate convention, palette,
fixed-noise policy and rare-class macro-IoU caveat; evaluator preserves the full
S100P/S600 accuracy/performance table and five-case S100P means.

Conversion retains both configs and OE3.7.0 image, graph-wide INT16/max rationale,
GridSample INT8 exception, INT8/head-only numerical comparisons and historical
HRT commands. Valid quantized input_file is included in performance examples,
with its preparation prerequisite explicit. Missing export rewrites/checkpoint
revision, original >=100 real calibration samples and profiling binaries are
not fabricated. Published-file hash gates mean a custom HBM needs new explicit
asset/binding integration; renaming or overwriting a checksum is not validation.

[README checks](evidence/2026-09-26-b8-diffusiondrive-host/check_readmes.py) verify
100 local links (including explicit anchors), bilingual command equality,
17 actual-parser commands without side effects and both Python API examples
through the real runner with an injected SDK. Nine cd/compiler/HRT command lines
are retained and labeled not executed, not silently included as parser proof.

## Host verification and remaining scope

- Required regression: shared144, ResNet52, Ultralytics78, OCR44, checker27 passed.
- Sample initially22 passed in that regression; final23 passed after the floating
  NONE/zero-placeholder correction. Combined final tested scope: **368 tests**.
- Sample contract 0 violations; migration44 samples /0 violations /45 CLI policy
  skips /0 exemptions. Root/sample indexes now contain44 unified vision samples.
- Node22 catalog build/typecheck passed:57 families,820 historical benchmark rows.
- Source copies21 exact; shell syntax passed; six actual-source postprocess/render
  comparisons use the supplied float cases, not synthetic expected screenshots.

[Evidence index](evidence/2026-09-26-b8-diffusiondrive-evidence.json) binds current
files and raw logs, preserving initial failures as development evidence. No
model download, real SDK/OE, board/HP/SSH, actuation, dataset accuracy or performance
run occurred. No independent review is claimed. Next scope is B9 family
consolidation/YOLOE, followed by B10/B11 and full shared/documentation/integration
work; do not mark the overall migration complete from this sample result.
