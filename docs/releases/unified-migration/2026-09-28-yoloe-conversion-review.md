# YOLOE conversion preparation — implementation self-check

Base: `71262f99c1de59f9b3059e1b9f90bf5dcda0e761`. This increment implements
canonical ONNX inspection, calibration preparation, target configuration and
explicit compilation capture. It does not close YOLOE, B9 or the H0–H9 goal.
This is an implementation self-check, not the required fresh independent review.

## Implementation

`samples/vision/yoloe/conversion/prepare.py` accepts a local self-contained ONNX,
the fixed PF vocabulary, a concrete target/variant and calibration image folder.
The real ONNX checker runs before output creation. Static RGB float32 input,
ten unique NHWC float32 output roles and the vocabulary digest are required.
External tensor data, dynamic/incorrect dimensions, integer outputs, unsupported
targets and existing output directories fail explicitly. Graph dimensions validate
the interface only; `variant_declared` is not proof of checkpoint size or semantics.

The code snapshots validated ONNX/vocabulary files, selects images deterministically,
records their decoded-byte identity and generated tensor identity, writes target
YAML and a conversion status record. Source-byte hashing uses the same bytes decoded
by OpenCV, avoiding an image reread that could bind a different file version.
X5 and S require different calibration formats/ranges according to the official
OE manuals linked from both customer READMEs: X5 raw float32 RGB 0..255, S NPY
float32 RGB 0..1. Shared runtime geometry preserves E11 truncate/127 and E26
round/114 preprocessing. Runtime NV12 scale 1/255 is separate from the S NPY
calibration input; it is not an extra normalization pass on that NPY.

Configurations cover X5 E11s/m/l, S100 E11s/E26n/s/m/l/x and S100P E26n/s/m/l/x.
They preserve source compiler/calibration policies but request no boundary-node
removal, avoiding both the stale v8 removal list in S11 and the deliberate output
dequantization removal in S26. The X5 source attention override is included only
if its exact Softmax node exists in the submitted graph. Missing nodes are exposed
as warnings. X5 11m/11l policy derives from the source 11s YAML and still requires
real compiler acceptance; host configuration coverage does not establish that.

Compilation is opt-in and never uses a shell command string. Exact argv/cwd,
UTC times, return code and complete combined output are captured. The expected
nonempty artifact is hashed. Even an exit-zero compiler produces status
`compiled_unverified`, observed output dtype null, board and accuracy not-run.
The intended `_float` filename does not certify actual compiled output precision.
No public manifest is relabelled and no S float publication is fabricated.

## Customer documentation

Two complete conversion READMEs contain source model, toolchain/target matrix,
export prerequisites, calibration, compile commands, validation, artifact layout
and known gaps. They explain all 14 configurations, the source E11/E26 export
commands, shape protocol, normalization differences with official references,
missing-attention behavior, failure codes and local model SHA selection. Root,
model and Python README pairs now link to that preparation path and distinguish
available preparation from remaining real-model/export/native work.

The source checkpoint exporters remain temporarily referenced, not accepted as
the final migrated implementation. Their canonical integration is still required.
Both READMEs disclose that no real checkpoint export or OE compile ran here.
Historical source instructions and files were preserved.

## Evidence

Full commands, UTC timestamps, return codes and logs are in
[host-results.json](evidence/2026-09-28-yoloe-conversion/host-results.json).
[result.json](evidence/2026-09-28-yoloe-conversion/result.json) binds implementation,
README and source recipe hashes, dependency versions, references and limitations.

- 445 Python tests: YOLOE 28, Ultralytics 141, shared 153, ResNet 52, OCR 44,
  checker 27. YOLOE adds nine conversion tests beyond the previous 19.
- New tests use real ONNX validation on synthetic graphs; compare calibration
  pixels; cover all 14 selections; reject malformed/external/dynamic/integer graphs,
  wrong vocabulary and unavailable compiler; preserve failure status; and exercise
  an explicitly fake compiler for success/failure recording. Fake files are never
  represented as real model artifacts.
- The preparation command is extracted from each bilingual README, substitutes
  only documented user paths/interpreter, and runs as a subprocess using synthetic
  ONNX/images. Both return config_only with no compiler. The two runtime examples
  also pass with explicit fake SDKs, preserving prior coverage.
- The 45-sample migration checker reports zero violations, 47 declared policy
  skips and zero exemptions. README validation checks 54 YOLOE local links, plus
  the existing 128 Ultralytics links and 16 executed API examples.

Host environment: Python 3.14.7, ONNX 1.23.0, NumPy 2.5.3, OpenCV 4.14.0,
PyYAML 6.0.3, SciPy 1.18.1. Additional ONNX dependencies were installed into an
ignored task-local directory, not a board or OE environment. Publisher manifests
and catalog inputs did not change in this increment; publisher checks were not
repeated. No native C++ behavior changed.

## Remaining work

Canonical checkpoint exporters, dataset evaluator, S native C++ implementation
and their complete bilingual documentation remain required. Actual checkpoint
export, OE compilation, compiled metadata inspection, dataset accuracy and board
tests are not-run. Lack of a board does not close or block the remaining host
implementation work. B9/H5/H8 and the full H0–H9 plan remain open, followed by
fresh whole-branch independent review and GitHub integration.
