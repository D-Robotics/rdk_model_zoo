# B8 LaneNet — host migration and author review

Status: canonical Python/C++ source migration and six-level bilingual README set
complete for host review. Board=not-run; real native SDK/OpenCV build=not-run;
independent Review=not-run; Closed=no. This is an author verification record,
not independent acceptance. DiffusionDrive remains in B8; H0–H9 remain open.

## Source preservation and decisions

The [source audit](2026-09-26-b8-lanenet-source-review.md) pins 21 source files to
S `380e1a2bf42041af54be6f34935e50197cfadff9`. The compiler YAML and all five input/
historical output images are byte-preserved. The only published contract is
`s:lanenet:s100/lanenet256x512.hbm`, without a publisher checksum. Other targets
fail explicitly, including S100P; an explicit external file requires that exact
contract reference without claiming publisher authentication.

Input preparation preserves RGB/INTER_AREA/ImageNet arithmetic and float32 NCHW
[1,3,256,512]. Python binds required output names instance_seg_logits and
binary_seg_pred; native code binds unique float32 [1,3,256,512] and int64
[1,1,256,512] or [1,256,512] roles. Source prose mentions a third output but does
not identify it: additional observed outputs are retained without an invented
name or forced total count. Real HBM metadata has not been observed in this
migration; source-derived binding and synthetic fixtures are not artifact proof.

No source code implements clustering, lane IDs or curve fitting. The task returns
raw CHW float embeddings and strict 0/1 uint8 labels on the model grid. Source
Python's uint8 wrap/truncate display is intentionally replaced with clip/nearest
half-to-even display, consistent with the native convention. Raw embeddings are
not clipped. Source display figures remain labeled historical, not regenerated
results. Nonbinary labels are errors rather than visually plausible masks.

## Responsibilities and native lifecycle

LaneNetTask contains constructor and pre_process/forward/post_process/predict.
Image arithmetic, raw tensor validation, visualization and CLI IO are separate.
Shared NamedArrayRunner handles lazy SDK transport and copies every named output;
its compatible SingleArrayRunner adapter was committed separately in 875351d,
after the shared S64 prerequisite c807b4a. No further shared change is made here.

The native ModelRunner uses scoped ownership for model, allocations and task,
checks SDK return values, validates all byte strides (including width padding),
and copies raw outputs before task release. Dynamic input-stride handling avoids
the source's out-of-rank access. Required roles must be unique; ambiguous layouts
fail before allocation. Native auxiliary types omit float16, which Python can
preserve; this boundary is documented rather than asserted universal parity.
Destructor cleanup is best-effort; release/free failure behavior is not certified.

Native host tests compile actual tensor/IO helpers and resource owner against
minimal fake SDK headers. They cover 20 injected SDK-call failures, partial handle
assignment, gate-before-SDK behavior, padded strides, invalid layouts, output role
ambiguity and exact int64 NPY serialization beyond 2^53. The full OpenCV task and
CLI executable have not been compiled against real dependencies. Fake SDK
compilation proves host control flow, not vendor ABI compatibility.

## Entry points, preparation and evidence

Both runtime entries preserve the source optional instance/binary display flags.
New output directories are required. Python writes all raw outputs to NPZ with
an explicit SDK-name/archive-key map, plus raw embedding, binary labels, displays
and provenance. Native writes individual typed raw NPY files and metadata/role
indices; its launcher adds exact command/cwd/UTC, observed file hashes and full
stdout/stderr. If native failure occurs before result-directory creation, only
terminal streams exist; the README does not promise saved evidence in that case.
No implicit dependency install, model download, warmup or timing is added.

Conversion retains source YAML and adds explicit caller-image calibration/config
preparation. Runtime and calibration share the same pure preprocessing function.
The compile wrapper checks manifest/hash/shape/dtype/range and a nonempty resulting
HBM, preserving source nash-e/int16/latency/O2 settings. Prepare-only does not invoke
OE. A caller ONNX is checked for file presence, not graph semantics; missing source
export code, original calibration data, model revision and OE version remain gaps.
Compiler nonzero failures retain a failed report; a zero return without the HBM
raises with captured streams but no final report. No real export/OE run occurred.

## README quality and source content

Twelve README files cover root, model, Python, C++, conversion and evaluator in
English and Chinese. They retain algorithm references, five historical images,
checkpoint/source dependency information, compiler settings and the historical
200-frame HRT record (14.245 ms, 69.894 FPS), while separating source figures from
new evidence. The source evaluator was a placeholder; no dataset scoring workflow
or accuracy metric is fabricated. Missing source accuracy figure/export/calibration
scripts and unnamed third output remain explicit.

Every runtime guide documents parameter defaults, targets, actual stage IO,
output semantics, errors, scheduling and resource boundaries. The Python API
example uses the actual task/runner. Native documentation explains full SDK build
requirements and the narrower host fixture coverage. Documentation checks found
an abbreviated default-image path inconsistent with the parser; both languages
were corrected to the actual path before final contract verification.

[README verification](evidence/2026-09-26-b8-lanenet-host/check_readmes.py) checks 96
local links, 18 command examples against actual parsers with side effects stopped,
bilingual command equality and both Python API snippets through the real runner
with an injected fake SDK. The evaluator's unittest command is executed separately
as the complete sample suite. Root/sample navigation now lists 43 samples.

## Verification

- Full host regression: shared142 + ResNet52 + Ultralytics78 + OCR44 + checker27
  + LaneNet22 = **365 passed**.
- Sample contract: 0 violations; one explicit CLI policy skip; no exemptions.
- Migration scope: 43 samples, 0 violations, 44 CLI policy skips, 0 exemptions.
- README checks: 96 links, 18 parser commands, 2 API fixture executions.
- Catalog: Node22 build and typecheck pass; 57 families, 820 historical benchmark rows.
- Compiler YAML and five source images byte-identical; shell wrappers parse.

[Evidence index](evidence/2026-09-26-b8-lanenet-evidence.json) identifies current
files and raw host logs, including RED/GREEN development records. Board inference,
real SDK/native OpenCV build, OE export/compile, dataset accuracy and performance
remain **not-run**. No hardware, HP or SSH action was performed. Neither a green
host suite nor this author report closes independent review or B8.
