# B8 YOLO26 Depth — host migration and author review

Status: host implementation and six-level bilingual documentation complete.
Board=not-run; real native SDK/OpenCV build=not-run; Torch export/OE compilation
and dataset measurement=not-run; independent Review=not-run; Closed=no.
Depth Anything V2, LaneNet and DiffusionDrive remain in B8. H0–H9 remains active.

## Source scope and preserved capabilities

The [source audit](2026-09-26-b8-yolo26-depth-source-review.md) pins X5
`ac115717197920355fc390bb04299b20e6436864` and S
`380e1a2bf42041af54be6f34935e50197cfadff9`, covering 81 source files, 20 assets and
29 conversion configs. The canonical sample includes Python for X5/S100/S100P/S600
and source-supported X5 C++. Each target has n/s/m/l/x; default n is retained.
X5 publisher hashes are preserved; S publisher hashes remain unknown.

All 29 YAMLs are byte-identical copies. The original X5 mapper and shared source
SUNRGBD extractor remain available, while the canonical conversion wrappers make
paths and tensor boundaries explicit. Source requirement lists remain separately
identified; the small host requirement file is not an export-toolchain lockfile.
The bus image is retained. Source trees, licenses and historical measurements
remain available; no source branch is represented as newly tested.

## Runtime contracts and intentional fixes

- X5 all variants and S n/s/m use one packed NV12 uint8 input, 768 letterbox with
  fill 114, and calibrated float32 log-depth output at 192 square. S l/x use RGB
  float32 NCHW scale-fill and raw logits; only those receive CPU clip/scale/bias.
  All variants apply exp and original-size restoration on CPU. Source prose
  claiming graph-internal exp/resize conflicts with source implementation.
- Yolo26DepthTask exposes preprocessing, forward, postprocessing and predict.
  SDK ownership/binding, geometry, rendering, IO and timing are separate modules.
  Forward returns raw output without activation. Per-call geometry travels with
  its tensors; input/output values, exact layout and context are checked.
- The S source evaluator's raw-lite-only assumptions cannot evaluate a calibrated
  NV12 output. Canonical offline evaluation requires an explicit boundary and
  protocol, sharing restoration with runtime. The audit preserves a concrete
  log-zero counterexample instead of claiming unconditional source parity.
- Native geometry now follows Python ties-to-even rather than C++ round-away;
  collapsed dimensions fail explicitly. Output copying honors valid/aligned
  shapes and byte strides with capacity/overflow checks. Unsupported padded
  input geometry fails rather than assuming a compact tensor.
- The SDK owner uses scoped cleanup for models/buffers/tasks, including partial
  allocation and failed inference paths. Thirteen injected SDK failure points
  are checked against the actual owner implementation with fake SDK interfaces.
  These fixtures do not establish the vendor SDK ABI or native image parity.
- Identity checks precede production SDK loading and launcher builds. Published
  X5 models retain hash verification; explicit user-converted mode binds only a
  declared tensor contract and reports different provenance. No silent S target
  fallback or implicit download remains.

Python defaults to three warmups and S scheduling priority 0/core [0]; native
warmups remain zero and SDK scheduling defaults are retained from source.
Measured forward times include validation/copies; no isolated BPU latency claim
is made. Old shell positional Python variants now use explicit --variant; native
three-positional invocation remains supported. New native NPY outputs accompany
the source-compatible raw F32 depth file for reproducible offline consumption.

## Conversion and evaluator

Export patches retain the source attention/depth changes, enforce exactly one
Depth head and record checkpoint calibration. Target-aware log/lite filenames
replace the source filename/config mismatch. Calibration explicitly emits X5
CHW uint8 binaries or S NCHW float32 arrays, validates image geometry and writes
file digests. Compilation validates the selected template, calibration profile,
shape/dtype/range, hashes and exact file set, then invokes the matching toolchain;
missing artifacts fail even if the external tool returned zero. Prepare-only
mode does not validate an ONNX graph or execute the toolchain.

Offline evaluation preserves deployment letterbox, deployment scale-fill and
source validator protocols. It requires matching candidate/reference record IDs,
checks finite values and handles empty valid-GT/zero-norm cases explicitly. Dataset
metrics use lower-median per-image alignment and float64 pixel pooling; the
single-image numeric comparison preserves the source NumPy median behavior.
Runtime-default OpenCV restoration is fixture-checked; optional Torch resize is
not run. Arrays' protocol and boundary remain caller-declared, not independently
proven by a saved NPZ. Custom weights need matching calibration and a new binding
if their contract changes. No generated model or dataset score is claimed.

## Documentation and evidence

Twelve README files cover root/model/Python/C++/conversion/evaluator in both
languages. They include all 20 paths, publisher-hash limitations, exact flags and
parser defaults, target-specific geometry, stage/API examples, lifecycle,
conversion prerequisites, saved-array schema, metrics and historical tables.
The S table's 0.9984 conflicts with source prose claiming all results ≥0.999;
this discrepancy is disclosed rather than edited into agreement. Source HRT
and other timing records remain historical and are not comparable to the new
whole-forward timer without a controlled measurement.

The reproducible [README check](evidence/2026-09-26-b8-yolo26-depth-host/check_readmes.py)
resolves 64 local links, checks paired shell commands, sends 34 command examples
through actual parsers without executing their side effects, and executes both
Python API examples with the real runner and injected SDK fixture. CMake/native
examples remain instructions, not evidence of a real SDK build. Shell syntax and
29 exact config copies were also checked. Root navigation now includes 41 samples.

Host results:

- Sample: 38 tests (runtime 12, CLI 5, conversion 7, evaluator 8, native 3,
  launcher 3), including source numerical fixtures, custom-artifact hash/gate
  behavior, actual image/NPY/report writes, padded output copies, cleanup failure
  injection, native NPY serialization and launcher provenance.
- Required regressions: shared 139, ResNet 52, Ultralytics 78, OCR 44, checker 27;
  378 tests including this sample, all passed.
- Migration checker: 41 samples, 0 violations, 42 explicit CLI policy skips,
  0 exemptions. No baseline is used.
- Node22 catalog build/typecheck: 57 families, 820 benchmark records. Existing
  missing-dataset warnings remain source metadata gaps.

[Host evidence and raw logs](evidence/2026-09-26-b8-yolo26-depth-evidence.json)
retain verification results and file hashes. No board, HP, SSH, model download,
Torch export, real OE build or dataset evaluation occurred. These author checks
are not independent acceptance and do not close B8 or the overall migration.
