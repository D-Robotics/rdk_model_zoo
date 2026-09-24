# YOLOv5 evaluator

<a id="dataset"></a>
## Dataset

The source supplies `test_data/bus.jpg` for X5 and `test_data/kite.jpg` for S, plus `coco_classes.names`; there is no labeled benchmark harness in this sample. The evaluator compares one complete source/unified run on the same image, target, artifact, and thresholds. It is a consistency evidence tool, not an mAP evaluator.

<a id="environment"></a>
## Environment

Run directly on a recognized target board with Python, NumPy, OpenCV, and the target `hbm_runtime`; the evaluator also needs the fixed source runtime import path. It is not a separate host that drives the board. The host unit tests inject a fake runtime and do not certify hardware or a board result. No model or image is downloaded by the evaluator.

<a id="command"></a>
## Evaluation command

From the repository root, after preparing the exact model and using a recognized board, choose a new empty evidence directory:

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

The utility runs source and unified paths, stores complete native input/output/result arrays as `.npy`, records metadata, code/model/image hashes, thresholds and board identity in `comparison.json`, and returns `0` only when every declared comparison passes. The output directory must not already exist. This migration did not run it on a board.

### Native C++ source/unified comparison (board)

The Python command above drives the Python runtimes and cannot serve as final
evidence for the C++ deliverables, whose preprocessing may differ. The native
comparison under `evaluator/native/` runs the FIXED C++ sources themselves
(X5 `main.cc` at `ac115717197920355fc390bb04299b20e6436864`, S
`src/yolov5.cpp`/`src/main.cpp` at `380e1a2bf42041af54be6f34935e50197cfadff9`,
SHA-pinned and fail-closed) as an instrumented copy that only adds read-only
observation points; the original preprocessing, inference, decoding and NMS
are untouched, and the unified decoder is never used as the legacy side. Each
side runs its own complete inference on the same board, model and image.

On the target board, with the repo checked out and the model downloaded
(X5 default: `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` with
`test_data/bus.jpg`; S default: the x-672 asset with `test_data/kite.jpg`;
identical thresholds on both sides):

```bash
# 0) The pinned snapshots must exist locally. A shallow clone (the usual board
#    checkout) does not carry them; fetch them first (instrument.py fails
#    closed with this same hint otherwise):
git fetch --depth=1 origin ac115717197920355fc390bb04299b20e6436864   # x5
git fetch --depth=1 origin 380e1a2bf42041af54be6f34935e50197cfadff9   # s100/s600

# 1) Generate the instrumented fixed-source build (fail-closed on SHA/anchor
#    mismatch; audit in instrumentation-audit.json — the S build closure
#    (yolov5.hpp + utils/c_utils) is pinned and copied into the work dir, so
#    no mutable working-tree file enters the build). X5 rebinding example:
python3 samples/vision/yolov5/evaluator/native/instrument.py \
  --target x5 --repo-root . --work-dir /tmp/yolov5-fixed-src \
  --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image-path samples/vision/yolov5/test_data/bus.jpg
#    (S passes --target s100 or s600; rebinding happens through gflags at run.)

# 2) Build the instrumented fixed source, then run it THROUGH the external
#    runner, which records the REAL process evidence (true argv including
#    gflags-stripped arguments, separate stdout/stderr, the actual exit code,
#    start/end UTC, cwd, board identity, pre/post hashes of the binary,
#    model and image, and the instrumentation-audit verification). The C++
#    observer only records what cannot be seen from outside the process
#    (tensor payloads and stage metadata) — it no longer tees stdout/stderr
#    or pretends to know the process exit code. The capture directory must
#    be EMPTY; an in-progress marker survives any crash and the comparison
#    rejects it.
cmake -S /tmp/yolov5-fixed-src -B /tmp/yolov5-fixed-src/build && \
  cmake --build /tmp/yolov5-fixed-src/build -j2
python3 samples/vision/yolov5/evaluator/native/run_capture.py \
  --binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --capture-dir /tmp/yolov5-source-capture \
  --model <model file> --image <image file> \
  --audit /tmp/yolov5-fixed-src/instrumentation-audit.json \
  -- [--model_path <m> --test_img <i> --label_file <l>]   # S flags, kept verbatim

# 3) Run the unified binary THROUGH THE SAME RUNNER (--role unified: full
#    process evidence for the unified side too — real argv/exit code and its
#    own binary/model/image hashes; no instrumentation audit on this role).
#    Build with -DYOLOV5_TARGET=s100/s100p/s600 matching the board.
python3 samples/vision/yolov5/evaluator/native/run_capture.py --role unified \
  --binary <build>/yolov5_cpp --capture-dir /tmp/yolov5-unified-run \
  --model samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image samples/vision/yolov5/test_data/bus.jpg \
  -- --target x5 --test-img samples/vision/yolov5/test_data/bus.jpg \
     --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
     --dump-dir /tmp/yolov5-unified-dump

# 4) Compare the two same-board runs (the --target must equal the unified
#    build_target exactly — an s100 comparison rejects s100p/s600 builds;
#    compare an s600 build with --target s600; BOTH sides need their runner
#    records):
python3 samples/vision/yolov5/evaluator/native/compare_native.py \
  --target x5 --repo-root . \
  --source-capture /tmp/yolov5-source-capture --unified-dump /tmp/yolov5-unified-dump \
  --source-binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --unified-binary <build>/yolov5_cpp \
  --unified-run-record /tmp/yolov5-unified-run/run-record.json \
  --output /tmp/yolov5-native-comparison
```

The comparison verifies BOTH sides' runner records (real exit code, argv,
cwd, UTC, pre==post binary/model/image hashes, and the unified side's own
record bound to the manifest binary hash) against the observer's
capture-time hashes, the attested binaries and the unified manifest's own
hashes and payload digests before any numerical stage. Board identity uses
the repository's EXACT alias registry (docs/release/platforms.json, the same
contract as samples/_shared/platforms.py): S100P is a distinct target from
s100, unknown strings such as S100Whatever are not identity, and X5 boards
resolve through their socinfo names (X5U/X5H/X5M). A failed instrumentation
audit never runs the source binary at all — run_capture.py aborts first and
persists the failure record; the audit check validates the full protocol
(schema, target, pinned commit, the exact pinned source set with every
anchor matched once, the complete per-target closure, the observer header
and CMake hashes). Structural requirements are enforced on BOTH sides —
the target-appropriate input count, exactly three uniquely-shaped output
heads and a complete shape bijection, matching dtypes and quantization
kinds/axis/scale lengths (never masked by a float cast), finite values, and
mandatory per-payload size+SHA on the unified manifest. Layout decoding
additionally rejects channel strides that overlap (stride[3] < itemsize) or
misalign (stride[3] not a multiple of itemsize) elements and pixel strides
below channels*stride[3], while still decoding the real supported families
(including channel padding at an aligned stride). Thresholds and scale
descriptors compare by float32 BITS (native semantics): 0.45f serialized as
0.44999998807907104 and the manifest string "0.450000" are the same value;
different bits fail. Final-image coordinates (detections_original) are
REQUIRED from both sides — a missing or one-sided capture blocks acceptance
rather than passing with a disclaimer. Layout decoding accepts only the two
proven families (uniform-pitch strided, exact-size compact) and rejects
anything else instead of guessing.

The comparison restores logical arrays from each side's recorded physical
layout (strides/dtype), so padded layouts compare correctly while
uninitialized padding bytes are preserved in the evidence (`originals/`) but
never claimed byte-equal. Fixed criteria, never adjusted per run: inputs
exact; raw outputs `allclose(atol=1e-5, rtol=0)`; scale/zero-point
descriptors exact; boxes `atol=1e-4`, scores `atol=1e-5`, class ids exact,
after a declared order normalization (both sides sorted by class_id, score,
x1..y2). Detections are compared in MODEL input space AND, mandatorily, in final
original-image coordinates (`detections_original` from both sides — the
capture emits it and the unified dump since this round); a missing or
one-sided final-coordinate capture fails the whole comparison rather than
passing with a disclaimer. Any missing material, nonzero run, model/image
hash or threshold mismatch fails nonzero with the gathered evidence
preserved; a native failure can never pass as empty arrays. **Board status: not-run in this migration; the
coordinator runs these steps on the real boards.** Host tests cover
instrumentation generation (including the pinned closure and the shallow-
clone preparation hint), stub compiles of the instrumented sources (which
prove only that the injected glue compiles, not that a real SDK build
passed), observer roundtrip precision/marker/refusal behaviour, comparison
against the REAL v2 board-manifest schema, stride restoration and every
failure mode.

<a id="metrics"></a>
## Metrics

Inputs are exact; raw output arrays use shape/dtype checks and `rtol=0, atol=1e-5`; result boxes use `atol=1e-4`, scores `1e-5`, and class IDs exact. X5 and S must be compared using their own source protocol; a host fake fixture is not a board result. Historical performance is listed below and is not a new measurement.

<a id="outputs"></a>
## Outputs

Each run directory contains `legacy_*` and `unified_*` `.npy` arrays plus `comparison.json`. Failed preload or mismatch runs retain an error/failed record and return nonzero; no mismatch is converted to pass. The arrays preserve all captured input/output tensors and decoded result fields.

<a id="reference-results"></a>
## Reference results

The complete source historical X5 table is retained below; it was not re-run:

| Model | Size | Params | BPU throughput | Python post-process |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

S100/S600 and current source/unified board comparison are `not-run`.

<a id="boundaries"></a>
## Boundaries

This evaluator does not download models, build conversion artifacts, or claim board compatibility from host tests. X5 source intentionally uses its OpenCV XYXY-to-NMSBoxes quirk while S uses class-wise XYXY NMS; cross-target equality is not a valid assertion.
