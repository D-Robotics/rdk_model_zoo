# B8 UNetMobileNet — host migration and author review

Status: Python/C++ source migration and six-level documentation complete for host
review. Board=not-run, real SDK build/link=not-run, independent Review=not-run,
Closed=no. B8 still has YOLO26 Depth, Depth Anything V2, LaneNet and DiffusionDrive;
the full H0–H9 plan remains active.

## Preserved capability and deliberate corrections

Source: S `380e1a2bf42041af54be6f34935e50197cfadff9`, audited across 17 files in
[source evidence](evidence/2026-09-26-b8-unetmobilenet-audit.json). Both S100/S600
HBM identities/URLs remain separate with unknown publisher SHA-256. The manifest
now names explicit download.sh; download_model.sh remains a forwarding spelling.
Both source images are byte-identical. The original Chinese README body is retained
under platforms/s with a leading canonical-entry banner. No source conversion or
dataset evaluator implementation existed, so none is invented.

Python and C++ retain INTER_AREA stretching to 2048×1024, Y/UV uint8 planes,
19-class semantics, original-resolution labels and the original rdk_colors palette.
alpha_f continues to weight the original image. Source Python defaults to core [0];
source C++ uses any core. These distinct scheduling defaults are preserved.

Corrections are explicit, rather than described as unconditional source parity:

- predict now returns labels; coloring, blending and file IO belong to the CLI/
  visualization layer. Old public wrappers remain available in the source snapshot.
- The source raw int32 argmax assumption is invalid with unequal channel scales.
  SCALE is validated and affine-decoded only in postprocess, with float64 comparison
  preserving adjacent int32 differences; explicit NONE int32 compares directly,
  F32 remains raw. Exact ties select the lowest ID. Absent integer quantization or
  malformed SCALE descriptors fail during binding, including missing zero_point
  fields (an explicitly empty zero_point means symmetric quantization).
- Native postprocess restores directly to original size, matching Python. The old
  intermediate resize can change nearest-neighbor labels for non-divisor output
  dimensions; the source audit retains a concrete counterexample.
- Unknown identity/S100P no longer falls back to an S100 artifact. Preparation is
  explicit, independent of inference, and cannot substitute for actual board identity.

Actual artifact output dimensions/quantization descriptors have not been observed
in this migration. Bindings validate the source geometry/semantics against runtime
metadata; synthetic fixtures are not evidence about the downloaded HBM.

## Native responsibilities and resource safety

UnetMobileNetTask contains constructor plus pre_process/forward/post_process/predict.
ModelRunner owns SDK/model/tensor/task lifetimes; tensor_contract.cpp handles
SDK-independent byte strides, capacity, affine decoding and nearest restoration.
visualization.cpp owns coloring, main.cpp owns arguments and output files.

The native owner validates rank, dtype, Y/UV order/geometry, byte strides, capacity
and quantization before allocations, then frees only acquired resources on partial
initialization or forward failures. A per-call task guard releases inference tasks.
Input dynamic row strides resolve for the selected S target; unsupported layouts
fail instead of assuming compact buffers. Returned raw bytes and metadata are owned.

The launcher checks identity, prepared model and image before CMake or execution;
CMake requires an explicit s100/s600 target. Native binaries also check the compiled
target and local identity, including the S100P board_type refinement. No implicit
package installation/download or unknown-SoC macro default remains. The runner's
explicit execution-gate callback is a host-test seam, not a CLI bypass.

## README quality

Twelve bilingual READMEs cover root/model/Python/C++/conversion/evaluator. Root has
an executable preparation/run path, support versus validation matrix, exact outputs,
source references and the preserved historical figure. Model documents target
subdirectories and unknown digest limitations. Runtime guides document every CLI
option/default, scheduling, original-size outputs, API changes, stage IO, ownership,
errors and source divergences. Python API examples execute under a real runner with
an injected SDK fixture. Native lifecycle and API wiring refer to actual symbols.

C++ PNG labels are uint8 on disk while its API remains int32; Python NPY labels are
int32. Documentation explains integer-array comparison before visualization rather
than using JPEG pixels as numerical evidence. Conversion and evaluator pages list
specific missing training/export/calibration/toolchain/dataset prerequisites instead
of offering generic recipes as verified commands. No source mIoU/FPS table exists,
and the old result image is explicitly historical.

All 40 local README links resolve; 8 bilingual executable blocks match. Root and
sample indexes now cover 40 canonical vision samples. The old source prose and
images remain available, with the current entry clearly signposted.

## Verification

- UNetMobileNet: 19 host tests, including nonuniform source score maps/restoration,
  source NV12 and palette parity, per-call geometry, SCALE ordering counterexample,
  large-int32 precision/ties, bad inputs/metadata, both exact assets, missing
  quantization field rejection, real CLI image/NPY/JSON outputs, explicit downloader,
  executable README APIs, host-safe Python/native launcher modes and pre-build gate.
- Two native test programs are compiled with C++17 and -Wall/-Wextra/-Werror by the
  suite: pure padded-score decoding/identity cases; actual ModelRunner code against
  intentionally minimal fake OpenCV/SDK interfaces. The latter injects 17 failure
  points plus a successful inference and checks task/buffer/model cleanup.
- Regressions: shared139, ResNet52, Ultralytics78, OCR44, PointNet21, UNet20,
  PP-LiteSeg18, checker27; 418 tests including this sample.
- Node22 catalog build/typecheck passed: 57 families / 820 benchmark records.
- Migration contract: 40 samples / 0 violations / 41 CLI policy skips / 0 exemptions.
  Shell syntax checks passed. Initial missing implementation, native compile RED,
  missing quantization field RED and final logs are retained with source hashes in
  [evidence](evidence/2026-09-26-b8-unetmobilenet-evidence.json).

The fake interfaces validate control flow against this implementation, not the real
SDK ABI. Full native OpenCV/DNN/UCP compilation/linking and the actual native image
pipeline were not run. No board/HP/SSH action, model download, export/compile,
dataset score or performance run occurred. These author checks do not constitute
independent acceptance or close the migration batch.

Whitespace audit reports only the raw catalog log's trailing blank line; evidence
bytes are retained rather than normalized after capture.
