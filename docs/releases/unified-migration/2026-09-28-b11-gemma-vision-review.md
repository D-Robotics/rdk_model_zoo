# B11 Gemma Vision stages and application IO

Base `1b625a3b`; source S `380e1a2bf42041af54be6f34935e50197cfadff9`.
Author implementation increment; B11/Closed remain open.

## Responsibilities

- `gemma4_image_io`: disk decoding into BGR pixels, outside the model stages.
- `PreprocessImage`: nonempty 2D CV_8UC3 to owned float RGB patches; unchanged
  bicubic 960×672 resize, divide-by-255 and row-major 16×16 patch/channel order.
- `ForwardVision`: validates explicit prepared input and calls the injected runner once.
- `PostprocessVision`: validates finite [280,1536] features and returns an owned value;
  no extra normalization or scale is introduced.
- `PredictVision`: composition only; no files, logging, SDK construction or hidden state.

Interactive `main` and single-shot VLM now load images in the application layer and use
that composition. `VisionEngine::Infer` consumes patches rather than a filename.
CMake includes the new sources. Native tokenizer/text/KV implementations are unchanged.

Ruling: replace the source path-based VisionEngine API with prepared-data input, updating
both in-repository callers and documenting the complete migration example. This makes file
IO and model execution independently reusable. Cost: external C++ callers must explicitly
load/preprocess or call PredictVision with the engine adapter.

## Verification

The new host-test project first failed configuration on the absent implementation. After
implementation, both Release and ASan/UBSan builds pass two CTest entries:

1. Stage contract, patch/channel address probes, noncontiguous ROI/no mutation, invalid image
   types, runner count, invalid/nonfinite inputs and outputs, ownership and unreadable files.
2. Four real source images compared byte-for-byte against the original C++ OpenCV preprocessing
   implementation, independently compiled with a renamed symbol. Original source bytes were
   checked against the fixed S Git object; input digests are recorded.

Release assertions remain enabled. Existing seven Python launcher tests pass. Migration
README contract remains 50 samples / 0 violations / 51 policy skips / 0 exemptions.
Bilingual runtime guides include stage contracts, a complete SDK library example, and the
actual SDK-free CMake/CTest commands. The SDK example was not compiled against vendor headers;
only the pure stage/IO test target was compiled here. No model, quantization or board ran.

## Remaining native work

This is not an SDK transport acceptance. The retained source Vision transport still needs
strict input/output count/shape/type/stride checks, replacement of its unknown-type float
fallback, and cleanup on constructor/inference failure. Text engine/KV/application boundaries,
explicit model preparation and final whole-branch review are also pending. The full H0–H9
objective is unchanged; no board environment is required to continue the host refactoring.

Evidence: [source, images and actual test output](evidence/2026-09-28-b11-gemma-vision/checks.json),
[initial missing-implementation failure](evidence/2026-09-28-b11-gemma-vision/red-configure.txt),
[sanitizer build and tests](evidence/2026-09-28-b11-gemma-vision/sanitized-build.txt).
