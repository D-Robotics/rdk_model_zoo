# Ultralytics classification stage and README review

Base: `8e41b5e`. Implementation/host verification by Codex; independent whole-branch
review remains not-run. Board, real SDK, OE conversion and dataset evaluation are
not-run. H2 and the full migration remain open.

## Change and source contract

YOLOv8/YOLO11/YOLO26 classification now uses the existing ModelRunner and explicit
ClassificationContract instead of loading an SDK in the task and taking the
first output without validating it. The task file contains configuration,
construction, scheduling delegation and pre_process/forward/post_process/predict.
Image/NV12 preparation is shared; classification numerical operations live in
classification_decode.py. No SDK load, download, label loading, drawing, filesystem
write or logging occurs in the stages.

The default is one 1000-class floating logit vector. Metadata must expose exactly
one output, static shape and dtype; optional singleton axes are supported, with
batch one required for multidimensional outputs. `(1000,1)` is rejected rather
than flattened into a class vector. Integer/SCALE outputs, batches, spatial maps,
missing descriptors and extra outputs fail explicitly. The common runner's
hardware identity gate now also covers direct classification library use.

Forward calls the runner once and preserves the physical floating array and
buffer identity in RawOutputs. Postprocess validates that carrier's binding,
performs one SciPy Softmax and the existing descending NumPy argsort, returning
independent Python scalar pairs. Exact tie order is deliberately unchanged.
Positive Top-K above class count still returns all classes; zero, negative,
boolean and noninteger values now fail explicitly. Nonempty uint8 BGR HxWx3
input is required. These are deliberate validation improvements, not claims of
unchanged malformed-input behavior.

The CLI resize policies remain: X5 YOLOv8/11 letterbox, S stretch, YOLO26 stretch
on every target. The library config defaults to stretch. The bilingual runtime
README explains the difference, metadata-based input sizes and the legacy
S100/S100P 640-ID/224-URL distinction. A complete classification example runs all
three stages and compares them to predict; it does not write an image.

## Evidence and limits

Pinned original X5 (`ac115717197920355fc390bb04299b20e6436864`) and S
(`380e1a2bf42041af54be6f34935e50197cfadff9`) classification modules and their
preprocessing helpers are preserved byte-for-byte with license headers and SHA-256
provenance in tests/fixtures/classification_sources.json. Tests execute actual
source preprocessing for two nonsquare images × two resize modes × two targets,
and source postprocessing for random, tied and extreme logits on both targets.
SDK constructors are bypassed only in these synthetic source comparisons.
Metadata binding and raw transport run through the real common runner with a fake
SDK on all four target profiles. This does not prove published asset metadata or
real runtime compatibility; those require board validation.

The first full host run found four old YOLO26 classification fixtures patching the
previous loader, so they reached the real execution identity gate. Their complete
failed output is preserved in the evidence directory. The fixtures were updated
to inject the fake SDK through the common runner while retaining full metadata
validation. A separate red test demonstrated the ambiguous `(1000,1)` shape before
its validation fix. No production identity check was weakened.

Final commands, timestamps, exit codes and logs are in
[evidence](evidence/2026-09-28-yolo-classification/host-results.json); the
[summary](evidence/2026-09-28-yolo-classification/result.json) records verified totals.
No catalog assets or conversion recipes changed in this increment. OBB, YOLO26
non-detection task stages, v10, native code and the remaining H0–H9 scope are not
closed by these classification tests.
