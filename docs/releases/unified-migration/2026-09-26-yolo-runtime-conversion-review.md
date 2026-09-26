# Ultralytics Python and conversion README review — 2026-09-26

Status: this documentation slice is implemented and host-checked. The sample/root/C++ README review and remaining migration are still in progress.

## Python runtime

Both languages now document every CLI option and alias with the parser's literal default, followed by the platform/task resolution rules. This matters for classification resize and detection NMS: a `null` parser value must not be described as one universal effective default. Added default entry plus segmentation/pose/classification/OBB usage, all five return schemas, actual CLI output behavior and the stage I/O contract. Classification only prints Top-K; it does not save `--img-save-path`. The existing complete image-loading integration example remains available, with its board/model prerequisite stated.

The code reading cross-check used main.py, yolo_dispatch.py, yolo_runtime.py, platform profiles and individual task implementations. The doc gate now verifies the full Python parameter tables rather than skipping them due to missing anchors.

## Conversion

Retained the existing source-branch container, calibration, compiler, artifact and troubleshooting detail. Added source-weight provenance requirements, all four existing non-detection YOLO26 export recipes, explicit calibration sampling defaults and honest missing environment/weight/dataset/version evidence. All Python commands use repository-root paths; the former export-directory `cd` no longer leaves later mapper commands in the wrong directory.

During parser checking, the non-detection YOLO26 exporters were confirmed not to support `--require-local`; examples and prose reflect that limitation. Generic and YOLO26 detection exporters do support it. The source/compiler/runtime distinction and DFL versus direct-LTRB protocols remain explicit. No source checkpoint hash, image version or conversion success was invented.

## Verification

[Structured evidence](evidence/2026-09-26-yolo-runtime-conversion-doc-check.json) records 19 Python command examples reaching their real argument parser successfully. The checker intercepted execution immediately after final parsing, so no checkpoint imports, training dependencies or compilers ran. Runtime examples additionally completed actual dry-run selection (the default-only entry used explicit x5 for a host). English/Chinese command lines match after stripping comments; all local links resolve. The library snippet was syntax checked only, not executed on a board.

Contract check: 36 samples / 0 violations / 37 policy skips / 30 exact exemptions. Removed 14 Python-runtime and 16 conversion section exemptions after the substantive rewrites. Remaining debt covers the sample-level and C++ READMEs; the exemption file is not yet removed. Parser/link checks prove document consistency, not numerical inference or actual model conversion. No runtime code changed in this slice; previous algorithm test results are not relabeled as new board evidence.
