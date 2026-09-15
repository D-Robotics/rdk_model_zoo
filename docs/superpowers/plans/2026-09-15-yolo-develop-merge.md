# YOLO develop merge implementation plan

**Goal:** On local develop, merge audited X5/S ultralytics_yolo into one sample with the existing conversion/evaluator/model/runtime/test_data layout.

**Spec:** User approved the preceding audit and requested a develop branch and first YOLO merges. Detailed baseline evidence: audit/YOLOV8-X5-S-MERGE-AUDIT.md (base 6fcef2b87c12435e11fbd7327ea70d4efd917b1c).

**Constraints:** Local only. No push, deployment, model-server changes or invented board validation. Preserve other samples and historic Benchmark records. Do not merge YOLO26/YOLOE/standalone yolov5 in this batch. Preserve X5/S compiler differences and X5-only C++ support. Model binary shapes remain unverified. No new global inference SDK. Explicit platform wins over host detection; unsupported assets must fail clearly.

- [x] Create develop from clean main; record baseline.
- [x] Add root samples/vision/ultralytics_yolo with one Python main and task implementations. Preserve existing CLI semantics where possible, add explicit platform selection and packed/split NV12 input handling. Retain YOLOv10 S-specific dispatch and all previously supported shared-directory families.
- [x] Consolidate only required utility dependencies; avoid overwriting platform-wide utils used by other samples. Use __file__-relative paths throughout new sample.
- [x] One download entry from authoritative asset metadata, correct X5 640 and S 224 classification asset names. No unrequested dependency installation. Dry-run asset resolution and --help work on a host without board runtime.
- [x] Share export patch with effective opset argument; preserve separate hb_mapper/hb_compile conversion workflows, calibration formats, and X5 C++ implementation.
- [x] Unified evaluators call current classes; preserve ImageNet/COCO metric definitions and handle empty predictions. Document exact support and platform defaults bilingually.
- [x] Replace old duplicated implementations with forwarding entrypoints or migration pointers as needed; preserve evidence docs, download compatibility and catalog provenance. Update registry/readmes/publisher as necessary so canonical code links resolve correctly without data loss.
- [x] Run CPU tests for protocol mapping, parameter precedence, deterministic pre/postprocessing, defaults, output normalization, empty outputs, downloader resolution and evaluator interfaces. Verify shell syntax and Python compilation. No tests may download models or install board dependencies.
- [x] Run catalog publisher checks, compare family/config/benchmark/asset identity to baseline; inspect diff independently and fix findings. Verify unchanged unrelated sample code. Commit locally on develop and report limits.

Validation commands: Python unittest discovery for the added sample tests; Python compileall for changed source; Bash -n for changed shell; npm --prefix tools/catalog-publisher run check. Record exact results in audit before final commit.
