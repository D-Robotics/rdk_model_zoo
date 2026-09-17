# Representative Sample integration revision

Date: 2026-09-17. Scope: YOLO detection, ResNet18 classification and one PaddleOCR sample.

The 2026-09-16 evidence proves fixed-input runtime compatibility, not completion
of the full source/documentation integration. The user rejected the latter
claim. This record tracks the corrective work against Spec R02/R20 and
D02/D03/D05/D07; testing alone does not close structural findings.

## Findings and changes

| Finding | Correction | Verification |
| --- | --- | --- |
| Root README accumulated conflicting pilot/completion notices and lacked a task-first path | Replaced with one bilingual task table, setup entry, source layout, platform/history separation | Eight help/list commands pass from outside the checkout; 44 Markdown files have no broken local links |
| Three NV12 conversion copies | `samples/_shared/image.py:bgr_to_nv12_planes`; original sample functions are validation/compatibility adapters | 24 exact comparisons with archived pre-refactor functions, known-color and noncontiguous-input tests; affected-board checks pass on both X5 boards, S100 and applicable S100P YOLO models; S600 reconnect pending |
| ResNet conversion/evaluation and S C++ not integrated into canonical sample | Canonical exporter, model downloader, labels, legacy adapters, C++ sources/build; bilingual workflows | 35 host tests; seeded PyTorch/ONNX Runtime comparison; S100 old/new C++ Top-5 exact |
| OCR conversion/evaluation and S C++ only referenced through old paths | Local target YAMLs, calibration helper, saved-record evaluator, canonical C++ and Python task; old APIs delegate | Both X5 boards and S100: old/new default and aspect pipelines exact; S100 old/new C++ rendered pixels exact |
| YOLO conversion still duplicates complete compiler orchestration | One workflow owns ONNX inspection, calibration, config, compilation and output handling; four mappers supply profiles | 59 host tests include compiler failure, workspace preservation and simulated successful artifact handling; no actual compiler available |

## Shared operation mapping

| Previous function | Current maintenance location | Preserved local responsibility |
| --- | --- | --- |
| `resnet/runtime/python/tensor_io.py:bgr_to_nv12_planes` | `_shared/image.py:bgr_to_nv12_planes` | uint8/HWC/nonempty and even-size errors |
| `paddle_ocr/runtime/python/tensor_io.py:_bgr_to_nv12_planes` | Same shared function | OCR image validation, X5 LINEAR / S AREA resizing, packed/split binding |
| `ultralytics_yolo/runtime/python/rdk_yolo_utils/preprocess.py:bgr_to_nv12_planes` | Same shared function | Existing task geometry and runtime tensor names |

Input is even-sized uint8 HWC BGR; output is contiguous uint8 Y `(1,H,W,1)`
and UV `(1,H/2,W/2,2)`. The algorithm retains OpenCV I420 conversion and U,V
interleave ordering; no model normalization, image resize or SDK call moved
into the common module. Runtime imports remain lazy. Previous source hashes
and exact comparison counts: [extraction evidence](evidence/2026-09-17-nv12-extraction.json).

## Verification status

- Host: 17 shared + 35 ResNet + 59 YOLO + 38 OCR = 149 tests passed. The final OCR suite includes non-empty calibration-directory preservation and zero-threshold disjoint-box rejection. [Combined snapshot](evidence/2026-09-17-host-p2-ocr-tests.log), [final OCR follow-up](evidence/2026-09-17-ocr-final-host-tests.log).
- Runtime source archive `77c9532feccd6f157b47b7df342a6a3bfff648fbcae65300feaf89729ed982cd`: both X5 boards, S100, and S100P passed applicable YOLOv8n/YOLO26n comparisons; ResNet passed both X5 boards and S100. S100P has no approved ResNet18 asset. Canonical algorithms are unchanged since that archive; two legacy CLI argument adapters received follow-up host checks.
- OCR runtime archive `7e75debcc4cc212b4a36560b1488cd45b111024b7a86cb9202d5238b50232e3f`: both X5 boards and S100 passed default/aspect preprocessing, raw-output and final-result comparison, compatibility wrappers, native command, output ownership and wrong-target rejection. Current runtime files match the archive.
- S100 C++ archive `ba1d74010890f326173ace7b5aa4700b1a629380cb629aad6c224f503424e99e`: ResNet and OCR canonical/old CMake entries built and executed successfully. ResNet Top-5 text and OCR rendered image pixels equal original baselines. All four launchers also pass from `/tmp` with explicit `--flag=value` paths. [Build/run](evidence/s100-2026-09-17-cpp-integration-attempt3.log), [launchers](evidence/s100-2026-09-17-cpp-launchers.log).
- Failed attempts are retained: OCR CMake initially resolved `samples/platforms` instead of repository `platforms` (fixed); attempt 2 upload interrupted; attempt 3 passed. OCR initially assumed unfiltered polygons aligned with filtered boxes (fixed and covered by a regression case).
- S600: September 17 retest is **not-run**. Three connection attempts failed; the latest SSH banner connection closed. User confirmed the same address and planned connection recovery. September 16 results remain historical and are not substituted for this revision.
- ResNet source exporter: seeded random-weight PyTorch and ONNX Runtime outputs agree within `1e-4`; observed max absolute difference `1.7881393432617188e-6`. This checks the graph, not pretrained accuracy or BPU compilation. [Evidence](evidence/2026-09-17-resnet-export-smoke.json).
- Conversion toolchains: not executed in this session. `hb_mapper`/`hb_compile` and original ResNet18 calibration/config are unavailable locally. Documentation and simulated orchestration do not establish newly compiled artifact compatibility.
- At validation time, no commit or push had been made. The subsequent user-authorized `develop` delivery is recorded in [development status](2026-09-17-development-status.md). No release, default-branch switch or system SDK change.

## Review status

Root reviewed source ownership, preserved model-specific preprocessing, compatibility return shapes, CLI argument forwarding, build paths, workspace lifecycle, and documentation claims. Independent review found no concrete blocker in shared NV12, sample navigation, ResNet binding or YOLO conversion command/workspace handling; shared, YOLO and ResNet suites were rerun by the reviewer. A local-link audit covered 44 Markdown files with zero missing targets. Two legacy command-forwarding checks confirm explicit equals-form paths are retained and S target selection uses detected identity for custom paths. This is a three-sample integration revision, not full-catalog completion or a new accuracy/performance certification.

The bounded source/documentation revision is complete with the unexecuted S600 retest and real OE compilation explicitly retained above. [Command forwarding](evidence/2026-09-17-legacy-command-forwarding.json), [entrypoint smoke checks](evidence/2026-09-17-entrypoint-smoke.json), [local links](evidence/2026-09-17-doc-links.json).
