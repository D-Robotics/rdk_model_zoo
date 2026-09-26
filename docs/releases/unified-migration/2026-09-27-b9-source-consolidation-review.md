# B9 standalone YOLO source consolidation — in progress

Status: source inventory and exact asset preparation/routing implemented on host.
Numerical/C++ consolidation and full README migration remain open.
Independent Review=not-run; Board=not-run; Closed=no. This is author investigation.

## Source scope and preservation

S source pin: `380e1a2bf42041af54be6f34935e50197cfadff9`.
The [reproducible file audit](evidence/2026-09-27-b9-source-consolidation/audit_sources.py)
reads the pin directly from Git. Its [78-file inventory](evidence/2026-09-27-b9-source-consolidation/source-files.json)
confirms all archived bytes match: YOLO11 detection 18, pose 18, segmentation 18,
iMoonLab YOLOv13 24. This includes all three standalone C++ implementations and
YOLOv13's YAML, three tool logs, six test/illustration assets and bilingual guides.
No archive is replaced, deleted or described as migrated merely by preserving it.

## Implemented asset routing

The unified CLI previously rejected the ten standalone identities because it
accepted only the `ultralytics_yolo` / `ultralytics_yolo26` sample IDs. The new
finite source selection maps three YOLO11 tasks (S100/S600, n) and four S100
YOLOv13 sizes (n/s/l/x) onto the existing task dispatch. Source asset references,
URLs and hashes remain in the active S manifest; no substitute family artifact
is selected. Source files use `model/standalone/<sample-id>/<filename>` to avoid
same-basename collisions. Main planning, public model_path and downloader agree.

S100 `--family yolov13` now resolves the iMoonLab source (whose filename is
`yolo13...`, not `yolov13...`). No S100P/S600 YOLOv13 family is advertised.
Exact source IDs reject target/task/family/size conflicts. Source pose/segmentation
IDs retain NMS 0.70 and detection IDs 0.45; explicit thresholds win. Ordinary
YOLO11 family selection retains its existing S default. Existing image/scheduler
CLI defaults remain unchanged and source comparison instructions are explicit.
The downloader's `--asset-id` selects source records; `--all` remains the family
inventory, now including S100 YOLOv13, and does not duplicate standalone YOLO11.
Bilingual root/model/runtime guides explain these distinctions and limits.

Ten targeted tests cover every source record, conflicts, all-path agreement,
listing, exact shared downloader calls, dry-run no-write behavior, same-basename
isolation and source defaults. Initial RED used a misspelled listing function;
that test typo was corrected before the retained RED run. A first implementation
run caught duplicate package/top-level exception classes; imports were corrected
before the final GREEN run. These are development findings, not board evidence.

## Source differences that must be handled before B9 closure

1. Detection and segmentation source wrappers dequantize with `output_quants`.
   The current canonical DFL runner only supports declared scalar quantization
   and applies it in output binding during forward; the segmentation task does
   not preserve the standalone dequantization step. Quantized source outputs
   need explicit metadata validation and post-process transformation, not a
   permissive float cast or hidden transform in the runner.
2. Source pose returns keypoint logits for its visualization helper; the unified
   result returns sigmoid probabilities. Compare after this declared conversion,
   and preserve source visibility semantics. Do not label raw tuples equal.
3. Detection's current mutable `last_transform` state conflicts with the explicit
   per-call context standard. Geometry must travel with prepared input. Existing
   tests/readmes alone do not close H2's task-purity audit.
4. Native detect/pose/segment exist on both sides. Their parameters, quantization,
   reusable API and lifetime behavior still need detailed mapping/host tests.
   Source presence is not proof that the unified C++ interface preserves them.
5. YOLOv13 conversion documentation and table contents need full integration.
   Its source YAML is for n, while retained compilation logs are for l and record
   hbdk4.1.17/hmct2.1.9/hb_compile3.3.11. Windows-absolute documentation links and
   calibration path prerequisites must be repaired rather than copied as working
   commands. Source performance/accuracy remains historical, not a new result.
6. S600 source filenames contain nashe. Preserve the manifest row; do not infer
   observed physical HBM architecture from the filename or advertise S100P fallback.

These findings keep B9 in progress; the implemented routes do not waive them.
YOLOE and all other unfinished H0–H9 work remain in scope. No model body download,
board/HP/SSH, real SDK, OE conversion, dataset accuracy or performance run occurred.

## Host verification for this routing increment

[Full commands and return codes](evidence/2026-09-27-b9-source-consolidation/host-results.json)
record the shared144/ResNet52/OCR44/checker27 regressions and initial Ultralytics89
run. The final Ultralytics90 run adds the legacy-inventory regression; combined
final checked scope is **357 tests**. The updated listing text was included in
that final run. Migration contracts:44 samples /0 violations /45 policy skips /
0 exemptions. These checks do not prove the unfinished numerical/native work.

[README verification](evidence/2026-09-27-b9-source-consolidation/readmes.json)
records local links across all six changed guides, exact bilingual equality of
the three new dry-run commands, and successful execution without SDK or network.
