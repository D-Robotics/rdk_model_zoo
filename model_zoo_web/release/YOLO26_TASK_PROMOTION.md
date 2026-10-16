# YOLO26 task release promotion

Publish the model packages through the existing Detect OSS release route, then
activate the Web catalog. Source remains in the repository's `develop` branch;
the four new task pages link to `develop/samples/vision/ultralytics_yolo`.

The batch contains 60 releases: Cls/Seg/Pose/OBB × n/s/m/l/x × S600/S100P/S100.
The separate X5 candidates are outside this batch.

## Select the repaired builds

Pass the immutable, hash-bound selection manifest when promoting this batch:

```sh
python3 scripts/promote_yolo26_task_batch.py \
  --workbench-root /home/zhengyi/work/rdk_model_zoo_workbench \
  --rebuild-selection /home/zhengyi/work/rdk_model_zoo_workbench/outputs/analysis/yolo26-web-preview-100-20261001/rebuild-selection.json \
  --release-date 2026-10-01
```

The command is a read-only dry run by default. Add `--apply` after it passes.
It selects exactly these four repaired releases and retains the original builds
for the other 56:

| Release | Selected BuildID |
| --- | --- |
| Pose-s S100P | `20260930T1402Z-becb0688-yolo26s-pose-s100p-fixed-max-asym-b8-r01` |
| OBB-x S600/S100P/S100 | `20260930T1837Z-becb0688-yolo26x-obb-int16-max1-b8-r03` |

Repaired public objects use
`models/ultralytics_yolo/yolo26/{task}/{size}/{platform}/rebuilds/{BuildID}/`.
The catalog validates this directory against the entry's recorded BuildID.
The first Pose repair uses its hash-bound historical staging directory, while
its finalized and public package uses the versioned directory above.

## Required evidence

Every entry must pass the workbench stage audit before promotion. The promoter
checks the selected campaign, model and board receipt, runtime source manifest,
full-split accuracy comparator, fixed-input runtime performance, C++ end-to-end
results and environment, and all corresponding SHA-256 bindings.

Each finalized package contains the compiled model, `oe_report.html`,
`oe_report_data.json`, schema-2 `release.json`, and `SHA256SUMS`. All five need
verified public-read OSS upload receipts matching their exact keys, bytes, hashes
and URLs: 300 receipts for the batch. The finalization receipt must also bind
the selected staged and finalized packages. A staged-local manifest cannot
activate an official catalog entry.

Agent review receipts remain labeled as agent reviews. They require a separate,
hash-bound explicit user publication authorization covering the exact release
scope. They do not represent human expert accuracy or license signoff.
This batch follows the user's AGPL-3.0 release instruction, with the source
entry in develop and license information in the model metadata. Evaluation
images and annotations are excluded from the public model packages.

Accuracy is copied from the accepted float/board comparator without replacing
measured values or claiming that every model has negligible precision loss.
Cls uses ImageNetV2 MatchedFrequency (10,000 images); Seg and Pose use COCO
val2017 (5,000 images); OBB uses local DOTA labeled validation (458 images),
single scale, rather than an official DOTA test submission. The float-to-board
comparison includes deployed preprocessing differences.

## Activate and deploy

Promotion validates all 60 entries and the Web schema before changing files.
It writes four task YAML records, 60 structured OE data files and the activation
in `release/inputs.json`, preserving the 40 existing Detect mappings.
Existing task files cannot be overwritten by another promotion.

```sh
npm ci
npm run check
npm run build:release
npm run check:release
```

The release build contains 100 platform/size entries in six model families:
YOLO11 Detect, YOLO26 Detect, Cls, Seg, Pose and OBB. Commit the Web changes to
`model_zoo_web`, then push a `web-vMAJOR.MINOR.PATCH` tag to run the existing
GitHub Pages deployment. Verify the live catalog and the four versioned repair
downloads after the workflow succeeds.
