# Model Zoo Web Release Inputs

This directory contains the reviewed, lightweight inputs required to reproduce
the published static site in a clean checkout.

- `inputs.json` maps catalog records to their release assets.
- `assets/` contains optimized website covers. Covers that show inference
  overlays must be produced from real model output.
- `reports/` contains compact structured OE data used by the website report
  view. This committed JSON remains the reproducible build input; the public
  Archive copy has identical bytes.

`assets/model-inference-banners.json` selects reviewed family/task covers.
The 2026-11-03 set contains 19 references: 14 classifiers plus depth, DINOv2
patch features, MODNet alpha matting, YOLOv5 detection and YOLOE11 segmentation.
ImageGen produces only the clean input photographs. The pinned conversion
ONNX runs on the host; native sample preparation and decoding produce the
Top-5 values, boxes, masks and visualizations. The existing photographic,
light-gray/white/blue website style is retained.

`assets/inference-banners-20261103/` keeps the exact inference inputs, compact
outputs, predictions, receipts, generation prompts and inference plan. Large
YOLO forward tensors remain in the recorded Workbench run. The rendering
commands are `scripts/infer_classification_banners.py` and
`scripts/infer_vision_banners.py`; use the plan, the recorded native source
checkout and existing ONNX files. Neither command downloads or converts models.
The execution source commit is separate from each model's frozen conversion
source commit. The build verifies asset bytes and checkpoint/ONNX identities
against the catalog before selecting a cover.

The retained cover metadata identifies these as host float inference references.
One recorded source variant represents its family/task across variants and
platforms. This does not establish BIN/HBM board behavior, dataset accuracy
or performance; those checks remain `not-run`. DINOv2 colors are a PCA
projection of actual patch features, not semantic segmentation. Existing
compatible covers are retained; unavailable models keep their placeholders.

Public deployment model URLs use the verified Archive platform/sample paths.
Public release/OE companions use
`https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/assets/rdk_<platform>/<source>/<model-stem>/`.
Each directory contains `release.json`, the byte-preserved `release.source.json`,
`oe_report.html`, `oe_report_data.json`, and `SHA256SUMS`. The relocated release
preserves model, license, version and measurement facts and records its source
manifest identity. The checksum file describes an assembled download of the
model and four companions; the deployment model is stored separately.

Existing published OSS objects remain available during the website migration.
Uploading companion assets does not deploy the site or publish a Git branch.

Paths in `inputs.json` are relative to this directory. The release build uses
this manifest by default; `MODEL_ZOO_INPUTS` may override it for local review.

New conversion-only publications use the same model directories and versioned
companions `assets/rdk_<platform>/<source>/<model-stem>/v1/` (then `v2`, `v3`
for measurements of the same model SHA). Existing versionless paths remain
valid. A changed binary requires a distinct artifact variant and filename.

`status: converted` requires fixed checkpoint/ONNX/config/source/toolchain
identities, conversion and host graph checks, a model license, real OE reports,
and verified public model bytes. Board checks declare `not-run`/`passed`/`failed`
individually; absent metrics display as pending measurement. Conversion estimates
never populate board performance. `released` retains the complete acceptance
contract. Pending document records may retain their historical facts while adding
formal `variants` under a separate `model_license` with `scope: model`.

Stage frozen models and their five metadata companions on dev-server with
`scripts/stage_converted.py`. Keep checkpoints, ONNX, BIN/HBM and model bundles
on that server; copy only the five metadata companions to the Mac. Initiate
uploads from the Mac with a direct SSH connection:

```bash
python3 scripts/upload_archive_model.py --manifest metadata/release.json \
  --model /server/frozen/model.hbm --source-host dev-server \
  --ssh-host robot@139.129.17.229 --receipt receipts/model.json
python3 scripts/upload_archive_assets.py metadata \
  --ssh-host robot@139.129.17.229 --receipt receipts/assets.json
```

The model uploader relays SSH stdout/stdin without a local model file and runs
the complete anonymous HTTPS verification on dev-server. A jump is optional
and must be selected explicitly with `--source-jump-host` / `--jump-host`.
After verified upload,
`register_converted.py --web model_zoo_web --manifest <release.json> --oe-data
<oe_report_data.json>` registers only that variant/platform and keeps legacy
references. Native download and runtime support must land on develop first.
For later measurements, upload a new metadata revision and register with
`--supplement`; the tool requires the same artifact URL/SHA and an increasing
revision. Promotion to released requires all board checks and full metrics.

The canonical Workbench procedure is
`/home/zhengyi/work/rdk_model_zoo_workbench/GUIDELINE.md` v2.16, sections 11.1–11.9.
Its embedded helper sources match these scripts. Generic website builds use the
same committed inputs and require no Workbench connection.

For independent converted weights, a platform row may carry its own model
`license`; it takes precedence over the family default without rewriting older
artifacts. Pin a 40-character checkpoint revision, or use
`sha256:<source_checkpoint_sha256>` for an official direct-download checkpoint.
Preserve the publisher's actual license and source notices. Multiple compiler
subgraphs use `scripts/bundle_oe_html.py` to retain every original HTML byte in
one companion report; static estimates remain separate from board measurements.

SAM and Faster R-CNN publish complete `.tar.gz` deployment pairs. The bundle
contains every HBM, a deployment descriptor, each original component OE report
and JSON, the original connected host check, licenses and native sample commands.
Each component retains its own ONNX/config/model identity and public HBM URL.
`component_package.py` verifies all member bytes and the connected report before
staging; the five companion files retain their existing Archive directory layout.
Upload and verify both component HBMs and the complete bundle before registration.
Combined compiler reports leave whole-pipeline latency and FPS empty until measured.
