# Model Zoo Web Release Inputs

This directory contains the reviewed, lightweight inputs required to reproduce
the published static site in a clean checkout.

- `inputs.json` maps catalog records to their release assets.
- `assets/` contains optimized website covers. Covers that show inference
  overlays must be produced from real model output.
- `reports/` contains compact structured OE data used by the website report
  view. This committed JSON remains the reproducible build input; the public
  Archive copy has identical bytes.

Deployment model URLs remain on their registered OSS or Archive paths.
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

From a frozen manifest and real files, use `scripts/stage_converted.py`, then
`upload_archive_model.py` and `upload_archive_assets.py` from the local Mac
through the existing Tailscale jump. After verified upload,
`register_converted.py --web model_zoo_web --manifest <release.json> --oe-data
<oe_report_data.json>` registers only that variant/platform and keeps legacy
references. Native download and runtime support must land on develop first.
For later measurements, upload a new metadata revision and register with
`--supplement`; the tool requires the same artifact URL/SHA and an increasing
revision. Promotion to released requires all board checks and full metrics.

The canonical Workbench procedure is
`/home/zhengyi/work/rdk_model_zoo_workbench/GUIDELINE.md` v2.13, sections 11.5–11.7.
Its embedded helper sources match these scripts. Generic website builds use the
same committed inputs and require no Workbench connection.

For independent converted weights, a platform row may carry its own model
`license`; it takes precedence over the family default without rewriting older
artifacts. Pin a 40-character checkpoint revision, or use
`sha256:<source_checkpoint_sha256>` for an official direct-download checkpoint.
Preserve the publisher's actual license and source notices. Multiple compiler
subgraphs use `scripts/bundle_oe_html.py` to retain every original HTML byte in
one companion report; static estimates remain separate from board measurements.
