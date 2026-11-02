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
