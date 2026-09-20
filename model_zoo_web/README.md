# RDK Model Zoo Web

This directory is the source of truth for the samples-only Model Zoo catalog
and its static website. It does not scan historical `platforms/` manifests.

```text
data/       reviewed release metadata grouped by model family and task
src/        catalog UI and empty report-library shell
public/     brand logo and interface icons
scripts/    catalog, website, and validation scripts
build/      generated catalog intermediate; ignored by Git
dist/       generated website artifact; ignored by Git
```

New model submissions must follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Ownership

- `samples/` owns conversion, evaluation, and runtime code.
- `model_zoo_web/data/` owns website metadata for released sample models.
- OSS owns binaries, release manifests, checksums, and complete OE reports.

Catalog records use this layout:

```text
data/<domain>/<source>/<series>/<task>.yaml
```

Each task record groups model sizes and target platforms. Descriptions are
authored in Chinese and English at the task level and describe model
architecture only. Deployment formats, model statistics, accuracy, and
performance remain in their dedicated release fields.

Public artifacts use stable OSS paths:

```text
models/<source>/<series>/<task>/<size>/<platform>/<artifact>
```

Build dates, repository commits, and toolchain versions belong in release
metadata rather than the object path.

## Build and validation

Install the local build dependencies and validate both the empty website shell
and the samples-only catalog:

```bash
python3 -m pip install -r model_zoo_web/requirements.txt
npm ci --prefix model_zoo_web
npm --prefix model_zoo_web run check
```

The catalog builder reads only `model_zoo_web/data/` and writes deterministic
intermediate files to `model_zoo_web/build/`:

```bash
npm --prefix model_zoo_web run build:catalog
npm --prefix model_zoo_web run check:catalog
```

The default website build remains an empty catalog shell. The release build
uses the reviewed `release/inputs.json` manifest, regenerates
`build/catalog.json`, and writes the complete site to `dist/`:

```bash
npm --prefix model_zoo_web run build:release

npm --prefix model_zoo_web run check:release
```

`MODEL_ZOO_CATALOG` and `MODEL_ZOO_INPUTS` may optionally override the default
catalog and release-input paths for local review.

The model-detail view links an OE conversion report only when the imported
release record supplies a reviewed public OSS URL. Missing reports keep the
non-interactive placeholder; do not add an empty `href`, a fake `#` target, or
a link to local report payloads.

The input manifest is keyed at two levels. `models` supplies one shared cover
per catalog model/task record, so every size and platform of YOLO26 Detect uses
the same reviewed image. `releases` supplies optional structured OE data for a
specific size/platform artifact:

```json
{
  "schema_version": 1,
  "models": {
    "ultralytics_yolo/yolo26/detect": {
      "cover": "results/yolo26-detect-cover.png"
    }
  },
  "releases": {
    "ultralytics_yolo/yolo26/detect/n/x5": {
      "oe_data": "runs/yolo26n-x5/oe_report_data.json"
    }
  }
}
```

Relative paths are resolved from the input-manifest directory. The preview
build transforms every released size/platform entry into the UI's model
contract. Structured OE JSON is copied into the site and matched to its exact
artifact; complete OE HTML payloads remain outside the normal generated site.
The website does not expose a separate original-HTML download action and does
not read `platforms/` manifests.

Runtime `threads` are displayed as Runtime submission concurrency. A separate
`performance.end_to_end` record supplies the single-pipeline C++ latency and
stage breakdown. Maximum-performance end-to-end records use every online CPU
thread and must state the CPU governor and CPU/BPU frequencies explicitly.

Compiled models, conversion outputs, and complete OE reports must not be
committed here. Their stable OSS URLs and checksums are supplied through the
reviewed catalog-generation step. This project also does not define or own a
GitHub Pages deployment.
