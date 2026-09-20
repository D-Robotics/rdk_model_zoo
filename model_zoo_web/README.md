# RDK Model Zoo Web

This directory contains the standalone source for the static RDK Model Zoo
website. It is the primary catalog UI source in this repository.

```text
src/       catalog UI and empty report-library shell
public/    brand logo and interface icons
scripts/   deterministic build and validation scripts
dist/      generated deployment artifact; ignored by Git
```

The default build remains an empty catalog shell. Released sample models are
imported explicitly with `build:preview` from the reviewed samples-only
catalog; the website never scans historical `platforms/` metadata.

The model-detail view links an OE conversion report only when the imported
release record supplies a reviewed public OSS URL. Missing reports keep the
non-interactive placeholder; do not add an empty `href`, a fake `#` target, or
a link to local report payloads.

```bash
npm ci --prefix model_zoo_web
npm --prefix model_zoo_web run check
```

To build a local preview from the samples-only catalog, provide the generated
catalog and a reviewed multi-model input manifest:

```bash
MODEL_ZOO_CATALOG=/path/to/model_zoo_web/dist/catalog.json \
MODEL_ZOO_INPUTS=/path/to/model-zoo-web-inputs.json \
npm --prefix model_zoo_web run build:preview
```

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
