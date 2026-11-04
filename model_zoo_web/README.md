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

The S600 `official` catalog contains four Gemma4 E2B/E4B profiles and 32 profiles from
12 additional series: Qwen2.5, Qwen3, Qwen3.5, Qwen2.5-VL, Qwen3-VL, Qwen3-ASR,
InternVL2, InternVL2.5, InternVL3.5, MiniCPM5, Phi-4 Mini and Whisper. VLA is excluded.
The full vendor ZIPs and member digests are verified, and downloads point directly to the SDK's canonical OSS URLs.
Published vendor TTFT, decode speed and Linux+ION memory are displayed with their source; the concurrency selector retains one stream.
Eagle3/DFlash speed ranges remain ranges, and ASR entries include RTF. Each series has its own
illustrative banner shared across its sizes and acceleration profiles, with saved generation provenance.
Local board records stay `not-run` and null, independently of the official references.
These entries have native preparation/run commands and source archives, without synthetic OE reports or upstream revisions.
The native [OELLM Sample](../samples/llm/oellm/README.md) uses the supplied SDK `basic_sample`
and `asr_demo`; vendor code, headers, libraries and binaries are obtained separately.
They are counted separately from locally verified `released` models and remain outside the released-only Ask AI scope.

## Ownership

- `samples/` owns conversion, evaluation, and runtime code.
- `model_zoo_web/data/` owns website metadata for released sample models.
- Archive serves deployment models, release manifests, checksums, and complete OE reports.

Catalog records use this layout:

```text
data/<domain>/<source>/<series>/<task>.yaml
```

Each task record groups model sizes and target platforms. Descriptions are
authored in Chinese and English at the task level and describe model
architecture only. Deployment formats, model statistics, accuracy, and
performance remain in their dedicated release fields.

Official Gemma and registered OELLM profiles use the verified SDK OSS ZIP URLs recorded in their Sample manifests.
Other public deployment models use the verified Archive platform and sample directory:

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_<platform>/<source>/<artifact>
```

Release manifests and OE reports use the model-specific Archive `assets/`
directory described in [CONTRIBUTING.md](CONTRIBUTING.md). Preserve the model's
SHA-256, size, source versions, and original publication receipts when changing
its download location. Public catalog and website validation accept the scoped official SDK OSS contract;
other OSS model downloads remain rejected, while historical packages stay readable for receipt validation.

Build dates, repository commits, and toolchain versions belong in release
metadata rather than the object path.

## Build and validation

Install the local build dependencies and validate both the empty website shell
and the samples-only catalog:

```bash
python3 -m pip install -r model_zoo_web/requirements.txt
npm ci --prefix model_zoo_web
export MODEL_ZOO_SAMPLE_REF=7a97c1780b0cfa131ff0007d662e255e5e41c12e
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

Pending records resolve `sample_path` and every evidence path at their own
40-character `sample_revision`. They do not require `MODEL_ZOO_SAMPLE_REF` or
the current `develop` tree. Use a full-history checkout (`git fetch --unshallow`
for a shallow clone); if a pinned commit is still missing, fetch that exact
SHA with `git fetch --no-tags origin <sample_revision>`. The build reads local
Git objects and never fetches automatically. CI and Pages use `fetch-depth: 0`.
For existing released records without `sample_revision`, local sample paths
are checked first. When their source is absent from the Web checkout, the
`MODEL_ZOO_SAMPLE_REF` export above is required for `check` and `build:release`.
CI and Pages use the same immutable fallback commit for those released records.

Every cover must be a site-relative local asset. Pending records without a
reviewed `hero_visual` use the shared SVG placeholder, even when an
`example_image` is recorded as source evidence. Sample code licenses are
displayed separately from model licenses; `scope: sample-code` does not
declare the model's license.

A host-inference cover keeps its own `hero_visual.commit`, matching the saved
prediction record. Its input and source documentation are validated at that
exact commit. Adding a new converted profile may advance `sample_revision`
without relabeling an existing cover or its predictions.

The model-detail view links an OE conversion report only when the imported
release record supplies a reviewed public Archive URL. Missing reports keep the
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
committed here. Their stable Archive URLs and checksums are supplied through the
reviewed catalog-generation step.

## Ask AI

Ask AI is temporarily disabled. The catalog, model details, empty-search view
and report pages have no AI entry points, and release builds exclude the Ask AI
client script and stylesheet. The shared model-request preview is also hidden.
The source module and local service tooling are retained for future restoration.

## GitHub Pages release

The CI and Pages workflows live with the website on the `model_zoo_web`
branch. Ordinary branch pushes run validation only. A semantic `web-v*` tag
builds the reviewed release inputs and deploys `model_zoo_web/dist` to Pages.

Push the branch before creating the release tag:

```bash
git push upstream model_zoo_web
git tag -a web-v1.0.0 -m "RDK Model Zoo Web v1.0.0"
git push upstream web-v1.0.0
```

The deployment workflow rejects malformed tags and commits that are not part
of `origin/model_zoo_web`. The repository Pages source must be set to GitHub
Actions, and the `github-pages` environment must allow the release tag.
