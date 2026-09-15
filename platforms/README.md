# Platform Registry

The main branch maintains all supported platforms. Platform-local samples remain here; audited shared samples move to the root `samples/` tree with compatibility entry points. Ultralytics YOLO is the first such migration. Historical release tags retain their original self-contained layout.

**English** | [简体中文](./README_cn.md)

| Platform | Directory | Historical branch | Manifest directory | Release tag | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RDK X5 | [`x5/`](./x5) | `rdk_x5` | [`x5/docs/release`](./x5/docs/release) | `x5-v1.1.2` | `hbm_runtime` |
| RDK S100 / S100P / S600 | [`s/`](./s) | `rdk_s` | [`s/docs/release`](./s/docs/release) | `s-v1.1.2` | `hbm_runtime` |
| RDK X3 | [`x3/`](./x3) | `rdk_x3` | [`x3/release`](./x3/release) | `x3-v1.1.2` | `hobot_dnn`, `bpu_infer_lib_x3` |

The same registry, with hardware identifiers and license pointers, is published as [`registry.json`](./registry.json) for tooling.

## Differences retained across platforms

The platforms are close relatives, not interchangeable builds. X5 and S both expose a module named `hbm_runtime`, with different platform backends. X5 artifacts use `.bin` and S artifacts use `.hbm`; X3 predates both and runs `hobot_dnn` with `bpu_infer_lib_x3`. Directory conventions differ too — X5 and S use `samples/vision/<model>/`, while X3 uses a legacy `demos/<task>/<Model>/` layout with PascalCase names.

## Shared sample migration

The first audited merge is [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README.md). It keeps the existing five-directory sample layout with shared Python task implementations and explicit platform input profiles. X5/S conversion workflows remain separate; C++ remains X5-only. Old platform commands forward to the root sample, so this sample now requires the complete repository checkout. Its original README/Manifest paths remain Benchmark provenance.

Other samples still use platform-local layouts. Migrate them individually after auditing runtime, compiler, artifact and evaluation differences. Historical tags keep the layouts and APIs they originally published; this local merge does not rewrite tags or require synchronized platform release tags.

## Platform contents

### `x5/` — RDK X5

The X5 implementation maintained on main. `samples/vision/` holds the standardized samples (lowercase names, `hbm_runtime`, packed NV12 input); `samples/robotics/` holds embodied-AI policies. `utils/py_utils/` provides the shared preprocessing, postprocessing, and visualization helpers. `docs/Model_Zoo_Repository_Guidelines.md` is the authoritative specification for sample structure and interfaces.

### `s/` — RDK S100 / S100P / S600

Same standardization as X5, with three board targets inside one tree and two additional sample categories: `samples/speech/` and `samples/vla/`. The VLA policies (ACT, Pi0) are declared as git submodules in the repository `.gitmodules`, whose paths carry the `platforms/s/` prefix; the submodule gitlinks themselves are unchanged. `docs/Python_API_User_Guide.md` and `docs/UCP_User_Guide.md` document the S-series runtime and the unified computing platform.

### `x3/` — RDK X3

The historical demo line, preserved as published: `demos/` for runnable demos, `resource/` for shared assets, and `release/` for the manifest pair. It is a baseline record of what X3 shipped, not an actively standardized tree, and it is not held to the X5/S directory conventions.

## Manifests

Every platform publishes `models.yaml` and `benchmarks.yaml` along with the JSON Schemas that validate them. The manifests are authoritative; the catalog data package produced by [`tools/catalog-publisher`](../tools/catalog-publisher) is a derived view.

| Platform | Samples | Benchmark records | Performance metrics | Accuracy metrics |
| :--- | ---: | ---: | ---: | ---: |
| X5 | 37 | 239 | 636 | 419 |
| S | 35 | 563 | 1382 | 2797 |
| X3 | 15 | 20 | 104 | 25 |

These are per-manifest counts and are **not additive**. Two RDK X3 `paddleocr` records were historically carried inside the X5 manifest as well; each platform publishes them from its own tree, and the catalog counts them once, under X3. The catalog therefore holds 820 benchmark records, not 822.

## Historical tags

Release tags keep the layout they were published with: the repository root, not `platforms/`. Checking out `x5-v1.1.2`, `s-v1.1.2`, or `x3-v1.1.2` yields `samples/`, `docs/release/`, or `release/` at the top level, and every `source.path` in those manifests resolves against that layout. Published tags are immutable and are never moved to the `platforms/` prefix.

## License

Each platform carries its own license file. X5 and S ship a `LICENSE`; upstream X3 published none, and none was added during the migration.

## Maintenance

All new models, bug fixes, manifests and release preparation land on main. Historical platform branches and tags remain compatibility references. Adding hardware means registering a platform directory and extending the catalog hardware mappings, not changing the main branch or moving the dashboard.
