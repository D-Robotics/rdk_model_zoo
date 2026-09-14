# Platform Registry

The main branch distributes every supported hardware platform side by side. Each directory under `platforms/` is a **complete, self-contained distribution**: its samples, runtime code, conversion configurations, documentation, and release manifests are exactly the ones published on that platform's own release line, with internal relative paths unchanged.

**English** | [简体中文](./README_cn.md)

| Platform | Directory | Historical branch | Manifest directory | Release tag | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RDK X5 | [`x5/`](./x5) | `rdk_x5` | [`x5/docs/release`](./x5/docs/release) | `x5-v1.1.2` | `hbm_runtime` |
| RDK S100 / S100P / S600 | [`s/`](./s) | `rdk_s` | [`s/docs/release`](./s/docs/release) | `s-v1.1.2` | `hbm_runtime` |
| RDK X3 | [`x3/`](./x3) | `rdk_x3` | [`x3/release`](./x3/release) | `x3-v1.1.2` | `hobot_dnn`, `bpu_infer_lib_x3` |

The same registry, with hardware identifiers and license pointers, is published as [`registry.json`](./registry.json) for tooling.

## Why the platforms are not merged

The platforms are close relatives, not interchangeable builds. X5 and S both expose a module named `hbm_runtime`, with different platform backends. X5 artifacts use `.bin` and S artifacts use `.hbm`; X3 predates both and runs `hobot_dnn` with `bpu_infer_lib_x3`. Directory conventions differ too — X5 and S use `samples/vision/<model>/`, while X3 uses a legacy `demos/<task>/<Model>/` layout with PascalCase names.

A single flattened `samples/` tree would have to rename directories, rewrite runtime imports, and collapse two incompatible APIs into one. That would break every relative path in the published documentation and silently invalidate the release manifests, which cite sources by exact repository-relative path. Keeping the platforms whole avoids all of it, and costs only a one-level prefix on every link.

## Layout invariant

Nothing inside a platform directory may depend on the platform prefix. A path such as `platforms/x5/samples/vision/ultralytics_yolo/runtime/python/run.sh` must work when the same subtree is checked out at the repository root, which is exactly what the historical release tags hold. Contributors must not introduce absolute paths, cross-platform relative imports, or references that assume a sibling platform exists.

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
