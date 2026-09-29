# Platforms, branches and release records

[简体中文](README_cn.md)

Current X5/S integration proceeds on `develop` and has not completed customer-release acceptance. `rdk_x5` and `rdk_s` are platform delivery lines; historical tags retain their published self-contained layouts. Integration neither moves tags nor turns development status into a release. X3 remains a historical distribution outside new adaptation scope.

The table follows this checkout's [registry.json](registry.json). Registered release tags are recorded metadata, not a query for the latest remote release. Manifest paths refer to the current integration tree:

| Platform | Delivery line | Registered tag | Current manifest | Platform material |
|---|---|---|---|---|
| RDK X5 | `rdk_x5` | `x5-v1.1.3` | [docs/release/x5](../docs/release/x5) | [x5](../platforms/x5/README.md) |
| RDK S100 / S100P / S600 | `rdk_s` | `s-v1.1.2` | [docs/release/s](../docs/release/s) | [s](../platforms/s/README.md) |
| RDK X3 | `rdk_x3` | `x3-v1.1.2` | [platforms/x3/release](../platforms/x3/release) | [x3](../platforms/x3/README.md) |

## What the three path types mean

- Root `samples/`: maintained unified samples in the [complete index](../samples/README.md), extending beyond the initial three pilots.
- Root `docs/release/x5` and `docs/release/s`: current maintenance locations for artifact identity, URLs, release facts and historical metrics. `tools/catalog-publisher` resolves these manifest paths from its own `sources.json`; `platforms/registry.json` is cross-checked against that configuration by tests and is not read by the build. Platform-local manifests remain migration source records; do not create divergence by editing both copies.
- `platforms/{x5,s}/`: source-platform material, pending capabilities and compatibility entries. Some entries forward to root samples; copying a subdirectory alone can omit dependencies, so keep a complete checkout.

Historical tags have `samples/`, `docs/release/` or `release/` at their original repository root. Resolve manifest `source.path` against that tagged layout, without imposing a `platforms/` prefix.

## Runtime and artifact differences

X5 and S both expose a module named `hbm_runtime`, but their SDK backends and models differ. Common X5 image artifacts use `.bin` and packed NV12; S uses `.hbm`, with S100/S100P/S600 mapped to nash-e/m/p and common image inputs split into Y/UV. Point-cloud, feature and audio tasks have their own protocols; not every sample uses NV12. X3 historical runtimes are `hobot_dnn` / `bpu_infer_lib_x3`.

[Hardware identity](../docs/release/platforms.json), asset publication, language implementation and board acceptance are distinct. Missing assets must not silently select another board; not-run is not passed. Ultralytics C++ now contains X5/S input adaptations: see its [task/validation limits](../samples/vision/ultralytics_yolo/runtime/cpp/README.md), replacing the obsolete X5-only description.

## Retained development references

- X5: [guidelines](x5/docs/Model_Zoo_Repository_Guidelines.md), [source references](x5/docs/source_reference/README.md), [datasets](x5/datasets), robotics source samples.
- S: [guidelines](s/docs/Model_Zoo_Repository_Guidelines.md), [Python API](s/docs/Python_API_User_Guide.md), [UCP](s/docs/UCP_User_Guide.md), [datasets](s/datasets), speech/VLA source samples.
- ACT/Pi0 remain gitlink entries in root [.gitmodules](../.gitmodules); an upstream reference is not migration or acceptance evidence.
- X3 retains its historical `demos/`, `resource/`, `release/` layout rather than adopting X5/S conventions.
- Source-branch online resources, inherited from the archived X5 (`ac11571`) and S (`380e1a2`) root guides: the published [online model catalog](https://d-robotics.github.io/rdk_model_zoo/), [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues), and the [D-Robotics developer community](https://developer.d-robotics.cc/) with its [user manual](https://developer.d-robotics.cc/information). They reference the delivery branches' published material; browsing the published catalog does not certify this integration branch, and no live link status is claimed here. Platform toolchain manuals stay in the archived guides rather than being re-asserted as current.
- Legacy entries are archives, not adaptation targets: X5 legacy demos live on the [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) archive branch and archived S demos in the separate [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) repository. The `rdk_x5`/`rdk_s` delivery lines remain the registered lines in the table above.

Source-platform READMEs preserve hardware background, model lists, FAQs, conversion/runtime instructions and community links. New shared samples follow the [inference contract](../docs/sample-standards/inference-contract.md) and [README contract](../docs/sample-standards/readme-contract.md); current scope is in the [plan](../docs/superpowers/plans/2026-09-26-host-completion.md) and [ledger](../docs/releases/unified-migration/x5-s-migration-map.md).

## Measurements and licenses

Historical benchmarks retain their original model, target, version and conditions. Registry/manifest counts and deduplicated catalog counts can differ; recalculate with the publisher instead of treating old snapshot totals as current inventory. See [catalog-publisher](../tools/catalog-publisher) for schemas, build rules and deduplication tests.

X5/S retain their license files. Upstream X3 supplied no license file and none is invented during migration. Check unified code and third-party models/data separately; moving directories does not change provenance.
