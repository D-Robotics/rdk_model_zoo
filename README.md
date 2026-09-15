<div align="center">
  <img src="platforms/x5/docs/assets/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo</h1>
  <p align="center">
    <b>Out-of-the-Box AI Model Deployment Pipelines and Full-Link Conversion Tutorials Based on D-Robotics BPU</b>
  </p>
</div>

<div align="center">

**English** | [简体中文](./README_cn.md)

<p align="center">
  <a href="https://github.com/D-Robotics/rdk_model_zoo/stargazers"><img src="https://img.shields.io/github/stars/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Stars"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/network/members"><img src="https://img.shields.io/github/forks/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Forks"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## Introduction

> **Mission**: Dedicated to providing D-Robotics developers with extreme performance, out-of-the-box, and full-scenario AI deployment validation experiences.

This repository is the official collection of BPU model examples and tools (Model Zoo) provided by D-Robotics. It is oriented towards AI model deployment and application development on BPU (Brain Processing Unit), helping developers to **quickly get started with BPU** and **fast-track model inference workflows**.

The main branch carries all supported platforms. Migration to shared samples starts with [Ultralytics YOLO](samples/vision/ultralytics_yolo/README.md): one Python entry point with explicit platform profiles, separate conversion toolchains, and X5-only C++. Its old platform commands forward to the shared implementation and require the complete repository checkout. Other samples remain under `platforms/`; board verification remains a separate release check.

### Platform Registry

| Target Hardware | Path | Historical branch | Documentation |
| :--- | :--- | :--- | :--- |
| RDK X5 | [`platforms/x5`](./platforms/x5) | `rdk_x5` | [README](./platforms/x5/README.md) · [中文](./platforms/x5/README_cn.md) |
| RDK S100 / S100P / S600 | [`platforms/s`](./platforms/s) | `rdk_s` | [README](./platforms/s/README.md) · [中文](./platforms/s/README_cn.md) |
| RDK X3 | [`platforms/x3`](./platforms/x3) | `rdk_x3` | [README](./platforms/x3/README.md) · [中文](./platforms/x3/README_cn.md) |

The registry is also published as machine-readable data in [`platforms/registry.json`](./platforms/registry.json), and described in prose in [`platforms/README.md`](./platforms/README.md).

### Historical Branches

All new models, fixes, manifests, and release preparation are maintained on **main**. Existing platform branches and tags are retained as historical and compatibility references; new development does not require synchronizing those branches. Platform release versions may evolve independently while their source directories stay on main.

| Target Hardware | Branch | Description |
| :--- | :--- | :--- |
| RDK X5 | [`rdk_x5`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | Historical RDK X5 branch. Recommended system version: RDK OS >= 3.5.0, based on Ubuntu 22.04 aarch64 and TROS-Humble. |
| RDK X5 legacy demos | [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) | Historical archive branch for the previous RDK X5 demos. Use it only when you need to reference legacy demo content. |
| RDK X3 | [`rdk_x3`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x3) | Historical RDK X3 branch. |
| RDK S series | [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) | Historical RDK S series branch. Historical archived demos for RDK S series boards are kept in [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s). |

## Repository Layout

```bash
rdk_model_zoo/
|-- platforms/
|   |-- x5/                  # Complete RDK X5 distribution (verbatim layout)
|   |   |-- samples/         # vision/ and robotics/ samples
|   |   |-- utils/           # shared Python utilities and batch tools
|   |   |-- datasets/        # dataset preparation helpers
|   |   |-- docs/
|   |   |   |-- release/     # models.yaml + benchmarks.yaml (X5 manifest pair)
|   |   |   `-- releases/    # published release notes
|   |   |-- tros/            # TROS integration references
|   |   `-- README.md        # X5 entry point
|   |-- s/                   # Complete RDK S100/S100P/S600 distribution
|   |   |-- samples/         # vision/, speech/, vla/
|   |   `-- docs/release/    # models.yaml + benchmarks.yaml (S manifest pair)
|   `-- x3/                  # Complete RDK X3 distribution
|       |-- demos/           # legacy demo layout
|       `-- release/         # models.yaml + benchmarks.yaml (X3 manifest pair)
|-- tools/
|   `-- catalog-publisher/   # catalog data generator, schema validation, errata
|-- archive/                 # local source snapshots (not committed)
|-- docs/superpowers/        # design records and implementation plans
`-- .github/workflows/       # catalog data build and validation
```

### Release Manifests

Each platform publishes the same manifest pair — `models.yaml` (artifact inventory) and `benchmarks.yaml` (performance and accuracy observations) — together with the JSON Schemas that validate them. The manifests are the authoritative record of what a release ships and measures; the catalog is a derived, read-only view.

| Platform | Manifest directory | Release tag | Counts |
| :--- | :--- | :--- | :--- |
| X5 | [`platforms/x5/docs/release`](./platforms/x5/docs/release) | `x5-v1.1.2` | 37 samples, 239 benchmark records |
| S | [`platforms/s/docs/release`](./platforms/s/docs/release) | `s-v1.1.2` | 35 samples, 563 benchmark records |
| X3 | [`platforms/x3/release`](./platforms/x3/release) | `x3-v1.1.2` | 15 samples, 20 benchmark records |

The counts are per manifest. Two RDK X3 `paddleocr` records also appear in the X5 manifest, so the sums are two higher than the catalog's deduplicated total of 820.

Historical release tags (`x5-v1.1.2`, `s-v1.1.2`, `x3-v1.1.2` and their predecessors) keep the **original repository-root layout** they were published with: a tag checkout holds `samples/`, `docs/release/`, or `release/` at the top level, not `platforms/`. Links inside old release notes and the manifests' own `source.path` fields therefore resolve against those tags exactly as before.

## Catalog Data

`tools/catalog-publisher` reads the three platform manifests, validates them against their own schemas, applies the documented normalisation and errata layers, and emits a versioned data package:

```bash
cd tools/catalog-publisher
npm ci
npm run check           # validate sources, run tests, typecheck, rebuild, verify
npm run catalog:build   # write dist/catalog.json and dist/catalog.meta.json
```

`dist/catalog.meta.json` pins the exact bytes of `catalog.json` with a SHA256 digest; consumers verify that digest before using the data. The package is uploaded as a build artifact by `.github/workflows/model-catalog-data.yml`. No branch of this repository deploys a website.

## License

Each platform distribution carries its own license file — see [`platforms/x5/LICENSE`](./platforms/x5/LICENSE) and [`platforms/s/LICENSE`](./platforms/s/LICENSE). Upstream X3 published no license file and none has been added here.
