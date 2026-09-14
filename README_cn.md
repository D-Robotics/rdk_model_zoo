<div align="center">
  <img src="platforms/x5/docs/assets/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo</h1>
  <p align="center">
    <b>基于 D-Robotics BPU 的开箱即用 AI 模型部署 Pipeline 与全链路转换教程</b>
  </p>
</div>

<div align="center">

[English](./README.md) | **简体中文**

<p align="center">
  <a href="https://github.com/D-Robotics/rdk_model_zoo/stargazers"><img src="https://img.shields.io/github/stars/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Stars"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/network/members"><img src="https://img.shields.io/github/forks/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Forks"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## 仓库简介 (Introduction)

> **使命**：致力于为地瓜机器人开发者提供极致性能、开箱即用、覆盖全场景的 AI 部署验证体验。

本仓库是 D-Robotics（地瓜机器人）官方提供的 BPU 模型示例与工具集合（Model Zoo），面向运行在 BPU（Brain Processing Unit）上的 AI 模型部署与应用开发，用于帮助开发者**快速上手 BPU**、**快速跑通模型推理流程**。

main 分支同时收纳**全部已支持硬件平台**的维护中发行版。每个平台在 `platforms/` 下保留各自完整、自包含的目录树——示例、运行时代码、转换配置、文档与发布清单——不做裁剪。平台目录树是逐字保留的发行版：内部相对路径不变，因此原有示例相对路径保持有效；板端运行仍需单独验收。各平台不共享运行时 API，本仓库也不做这种假装统一。

### 平台登记 (Platform Registry)

| 目标硬件 | 路径 | 历史分支 | 文档 |
| :--- | :--- | :--- | :--- |
| RDK X5 | [`platforms/x5`](./platforms/x5) | `rdk_x5` | [README](./platforms/x5/README.md) · [中文](./platforms/x5/README_cn.md) |
| RDK S100 / S100P / S600 | [`platforms/s`](./platforms/s) | `rdk_s` | [README](./platforms/s/README.md) · [中文](./platforms/s/README_cn.md) |
| RDK X3 | [`platforms/x3`](./platforms/x3) | `rdk_x3` | [README](./platforms/x3/README.md) · [中文](./platforms/x3/README_cn.md) |

平台登记同时以机器可读形式发布于 [`platforms/registry.json`](./platforms/registry.json)，说明文字见 [`platforms/README.md`](./platforms/README.md)。

### 历史分支 (Historical Branches)

新增模型、修复、Manifest 和发布准备统一在 **main** 维护。现有平台分支和 Tag 保留为历史与兼容入口，新开发不再要求同步这些分支。各平台版本可以独立演进，源码目录始终保留在 main。

| 目标硬件 | 分支 | 说明 |
| :--- | :--- | :--- |
| RDK X5 | [`rdk_x5`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | RDK X5 历史分支。推荐系统版本：RDK OS >= 3.5.0，基于 Ubuntu 22.04 aarch64 与 TROS-Humble。 |
| RDK X5 历史 Demo | [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) | 旧版 RDK X5 Demo 的历史归档分支，仅在需要参考旧版 Demo 内容时使用。 |
| RDK X3 | [`rdk_x3`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x3) | RDK X3 设备分支。 |
| RDK S 系列 | [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) | RDK S 系列板卡分支。RDK S 系列历史归档 Demo 保留在 [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s)。 |

## 仓库结构 (Repository Layout)

```bash
rdk_model_zoo/
|-- platforms/
|   |-- x5/                  # RDK X5 完整发行版（逐字保留原布局）
|   |   |-- samples/         # vision/ 与 robotics/ 示例
|   |   |-- utils/           # 共用 Python 工具与批处理工具
|   |   |-- datasets/        # 数据集准备脚本
|   |   |-- docs/
|   |   |   |-- release/     # models.yaml + benchmarks.yaml（X5 清单对）
|   |   |   `-- releases/    # 已发布版本说明
|   |   |-- tros/            # TROS 集成参考
|   |   `-- README.md        # X5 入口文档
|   |-- s/                   # RDK S100/S100P/S600 完整发行版
|   |   |-- samples/         # vision/、speech/、vla/
|   |   `-- docs/release/    # models.yaml + benchmarks.yaml（S 清单对）
|   `-- x3/                  # RDK X3 完整发行版
|       |-- demos/           # 旧版 demo 布局
|       `-- release/         # models.yaml + benchmarks.yaml（X3 清单对）
|-- tools/
|   `-- catalog-publisher/   # 目录数据生成、Schema 校验与勘误
|-- archive/                 # 本地源码快照（不入库）
|-- docs/superpowers/        # 设计记录与实施计划
`-- .github/workflows/       # 目录数据构建与校验
```

### 发布清单 (Release Manifests)

各平台发布同一组清单——`models.yaml`（产物清单）与 `benchmarks.yaml`（性能与精度观测）——以及校验它们的 JSON Schema。清单是发布内容与实测结果的权威记录，目录数据只是派生的只读视图。

| 平台 | 清单目录 | 发布 Tag | 统计 |
| :--- | :--- | :--- | :--- |
| X5 | [`platforms/x5/docs/release`](./platforms/x5/docs/release) | `x5-v1.1.2` | 37 个示例，239 条基准记录 |
| S | [`platforms/s/docs/release`](./platforms/s/docs/release) | `s-v1.1.2` | 35 个示例，563 条基准记录 |
| X3 | [`platforms/x3/release`](./platforms/x3/release) | `x3-v1.1.2` | 15 个示例，20 条基准记录 |

历史发布 Tag（`x5-v1.1.2`、`s-v1.1.2`、`x3-v1.1.2` 及其更早版本）保留发布时的**仓库根目录布局**：检出 Tag 后顶层是 `samples/`、`docs/release/` 或 `release/`，而不是 `platforms/`。因此旧版版本说明中的链接与清单自身的 `source.path` 字段，仍按原有方式解析到这些 Tag。

## 目录数据 (Catalog Data)

`tools/catalog-publisher` 读取三个平台的清单，用各平台自带的 Schema 校验，套用已归档的归一化与勘误层，产出带版本的目录数据包：

```bash
cd tools/catalog-publisher
npm ci
npm run check           # 校验源、跑测试、类型检查、重建并复核
npm run catalog:build   # 写出 dist/catalog.json 与 dist/catalog.meta.json
```

`dist/catalog.meta.json` 用 SHA256 摘要锁定 `catalog.json` 的精确字节；使用方在读取前需校验该摘要。该数据包由 `.github/workflows/model-catalog-data.yml` 构建并上传为构建产物。本仓库任何分支都不部署网站。

## 许可证 (License)

各平台发行版各自携带许可证文件——见 [`platforms/x5/LICENSE`](./platforms/x5/LICENSE) 与 [`platforms/s/LICENSE`](./platforms/s/LICENSE)。上游 X3 未发布许可证文件，本次迁移也未新增。
