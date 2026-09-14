# 平台登记 (Platform Registry)

main 分支并列收纳全部已支持硬件平台。`platforms/` 下的每个目录都是一个**完整、自包含的发行版**：其中的示例、运行时代码、转换配置、文档与发布清单，与该平台独立发布分支上的内容完全一致，内部相对路径保持不变。

[English](./README.md) | **简体中文**

| 平台 | 目录 | 历史分支 | 清单目录 | 发布 Tag | 运行时 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RDK X5 | [`x5/`](./x5) | `rdk_x5` | [`x5/docs/release`](./x5/docs/release) | `x5-v1.1.2` | `hbm_runtime` |
| RDK S100 / S100P / S600 | [`s/`](./s) | `rdk_s` | [`s/docs/release`](./s/docs/release) | `s-v1.1.2` | `hbm_runtime` |
| RDK X3 | [`x3/`](./x3) | `rdk_x3` | [`x3/release`](./x3/release) | `x3-v1.1.2` | `hobot_dnn`、`bpu_infer_lib_x3` |

同一份登记表连同硬件标识与许可证指针，以机器可读形式发布于 [`registry.json`](./registry.json)。

## 为什么不做平台合并

这三个平台是近亲，但不是可互换的构建。X5 与 S 都提供名为 `hbm_runtime` 的模块，但底层依赖不同；X5 模型使用 `.bin`，S 模型使用 `.hbm`；X3 早于两者，运行 `hobot_dnn` 与 `bpu_infer_lib_x3`。目录约定也不同——X5、S 使用 `samples/vision/<model>/`，X3 使用旧版 `demos/<task>/<Model>/` 布局与 PascalCase 命名。

如果摊平成单一的 `samples/` 目录树，就必须重命名目录、改写运行时导入，并把两套互不兼容的 API 强行合并。这会破坏已发布文档中的每一条相对路径，并让发布清单静默失效——清单正是以仓库相对路径精确引用来源的。保持平台完整可以完全避免这些问题，代价只是每条链接多一层前缀。

## 布局不变式

平台目录内部的任何内容都不得依赖平台前缀。像 `platforms/x5/samples/vision/ultralytics_yolo/runtime/python/run.sh` 这样的路径，必须在同一子树被检出到仓库根目录时同样可用——历史发布 Tag 正是这种布局。贡献者不得引入绝对路径、跨平台相对导入，或假设存在同级平台。

## 各平台内容

### `x5/` —— RDK X5

在 main 维护的 X5 实现。`samples/vision/` 存放规范化示例（小写命名、`hbm_runtime`、packed NV12 输入）；`samples/robotics/` 存放具身智能策略。`utils/py_utils/` 提供共用的前处理、后处理与可视化工具。`docs/Model_Zoo_Repository_Guidelines.md` 是示例结构与接口的权威规范。

### `s/` —— RDK S100 / S100P / S600

规范化方式与 X5 相同，但在同一目录树内面向三个板卡目标，并额外提供两类示例：`samples/speech/` 与 `samples/vla/`。VLA 策略（ACT、Pi0）在仓库 `.gitmodules` 中声明为 git 子模块，其路径带有 `platforms/s/` 前缀；子模块 gitlink 本身未做改动。`docs/Python_API_User_Guide.md` 与 `docs/UCP_User_Guide.md` 分别记录 S 系列运行时与统一计算平台。

### `x3/` —— RDK X3

历史 Demo 线，按发布原样保留：`demos/` 存放可运行 Demo，`resource/` 存放共用资源，`release/` 存放清单对。它是 X3 交付内容的历史基线记录，不是仍在规范化的目录树，也不适用 X5/S 的目录约定。

## 清单 (Manifests)

每个平台都发布 `models.yaml` 与 `benchmarks.yaml`，以及校验它们的 JSON Schema。清单是权威记录；[`tools/catalog-publisher`](../tools/catalog-publisher) 产出的目录数据包只是派生视图。

| 平台 | 示例数 | 基准记录 | 性能指标 | 精度指标 |
| :--- | ---: | ---: | ---: | ---: |
| X5 | 37 | 239 | 636 | 419 |
| S | 35 | 563 | 1382 | 2797 |
| X3 | 15 | 20 | 104 | 25 |

以上为**各清单自身**的统计，**不可直接相加**。历史上 X5 清单中同时携带了两条 RDK X3 的 `paddleocr` 记录；两个平台都从各自目录树发布这两条记录，目录数据只统计一次并归属 X3。因此目录数据共有 820 条基准记录，而非 822 条。

## 历史 Tag

发布 Tag 保留发布时的布局：仓库根目录，而非 `platforms/`。检出 `x5-v1.1.2`、`s-v1.1.2` 或 `x3-v1.1.2` 后，顶层是 `samples/`、`docs/release/` 或 `release/`，这些清单中的每一个 `source.path` 都按该布局解析。已发布 Tag 不可变，永远不会被移动到 `platforms/` 前缀下。

## 许可证

各平台各自携带许可证文件。X5 与 S 提供 `LICENSE`；上游 X3 未发布许可证，迁移过程中也未新增。

## 维护方式

新增模型、修复、Manifest 与发布准备统一在 main 维护。旧平台分支与 Tag 保留为历史兼容入口。新增板卡需要登记平台目录并扩展目录数据的硬件映射，无需更换 main 或迁移仪表盘。
