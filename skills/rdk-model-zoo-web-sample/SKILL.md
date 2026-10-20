---
name: rdk-model-zoo-web-sample
description: "Use when adding or completing a Model Zoo Web sample record: which fields each task requires, what may stay empty, and how a registered-but-pending model differs from a complete release. 触发词：登记模型、网页示例、数据待补充、补齐基准。Do not invent latency, accuracy, file size, or download URLs."
version: "1.0.0"
license: Apache-2.0
metadata:
  author: "RDK Model Zoo maintainers"
  content-license: "CC-BY-4.0"
  pack: "rdk-model-zoo"
  data-classification: "public"
---

# Model Zoo Web 标准示例

## Purpose

规定一条模型怎样才算 Model Zoo Web 上的标准示例。人或者 Agent 在新增示例、把 `main` / `develop` 上已有示例登记进网页、或把待补记录补成正式发布时，都按这里的字段边界执行。

网页目录在 `model_zoo_web/data/<domain>/<source>/<series>/<task>.yaml`。任务代码表是 [`model_zoo_web/task-registry.json`](../../model_zoo_web/task-registry.json)。正式发布的精度字段表是 [`model_zoo_web/accuracy-metrics.json`](../../model_zoo_web/accuracy-metrics.json)。

## When to use

适用：决定一条记录现在应该写成 `registration: pending` 还是正式发布；为一个示例补 YAML；核对某类任务还缺什么。

不适用：改示例推理代码、做量化编译、重新测量板端性能。那些工作仍走仓库里对应示例的 README，以及 `main` 上的开发、校验和发布技能。

## Two states

一条记录只有两种状态。细节和禁止项见 [registration-states.md](references/registration-states.md)，按任务的字段见 [field-matrix.md](references/field-matrix.md)。

| 状态 | YAML | 网页上是什么 | 何时使用 |
| --- | --- | --- | --- |
| 待补 `registration: pending` | 身份、任务、示例路径、已核对的板卡 | 卡片和详情显示“数据待补充”，没有下载、没有基准数字、没有封面 | 示例已经在 `main` / `develop`，但还没有一套满足正式发布门槛的证据 |
| 完整（省略 `registration`） | 现有发布记录的全部字段 | 可筛选、可下载、可看精度和性能、可打开转换报告 | 转换、精度、Runtime、端到端、制品、许可证和封面都已核对 |

已有的完整记录不要降级，也不要为了“统一格式”改写它们的数字。

## What a pending record must contain

必填：

- `schema_version: 1`、`registration: pending`
- `id`，等于 `<source>/<family>/<task>`，并且与文件路径 `data/<domain>/<source>/<family>/<task>.yaml` 一致
- `name`、`domain`、`source`、`family`、`task`
- `description.zh` 和 `description.en`：只写架构或示例实际在做什么。不写帧率、精度、体积、宣传语
- `sample_path`：`samples/` 下的真实目录。本地没有时，用 `MODEL_ZOO_SAMPLE_REF`（CI 为 `refs/remotes/origin/develop`）核对
- `sample_revision`：读到这些事实时的完整 40 位提交。链接必须指向这次提交里的示例目录
- `supported_platforms`：至少一块板卡。每项是 `platform`（`x5`、`s100`、`s100p`、`s600`）和 `evidence`（`docs/release/` 或 `samples/` 里真实存在的文件）

`task` 必须是任务代码表里的键。同一示例如果发布了彼此独立的模型（例如检测器和识别器），每种任务一个 YAML。同一份导出同时带框和掩码时，用一个 YAML，并用 `facet_tasks` 让它出现在多个筛选项里，不要复制出第二条没有独立导出的记录。

可选，而且只有原文写明了才能填：

- `license.name` 和官方 `https` 许可证页
- `provider`：README 点名的上游项目
- `sample_kind: gitlink`：示例在仓库里是子模块提交而不是目录树时
- `manifest_tasks`：发布清单里的任务 id，便于和 `docs/release/{x5,s}/models.yaml` 对照
- `facet_tasks`：这个示例实际覆盖的筛选项 id

待补记录可以带上仓库里已经写明的事实，但每一项都要有 `source`，并且原文能在 `sample_revision` 的那个文件里对上。允许的摘录：

- `upstream`：README 或集成清单里出现的 https 仓库、论文或模型卡链接
- `license`：原文点名的许可证。官方许可证页可以配已点名的 SPDX 名称；不要配原文没写的许可证
- `published_variants`：支持表或清单里的变体名
- `conversion_configs`：`conversion/` 下真实存在的 yaml 或 `mapper_*.py`，以及文件里写明的 march、输入类型和分辨率
- `inputs`：`docs/release/*/benchmarks.yaml` 某一条里写明的 shape、layout、format
- `usage`：README 快速体验代码块里的命令原文
- `sourced_benchmarks`：基准清单里的数字，保留 metric、单位、统计量和条件（包括 “compiler estimate” 这种字样）。这不是本站正式发布基准
- `documented_downloads`：`docs/release/*/models.yaml` 里已经写出的 https URL。`sha256: null` 不要填成哈希。S 系列只在文件名或 URL 点名具体板卡时写入
- `example_image`：`test_data` 里适合当示例的图片。架构图和性能截图不要当封面

仍然禁止用完整记录的字段名（`variants`、`accuracy`、`performance`、`assets`、`benchmarks`、`cover`）塞进待补记录。没有精度字段表、制品校验、Runtime 与 C++ 端到端的任务，即使摘录了数字，也保持 `registration: pending`。

## How to name a supported board

一块板卡可以写入 `supported_platforms`，仅当下面至少一条成立：

1. 示例出现在 `docs/release/x5/models.yaml`，则可以写 `x5`。
2. S 系列清单或基准文件里，该示例的文件名、`environment.hardware` 或转换配置路径明确写出 `s100`、`s100p`、`s600`（或 Nash-E / Nash-M / Nash-P）。
3. 示例 README 的支持表逐块板卡写明支持。

`docs/release/s/models.yaml` 顶部的 “RDK S100/S100P/S600” 是整份 S 清单的兼容范围，不能当成每一个模型都支持三块板。只在 S 清单里出现、又没有任何一条点名具体板卡时，不要猜测板卡；把缺的板卡留空，并在说明里写出缺的是哪一条证据。

X3 是历史平台，不写入新的网页记录。

## How to mark a record complete

删除 `registration: pending`，按 [field-matrix.md](references/field-matrix.md) 补齐该任务的完整字段，并让 `python3 model_zoo_web/scripts/build_catalog.py` 通过。完整记录还必须满足 `model_zoo_web/CONTRIBUTING.md` 的发布门槛：可复现的转换和运行入口、稳定 OSS 制品、全量精度、Runtime 与 C++ 端到端、许可证和 16:9 真实推理封面。

`accuracy_profile` 为 `null` 的任务还没有网页精度字段表。在 `accuracy-metrics.json` 增加由评测回执支持的字段之前，不要把这种任务标成完整，也不要借用分类或检测的 mAP 字段。

## Procedure

1. 读目标示例的 README、`docs/release/{x5,s}/models.yaml`，以及该示例 `conversion/` 里点名板卡的配置。示例源码以 `main` / `develop` 为准；网页分支可以不携带源码。
2. 若该 `sample_path` 已经有完整 YAML，停下来。不要另写一条待补记录覆盖它。
3. 选择任务代码。路径、`id` 和 `task` 使用同一代码。
4. 只填写上一节允许的事实。描述从 README 的算法说明改写，去掉性能数字。
5. 运行：

```bash
MODEL_ZOO_SAMPLE_REF=refs/remotes/origin/develop python3 model_zoo_web/scripts/build_catalog.py
npm --prefix model_zoo_web run check
```

发布预览（待补卡片会出现在这里）再跑 `npm --prefix model_zoo_web run build:release` 和 `npm --prefix model_zoo_web run check:release`。

6. 打开目录和一条待补详情，确认筛选项、板卡和“数据待补充”一致，且没有下载按钮和数字。

## Output

登记时说明：示例路径、网页 YAML 路径、状态（complete 或 pending）、写入了哪些板卡、每块板卡的证据路径、故意没写的字段。跳过的目录要写出原因。

## Safety

未知就保持未知。不要用邻近模型、同一系列的另一尺寸、或 S 清单的总兼容行填数字和板卡。不要把 `sha256: null` 写成一个哈希。子模块示例只链接仓库里的 gitlink 路径，不要假装子模块里的文件已经在本仓库展开。
