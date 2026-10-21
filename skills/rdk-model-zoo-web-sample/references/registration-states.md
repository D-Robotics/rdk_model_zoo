# 登记状态

路径固定为：

```text
model_zoo_web/data/<domain>/<source>/<family>/<task>.yaml
```

`id` 必须是 `<source>/<family>/<task>`。`family` 是系列。一个示例目录可以有多条记录，例如同一 `sample_path` 下的检测器和识别器，或 YOLOE-11 与 YOLOE-26。

## 待补

`registration: pending`。校验器拒绝任何测量和制品字段，所以“先填个占位数字”无法通过构建。

| 字段 | 要求 |
| --- | --- |
| `schema_version` | 必须为 `1` |
| `registration` | 必须为 `pending` |
| `id` `name` `domain` `source` `family` `task` | 必填。`task` 是 `task-registry.json` 的键 |
| `description.zh` `description.en` | 必填。只描述架构或示例流程 |
| `sample_path` | 必填。`samples/` 下的目录，或 `sample_kind: gitlink` 的子模块 |
| `sample_revision` | 必填。40 位提交，且本仓库能解析到该提交 |
| `supported_platforms[].platform` | 必填，至少一个，取 `x5` `s100` `s100p` `s600` |
| `supported_platforms[].evidence` | 必填。`docs/release/` 或 `samples/` 中的真实路径 |
| `license` | 可选。名称必须出现在 `source` 文件里。`url` 要么也出现在该文件，要么是该名称的官方许可证页 |
| `provider` | 可选。只写 README 点名的上游 |
| `upstream` | 可选。每一项的 URL 必须出现在它的 `source` 文件里 |
| `published_variants` | 可选。变体名必须出现在 `source` 文件里 |
| `conversion_configs` | 可选。`path` 必须是该提交里的文件；march、输入类型、分辨率必须出现在该文件或路径里 |
| `inputs` | 可选。shape、layout、format 必须出现在 `source_id` 那一条基准记录里 |
| `usage` | 可选。每一行命令必须出现在 `source` 文件里 |
| `sourced_benchmarks` | 可选。数字、单位、条件和变体名必须出现在 `id` 那一条基准记录里 |
| `documented_downloads` | 可选。文件名和 URL 必须出现在 `source` 清单里。只在能点名板卡时填写 `platform` |
| `example_image` | 可选。`path` 与 `source` 相同，并且是该提交里的图片文件 |
| `sample_kind` | 可选。默认 `directory`；子模块必须是 `gitlink` |
| `manifest_tasks` | 可选。发布清单中的任务 id |
| `facet_tasks` | 可选。覆盖的筛选项；缺省时用该任务代码自己的筛选项 |
| `variants`、`accuracy`、`performance`、`assets`、`benchmarks`、`cover` | 禁止。这些名字属于完整发布记录 |

待补记录在发布预览里仍是 `releaseStatus: data-pending`：没有正式 `assets`。详情页和已发布模型用同一套结构。类别和“数据待补充”在同一行，标题、说明、按钮紧随其后。获取模型用同一个主按钮：仓库清单里有真实 HTTPS 下载就链过去，多条时打开这些链接；没有就禁用，提示“模型产物待发布”，不编造制品。主图用和已发布页一样的白底卡片；只有仓库里真有推理结果图时才带结果侧栏，不编造 Top-K。性能瓦片与该任务已发布页相同：分类是 Runtime 延迟、端到端延迟、吞吐量、Top-1、Top-5；检测、分割、姿态、旋转框用各自已发布的指标。没有已发布对照的任务仍是这三块性能瓦片，再加 [field-matrix.md](field-matrix.md) 里写明的两个任务指标，凑满五格，不用笼统的“精度”。没有原文的格子数值为“数据待补充”。Runtime 并发缺省显示“Runtime 单路”，不写成禁用的“数据待补充”。转换详情用和已发布页相同的“查看转换详情”和图标；没有报告时同一文案置灰。端到端延迟的 `?` 只保留已发布的七行（实现、CPU 线程、流水线、Runtime 提交、前处理、后处理、端到端吞吐量）。条件、来源和“摘自仓库文档”写在对应数字自己的 `?` 里。模型参数固定为参数量、浮点计算量、源模型输入、Runtime 输入、量化精度。源模型输入用和已发布页相同的格式（格式、dtype、布局、shape，用 ` · ` 连接），只填 `source_input` 或转换配置里能核对到的字段。相关模型先取同一任务，没有再取同一领域。钉住文档里有逐字的下载命令或运行命令时，主栏用和已发布页相同的「快速上手」：步骤、复制按钮和「完整说明」。命令必须是原文，不改写成 wget。没有原文的步骤省掉；下载和运行都没有就整段省掉，已发布的 ResNet 也是这样。上游、转换配置和文档下载仍不另开主栏，放进侧栏的开发资源（模型仓库、运行文档、模型转换、论文或上游仓库）和模型许可。封面只用干净的样例图；代码或终端截图改为统一的中性占位图。

`npm --prefix model_zoo_web run check` 仍然构建不含模型的空壳，并校验 YAML。访客看到的列表来自 `build:release` / `check:release`。

## 完整

省略 `registration`。这是当前 MobileNet、ResNet 和 Ultralytics YOLO 记录的形状。`scripts/build_catalog.py` 的 `validate_record` 要求：

- 许可证名称和官方 `https` 页面
- 每个规格的参数量、GFLOPs、源模型输入（格式、dtype、布局、四维 shape、scale）和宽高
- 每个平台 `status: released`、`released_at`、制品 URL、大小、SHA256、`release.json`、`SHA256SUMS`
- 与制品同目录的 `oe_report.html`
- 任务精度字段表要求的 float 与 runtime 指标，取值在 `[0, 1]`
- Runtime 性能：工具、口径、单线程和多线程测量
- C++ 端到端：单路与多路流水线、分段时间和可由墙钟复核的吞吐

平台状态不能写成 `candidate`。候选批次留在 `candidate-staging/`，不进 `data/`。

封面和结构化 OE 数据通过 `model_zoo_web/release/inputs.json` 引入，不把占位图写进待补记录。

## 从待补到完整

1. 按任务矩阵备齐证据。没有精度字段表的任务先扩展 `accuracy-metrics.json`。
2. 删除 `registration` 以及只属于待补的 `supported_platforms`、`sample_kind`、`manifest_tasks`、`facet_tasks`。板卡改写进 `variants[].platforms`。
3. 把真实封面和 OE 数据登记进 `release/inputs.json`。
4. 重新构建目录和发布预览。原来的待补卡片会被这条完整记录替换，而不是并列两条。
