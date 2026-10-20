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
| `license` | 可选。填写时名称和官方 `https` 页面都要有 |
| `provider` | 可选。只写 README 点名的上游 |
| `sample_kind` | 可选。默认 `directory`；子模块必须是 `gitlink` |
| `manifest_tasks` | 可选。发布清单中的任务 id |
| `facet_tasks` | 可选。覆盖的筛选项；缺省时用该任务代码自己的筛选项 |
| `variants` 及精度、性能、制品、封面 | 禁止。出现即校验失败 |

待补记录在发布预览里变成 `releaseStatus: data-pending` 的条目：每块已登记板卡一条，共用一个目录卡片。没有 `coverImage`、没有 `assets`、没有 `benchmark`。详情页显示“数据待补充”和“基准数据待补充”，下载按钮不出现。

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
