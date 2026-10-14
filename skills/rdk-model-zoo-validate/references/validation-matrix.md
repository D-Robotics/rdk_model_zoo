<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 验证选择矩阵

| 变更或目标 | 最小计划 | 不可声称 |
|---|---|---|
| 仅文案/链接 | 实际路径、语义一致性、所改链接/索引检查 | 重新完成板测 |
| CLI/default/path | 静态参数对照；在可用且安全环境下验证默认/覆盖行为 | AST 解析证明所有命令能跑 |
| 图像预/后处理 | 真实尺寸/色彩/空结果边界；固定输入基线对照；目标板 smoke | 看见图片就证明 mAP 达标 |
| 新模型/新权重 | metadata 与 I/O、目标加载、真实输入、对应 accuracy/evaluator | 仅量化工具返回零就部署成功 |
| 共享 utils | 全调用方影响图、代表性任务/平台回归、必要一致性比较 | 一个 sample 通过等于全仓无回归 |
| Python/C++ 双实现 | 同模型同输入、明确 bitwise 或容差、输出语义、错误路径 | 不同计时范围直接归因为图执行提升 |
| 语音 | 采样率/格式/分块/状态、token/文本基线 | 图像流程参数可直接复用 |
| 多模态/多模型 | 每组件身份与跨组件协议、必要 tokenizer/host 模型 | 只验证一份 BPU 文件代表完整 pipeline |
| 机器人策略 | 观测历史/动作语义/离线输出、双 runtime 合同 | 离线通过证明闭环或执行器安全 |
| 性能 | 相同模型/输入、真实环境、warmup/repeat/concurrency、计时边界 | 缺失条件可推算、跨板卡代填 |

这些是计划选择规则，不是自动要求每个 PR 跑所有测试。先绑定用户指定的目标 ref 和实际 SoC（X5、S100/S100P/S600、X3 或 legacy）；Pack 维护分支不能替代目标。用户交付明确要求的检查必须列 required；环境缺失用 not-run。随机张量 perf 与数据集 accuracy 必须是两行。目标 ref 冲突时保留约束并停止依赖该身份的执行，不切换分支。

## 统一集成线的仓库门禁划分

统一集成线（`develop`）把静态契约、主机验证和目录数据分成三个 CI workflow，计划验证范围时按此对齐，具体以目标 ref 当前 `.github/workflows/` 与 `utils/tools/host_validation/` 为准：

| 门禁 | 命令 | 说明 |
|---|---|---|
| sample-contract（每个 PR） | `python utils/tools/sample_contract/check.py --scope maintained --parser-mode import --rule-set delivery` | 维护 sample 清单的交付规则：双语配对、章节锚点、本地链接、CLI 默认值、中英表格一致；Ubuntu 24.04 + Python 3.12 单腿，报告写 runner 临时目录 |
| host-validation（默认单 Linux 腿） | `python utils/tools/host_validation/run.py --repo . --report <path>` | 全部第一方 Python 套件、sample 覆盖、host-safe 原生 CTest 与历史 pin；CI 中以 `--skip-catalog`/`--skip-contract` 跳过由专属 workflow 拥有的部分，避免同一检查跑两次 |
| model-catalog-data | `npm --prefix utils/tools/catalog-publisher run check` | 目录数据校验、测试与可复现构建；仅当目录自身输入变化时运行 |

模板章节顺序与阶段纯净度属于 strict 规则集，只进入本地完整门禁（`run.py` 的 strict 契约段；检查器 `--rule-set strict` 可单独运行）。纯文档变更（含 sample README）经显式分类步骤跳过原生/完整门禁，其 CLI/链接契约仍由 sample-contract workflow 在每个 PR 上强制。本地全量门禁保持可用；本地绿灯不是 CI 结论，Linux 结果由 CI job 证明。`workflow_dispatch` 可用 expanded 矩阵恢复 Python 3.10 与 macOS 兼容腿。

板端操作只在批准环境中执行。运行来自外部 PR 的代码时，先做静态安全审阅，使用隔离临时工作区和受控板卡，不暴露长期凭据。不要在通用 pull_request_target job 中执行 head 代码。
