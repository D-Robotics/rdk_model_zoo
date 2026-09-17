<!-- GENERATED from skills/_shared/evidence-contract.md; edit source then run skills/tools/sync_references.py --apply. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 分范围验证与交付证据

## 验证不是单个等级

每个检查绑定：目标仓库 ref/提交或脏工作区内容、sample、实际平台/SoC、模型变体和文件哈希、任务、语言/runtime、输入/数据集和检查类型。一个 Python detect 结果不能覆盖 C++、seg、S100、S100P、S600、X3、legacy 或其他权重。

层次使用 `static / host / board`，检查结果使用 `passed / failed / not-run / not-applicable`。必需检查以 `required: true` 记录；缺板卡使用 not-run + reason，而不是 not-applicable。分阶段证据不能自动升级成全仓兼容性认证。

## 人工/Agent 工作流

先列验证计划、命令工作目录和权限，再执行。保存准确 argv、cwd、开始结束时间、exit code、环境摘要、输入和输出身份、日志和判断条件。stderr 也是证据；避免用管道丢失真实退出码。输出图像要检查本次生成时间/内容，不能用仓库内已提交截图证明刚运行成功。

精度比较先定义数据集、训练/量化基线、指标和容差。只有数值逐位一致明确为适用要求时才用 bitwise；其他任务容差须有依据。随机输入/单张图只能证明限定 smoke 或吞吐，不代表数据集精度。校准集与独立评估集的隔离须根据任务记录。

性能记录计时范围、预热、次数、batch、线程/核、同步方式、温度/后台负载等已知条件。编译估算、纯推理、端到端延迟、并发吞吐分开；缺失条件保持未知，不能由 FPS 补造延迟，不能跨板卡复制数值。文档的 `200+` 保留 lower-bound。

用户日志、CI 产物和历史发布记录注明来源，不声称当前 Agent 亲自执行。来源无法绑定目标提交、模型或输入时列待核验。修改 wrapper 后旧工具链收据仍可作为来源，但不能替代修改后的端到端验证。

## JSON 契约

`rdk-model-zoo-validate` 随包提供 `schemas/verification.schema.json` 和模板。字段为：

- `schema_version: 1`, `kind: verification`。
- `target`: repository、commit、dirty、patch_sha256、sample_path。执行过的检查必须有真实提交；dirty 时对实际变更及相关未跟踪输入建立可复现内容快照/摘要，不能只取一个漏掉 untracked 的 git diff 哈希。
- `checks[]`: id、required、level、purpose、status、scope、environment、execution、result、evidence、reason。
- `execution`: argv 数组、cwd、exit_code、started_at、ended_at。不执行时必须为 null。
- `result`: summary、acceptance；数值判断写出具体接受条件与实测结论，并将完整指标放入证据文件。
- `evidence[]`: 相对独立 evidence root 的路径与文件 SHA-256。不是用户数据上传地址。

默认模板全部 not-run，不填虚构哈希/时间/板卡。执行 host 的 accuracy、consistency 或 performance 检查时，也要填写 scope 中的平台、模型变体、任务、runtime、模型哈希和输入哈希；static 结构检查可保持这些字段为 null。缺少验证工具时仍用同样字段输出人类可读报告，不伪造脚本输出。

证据校验器只验证结构，并在提供 `--evidence-root` 时核对文件哈希。它不会运行 receipt 中命令、读取网络或判定模型数值。即使退出 0，其 `board_verified` 仍为 false——板端结论由完整记录和独立审阅支持。

## 与工具链收据的关系

工具链 Pack 可能已有 input/environment/route/plan/artifacts/verification/receipt 等平台专属运行文件。只有目标 ref 和实际平台匹配时才能引用它们；X5、S100/S100P/S600、X3 与 legacy 的收据不能互相代用。保留原始格式，以路径、来源提交和哈希引用，不复制或改写它们来伪装本次成功。本包验证报告只记录新增的 sample 层检查。

[原始 X5 run contract](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/skills/oe-skills-x5/platforms/x5/references/run-contract.md)
