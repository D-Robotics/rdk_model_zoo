---
name: rdk-model-zoo-validate
description: "Use when planning or executing sample-scoped RDK Model Zoo smoke, numerical, accuracy, performance, or regression checks and preparing verification evidence. 触发词：样例验收、回归测试、数值一致性。Do not use as a PR verdict, a published-benchmark lookup, or a quantization implementation."
version: "1.1.0"
license: Apache-2.0
metadata:
  author: "RDK Model Zoo maintainers"
  content-license: "CC-BY-4.0"
  pack: "rdk-model-zoo"
  data-classification: "public"
---

# RDK Model Zoo Validation

## Purpose

为目标样例提供可复现、分范围的验证计划和证据。组织既有 evaluator 与平台工具，不重复实现通用 Benchmark，也不把生成 JSON 当成真实测试。

## When to use

适用：跑 smoke、比较 Python/C++、精度对照、测当前模型性能、验证修复、组织缺板卡情况下的可执行计划。查询已发布数字用 `rdk-model-zoo`；判断 PR 是否满足规范用 review；量化调优交工具链。

## Instructions

1. 读取 [context-policy.md](references/context-policy.md) 和 [evidence-contract.md](references/evidence-contract.md)。确定目标提交及 dirty 内容、实际平台/SoC（X5、S100/S100P/S600、X3 或 legacy）、模型/输入/runtime、交付条件。`rdk_x5` 维护源不能替代目标 ref；用户约束冲突时保持 not-run 并不切换分支。
2. 按 [validation-matrix.md](references/validation-matrix.md) 列出检查与 required 属性，先声明阈值和范围。没有板卡、私有数据或可信模型时只阻断相应行；其余 host 检查可继续。验证 README 命令与 API 示例时，先把文档代码块按 **说明 / 主机 / 板端 / 转换** 分类：主机类在授权内实际执行并绑定代码 SHA、cwd 与制品/输入身份；板端与转换类只形成待执行计划交对应执行者；纯说明文字不冒充已验证命令。结构校验通过不是命令验证。
3. 选择目标版本已有脚本/evaluator/构建命令。先读源码和文档，再确定 cwd、argv、输出和副作用；`--help` 也不能盲目执行。首次范围不自动扩成全仓下载和重测。
4. 实际执行前确认权限、环境、空间与输出隔离；有板卡串行占用策略时遵守，不在共享板卡上任意停进程或改频率。PR 不可信代码不得接触生产凭据和长期自托管 runner。
5. 保存退出码、日志、模型/输入身份和结果；准确对照判定条件。性能测量记录真实计时口径；工具链 perf/UCP 等按当前实际可用平台工具委托，不能把 X5 工具用于 S 产物。
6. 用 [verification.template.json](assets/verification.template.json) 记录结果，schema 位于 [verification.schema.json](schemas/verification.schema.json)。不执行的行保持 not-run 并给原因，不能填模板性 passed。
7. 可用只读校验器检查报告结构及证据哈希：
   ```bash
   python3 "$SKILL_ROOT/scripts/validate_evidence.py" "$RECEIPT" --evidence-root "$EVIDENCE_ROOT"
   ```
   `RECEIPT` 是实际报告路径，`EVIDENCE_ROOT` 是其中相对证据路径的根。不提供该根时只做结构检查；退出 0 不是模型通过证明。
8. 汇总每行状态、未覆盖范围和必需门禁；将报告交给 review。失败时交 develop 或对应工具链诊断，保留失败输入，不自动降低阈值重判通过。

## Output

验证矩阵、精确命令和证据、条件/指标、通过与失败、未运行原因。结构通过、host 通过、单项 board 通过和整体交付完成分别陈述。

## Safety

除审阅过且获授权的测试外不修改系统，不伪造数值/哈希/板测。脚本校验不执行 JSON 中的命令、不上传文件。机器人样例只验证离线输出；闭环安全需要额外明确批准的流程。
