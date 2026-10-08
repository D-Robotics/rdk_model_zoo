[English](README.md) | 简体中文

# Agent 行为评测

七个 Model Zoo Skills 共定义 75 条核心行为用例，分别保存在各 Skill 的
`evals/tasks.yaml`。用例记录输入、fixture 和预期路由或行为；评测结果与逐条
证据见[行为报告](REPORT.md)。归档校验值见
[SHA256 文件](codex-evidence-2026-09-17.sha256)。

## 评测记录

每条结果记录 `run_id`、`case_id`、评测变体（`baseline`、`agents-only`、`full`）、
Agent/模型/版本、fixture 摘要、实际主 Skill、工具轨迹和产物路径、断言状态及
reviewer。断言状态使用 `pass`、`fail`、`fixture-invalid`、`not-run`；无法执行的
断言保留原因。负向路由中的 `expect.skill: none` 表示该 Skill 不作为主 Skill。

评测使用固定目标仓库、提交和输入条件，并记录 Agent 配置、可用工具及会话上下文。
每种变体使用独立会话；合成情境使用独立临时 fixture。评审依据实际工具调用、文件
变化、日志和最终产物逐项核对预期行为。

## 当前报告

[行为报告](REPORT.md)记录 75 条核心用例的执行情况、对照结果、评分状态和已知限制。
它也链接到包含输入、提交身份、工具事件、文件变化和评分记录的归档。修改 Skill 后，
按受影响用例及相关负向路由场景开展新的 Agent 评测，并在对应报告中记录结果。
