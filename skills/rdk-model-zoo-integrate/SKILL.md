---
name: rdk-model-zoo-integrate
description: "Use when integrating a custom model artifact or changed I/O contract into an RDK Model Zoo sample, including class-count, shape, wrapper, or cross-platform adaptation. 触发词：自训练接入、替换权重、接口适配。Do not use to implement PTQ/QAT or to review an unchanged sample."
version: "1.0.0"
license: Apache-2.0
metadata:
  author: "RDK Model Zoo maintainers"
  content-license: "CC-BY-4.0"
  pack: "rdk-model-zoo"
  data-classification: "public"
---

# RDK Model Zoo Integration

## Purpose

把模型和目标 sample 的输入输出、runtime、前后处理对齐。编译由既有工具链负责，本 Skill 负责交接前的需求与交接后的仓库接入。

## When to use

适用：自训练模型替换、类别/尺寸/输出节点变化、已量化产物接入、跨平台适配、encoder/decoder 组合。纯粹使用官方默认样例用 `rdk-model-zoo`；纯 PTQ/QAT 子任务委托工具链；完整新 sample 交付由 `rdk-model-zoo-develop` 主导。

## Instructions

1. 读取 [context-policy.md](references/context-policy.md)，确认目标 sample、平台/SoC、模型来源、runtime 与用户目标。`rdk_x5` 维护源不等于 X5 目标；按实际 ref 区分 X5、S100/S100P/S600、X3 和 legacy。用户指定的平台、版本或路径冲突时保留约束，不切换分支。区分私有应用接入和公开仓库贡献；后者才要求仓库级索引/发布清单等交付。
2. 阅读目标 export/conversion、wrapper、main、run.sh 和模型文档。按 [integration-contract.md](assets/integration-contract.md) 建立源模型与目标 runtime 的 I/O 对照；每个输入输出分别记录，不能只比较文件名。
3. 已有可用产物先核对架构、格式、模型 metadata、shape/layout/dtype、输入名、输出次序、量化/反量化责任。YOLO 类别数变化同时审查标签、输出 reshape 和解码；多模型任务逐组件核对。缺少适配证据时不强行运行。
4. 需要转换时按 [toolchain-handoff.md](references/toolchain-handoff.md) 交出精确子任务。检查对应 router 和完整资源树可用；安装或升级另行确认。将输入协议、基线/容差、输出目录和授权边界传递，不复制量化实现。
5. 接回原始模型、配置、metadata、日志和收据，核验哈希和验证范围；不把新格式改后缀伪装为默认模型。缺兼容性证据时仅推进不依赖它的文档/静态工作。
6. 按已批准范围修改配置/标签/wrapper/main/run.sh；查已有 utils 的语义后复用。保留用户数据与上游版权，不提交私有权重、校准数据或临时 dump。必要接口变更明确记录影响。
7. 用同一可复现输入对照浮点基线、工具链结果和接入后结果；根据 [evidence-contract.md](references/evidence-contract.md) 分开记录。工具链旧收据不能替代修改后的前后处理验证。
8. 公开贡献时交 `rdk-model-zoo-develop` 完成双语文档、conversion/evaluator、模型获取和索引；再用 `rdk-model-zoo-review` 审阅。不为私人实验强制生成整个仓库贡献。

## Output

I/O 对照、委托及返回证据、文件变更、sample 层验证、剩余阻断。分别报告“工具链产物状态”和“样例接入状态”，不把编译成功写成部署成功。

## Safety

模型、训练数据和标签默认私有；无授权不上传、不覆盖、不升级 OE 工作区。机器人输出默认离线文件，不直接应用动作。需要改变 runtime/图结构/容差时明确提出变更，不静默降低交付标准。
