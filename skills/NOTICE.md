# 来源、迁移和许可记录

本候选包由已批准的“Model Zoo 内维护 + Hub 分发”设计编写。不是直接改动上游仓库，也不是已完成的线上迁移。

## 参考与继承

1. 现有 `rdk-model-zoo`：维护源 D-Robotics/rdk-device-skills；本次依据 Hub `131d3048d5b1b8012b1383dc70be4f8264e25918` 所镜像入口的范围和交接规则重写。保留同名用户入口，成员版本提升到 1.1.0。
2. 现有 `rdk-model-zoo-demo-review`：参考 maxma615/skills `d45931b688722fd9895558cec860b6762a91c1d9` 的双轴评审、证据与默认只读思想；新名称为 `rdk-model-zoo-review`，增加明确差异范围和正确性/回归维度。
3. Model Zoo 规范与发布源码：以本包记录的 X5/S/X3 提交为调查基线；Skill 要求执行时读取实际目标版本，不复制平台支持表作为永久事实。
4. RDK Skills Hub：引用其平铺分发、治理卡、评测与独立发布约定。工具链收据保留上游格式，不新增替代 PTQ/QAT。

## 迁移差异（需要维护者在 PR 中复核）

旧 branch_selector 的静态选择不再作为硬件结论；改为真实 checkout 检查与明确平台确认。旧 benchmark_lookup 的硬编码指标表不复制，新 read_catalog 读取目标版本 Manifest。旧参考表需要逐文件审核后退役或迁移，不承诺旧内部脚本 CLI 原样兼容。

本包是重构后的替换候选，不冒充旧目录的逐字迁移。实际迁移时必须列出旧源完整目录，保留其署名/许可，说明每个脚本/文档/用例保留、替换或退役原因，并测试直接安装用户的更新路径。不能改旧 tag 或抹去历史。

## 文件许可

新写说明、Skill、治理卡、参考与 eval 文档：CC-BY-4.0。新写 Python/配置脚本：Apache-2.0。沿用/复制的上游文件保持原许可。为当前 Hub 兼容，SKILL.md 顶层 `license: Apache-2.0` 同时配套 `metadata.content-license: CC-BY-4.0`；该兼容字段不改变文件内容许可。

- [CC-BY-4.0 条款](https://creativecommons.org/licenses/by/4.0/legalcode)
- [Apache-2.0 条款](https://www.apache.org/licenses/LICENSE-2.0)
- [Hub 贡献规范](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/CONTRIBUTING.md)
