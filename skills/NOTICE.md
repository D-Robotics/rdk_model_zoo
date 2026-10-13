# 来源与许可记录

本包在 Model Zoo 仓库内维护。

## 参考与继承

1. `rdk-model-zoo`：维护源 D-Robotics/rdk-device-skills；依据 Hub `131d3048d5b1b8012b1383dc70be4f8264e25918` 所镜像入口的范围和交接规则重写。保留同名用户入口。
2. `rdk-model-zoo-review`：参考 maxma615/skills `d45931b688722fd9895558cec860b6762a91c1d9` 的双轴评审、证据与默认只读思想。
3. Model Zoo 规范与发布源码：以本包记录的 X5/S/X3 提交为调查基线；Skill 要求执行时读取实际目标版本，不复制平台支持表作为永久事实。
4. RDK Skills Hub：引用其平铺分发、治理卡、评测与独立发布约定。工具链收据保留上游格式，不新增替代 PTQ/QAT。

## 文件许可

新写说明、Skill、治理卡、参考与 eval 文档：CC-BY-4.0。新写 Python/配置脚本：Apache-2.0。沿用/复制的上游文件保持原许可。为当前 Hub 兼容，SKILL.md 顶层 `license: Apache-2.0` 同时配套 `metadata.content-license: CC-BY-4.0`；该兼容字段不改变文件内容许可。

- [CC-BY-4.0 条款](https://creativecommons.org/licenses/by/4.0/legalcode)
- [Apache-2.0 条款](https://www.apache.org/licenses/LICENSE-2.0)
- [Hub 贡献规范](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/CONTRIBUTING.md)
