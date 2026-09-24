<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Model Zoo 与工具链 Skills 的交接

## 职责分界

Model Zoo 负责样例选择、接口合同、代码/文档规范、接入、回归和交付。量化、编译、敏感度/精度调优、工具链安装等由既有 OE Pack 承担。不要创建另一套 PTQ/QAT，也不要把完整仓库任务原封不动交给工具链 router。

## 选择能力，而不是匹配后缀

先根据 `REPO_ROOT` 的 README、SoC、用户约束和（有清单时）Manifest 确认目标，再选择工具链；`rdk_x5` 只是本 Pack 的维护源，不能触发 X5 工具链。目标明确为 X5 且需要转换/编译/底层诊断时检查 `x5-router`；S100、S100P、S600 按目标实际可用的 S 系列 `drobotics-router` 和版本检查；X3 不借用 X5 Pack，先用 `rdk-docs-reference` 查目标官方文档；legacy 继续使用其历史 ref 对应的工具链说明，不自动升级成当前 X5/S 流程。`.pt` 可能是权重、TorchScript、QAT 产物等；先识别内容及用户目标，不凭扩展名自动开始 QAT。

只运行已安装、可读取且与当前宿主兼容的 Skill。引用一个名字不等于已经调用。缺少 router 时：解释需安装的 Pack、写入位置、权限和会话加载要求；通过可用 `rdk-pack-installer` 按确认流程安装。安装未授权、网络失败或会话不可用时，保存未执行交接，不默默改成手写工具链命令。

完整部署工作流和单步编译的验收不同。需要编译子步骤就以精确范围委托；采用某 Pack 的完整部署工作流，就必须保留它的完整完成条件。UCP 交付要求不自动成为 Model Zoo 所有 Python sample 的规范，也不能省略 UCP 后仍声称该完整工作流成功。用户指定的目标 ref、平台或工具版本与候选 router 冲突时，不切换分支来凑条件，先报告冲突并保留未执行交接。

## 交出前填写

目标 repo/commit/sample、平台/SoC、任务、runtime API/版本；原模型/定义/上游提交；每个输入输出的 name/shape/dtype/layout/语义；host 与编译器各自承担的预处理/反量化；类别/标签、后处理；校准数据来源与许可、浮点验证基线和容差；输出根、资源预算、允许/不允许的安装覆盖上传动作。

未确定值用 null 或明确问题，不编造。Model Zoo sample 的导出脚本可以作为目标模型协议的参考，但必须读实际代码；不假设工具链的通用导出必然匹配该 wrapper。

## 返回后验收

读取原始收据和模型 metadata；复核目标架构、真实格式、runtime 支持、产物哈希、输入输出和验证条件。下载模型哈希不能自动升级为发布者可信哈希。Plugin HBM/HBIR 与 Mapper BIN 不能改后缀互换；由匹配版本和实际执行证据判断兼容。

继续由 Model Zoo 接入 wrapper/run.sh、文档、evaluator 和必要 Manifest。模型编译/工具链验证完成与样例接入完成分别报告。无法适配时指出是哪一个 I/O 或 runtime 条件阻断，不无授权地改模型图、换 runtime 或降低验收指标。

## 来源

- [X5 router 与范围](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/skills/oe-skills-x5/skills/x5-router/SKILL.md)
- [X5 PTQ 完成条件](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/skills/oe-skills-x5/skills/x5-ptq-deploy/SKILL.md)
- [X5 QAT 产物边界](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/skills/oe-skills-x5/skills/x5-qat-compile/SKILL.md)
- [S 全链路部署范围](https://github.com/D-Robotics/oe-skills-s/blob/v1.1.0/drobotics-s/skills/drobotics-router/references/deployment-workflow.md)
