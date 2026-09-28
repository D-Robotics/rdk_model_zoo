---
name: rdk-model-zoo-release
description: "Use when preparing or checking RDK Model Zoo model releases, Skills Pack releases, manifests, Hub registration migration, tags, or release isolation. 触发词：模型发版、Skills 发版、Hub 接入、Tag 兼容。Do not use for routine sample development, OE Pack releases, or publishing without explicit authorization."
version: "1.0.1"
license: Apache-2.0
metadata:
  author: "RDK Model Zoo maintainers"
  content-license: "CC-BY-4.0"
  pack: "rdk-model-zoo"
  data-classification: "public"
---

# RDK Model Zoo Release

## Purpose

准备可追溯、相互隔离的模型与 Skills 发布。默认只做检查和发布计划；本 Skill 不提供未经审批的自动发版权限。

## When to use

模式 `model-release`：目标平台模型版本、Manifest、Benchmark 和目录发布。
模式 `skills-release`：本仓库 skills/ 的独立版本、结构/行为验收和正式 Release。
模式 `hub-onboarding`：组件注册、同名入口维护源迁移和安装/升级验收。

请求未指明“模型”还是“Skills”且无法从文件范围判断时，先明确发布对象，不默认修改根 VERSION。

## Instructions

1. 读取 [context-policy.md](references/context-policy.md)、目标仓库发版规范和 [release-contract.md](references/release-contract.md)。`rdk_x5` 仅是 Pack 维护源；按目标 ref 识别 X5、S100/S100P/S600、X3 或 legacy，并核对该 ref 的 Manifest（有清单时）。记录发布类型、候选提交、版本、原 Tag/Latest/Manifest 状态和权限；用户指定的对象冲突时不切换分支。
2. `model-release` 只修改该平台版本线的 VERSION、CHANGELOG、适用布局的 Manifest 与发布说明；已发布 Tag 不变。遵循当前 S/X3 固定来源与 X5 目录发布流程，不把静态清单构建写成全量板测。
3. `skills-release` 读取 skills/VERSION、技能自身版本与 skills/CHANGELOG；严格区分 Pack 版本和单 Skill 版本。裸 vMAJOR.MINOR.PATCH 只表示 Skills Pack；模型 x5-v*/s-v*/x3-v* 保持原样。核对技能自包含、引用、行为评测和许可/来源。
4. 验证发布隔离：Skills 不改模型 Manifest/根 VERSION；非模型事件不进入 pages 并发组；Skills Release 正式、非草稿、非预发布，显式 latest=false。附注 Tag 指向批准的真实提交；不能把旧模型 Tag 重命名为 Skills Tag。
5. `hub-onboarding` 核对组件 ref/源路径/catalog_dir 唯一。先发布真实 Skills 版本，再在同一个 Hub PR 中移除 Device 的 rdk-model-zoo 条目并新增 Model Zoo 组件。注册变更、镜像和生成物一起验证，不能依赖每小时同步会自动完成迁移。
6. 默认输出 [release-checklist.md](assets/release-checklist.md)，列命令、审批点和失败回退。仓库级规则、App/Environment、发布通知和标签配置只读取；没有权限不能声称已配置。
7. 只有用户明确授权了准确对象、提交、版本和副作用，才进入实际 Tag/Release/Hub 写入流程。使用只增不改的附注 Tag，创建 Release 需 verify-tag；Skills 必须 latest=false。发布前重验候选与审批后状态。
8. 半发布失败保留 Tag；核对精确提交和原批准说明后才恢复缺失 Release，不删除重打。不移动旧 Tag 回滚；已发布缺陷用新补丁版本，Hub 回指旧合法 ref 需显式批准且不能伪装更高版本。

## Output

发布对象与候选、检查证据、未满足项、需审批动作、回退和发布后核对项。没有执行就写“发布准备完成/未发布”，不能输出不存在的 Release URL。

## Safety

禁止强推/移动/删除已发布 Tag；不将 GitHub 写权限视为批准。不开启自动合并、不绕过保护、不共享 App 密钥。模型和 Skills 发布/通知并发及 Latest 相互隔离；历史内容保留。
