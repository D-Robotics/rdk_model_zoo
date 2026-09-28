<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 发布准备/执行记录

## 身份

release-kind、候选 commit、维护分支、计划版本/Tag、变更范围、批准人/动作边界。准备阶段没有授权，不运行发布命令。

## 前置证据

候选工作区/差异清楚；版本对应自身文件；旧模型 Tag 对象及解引用 SHA 快照；原 Latest；模型 Manifest 摘要；本地结构/单测、Agent 验收、Hub/安装演练的真实状态；目标 Tag/Release 不存在的读取证据。

## 模型发布适用项

平台 VERSION/CHANGELOG、两个 Manifest、对应 release notes、固定跨平台来源、目录构建及源数据一致；未知校验和披露。无仓库级板测不说全部通过。

## Skills 发布适用项

skills/VERSION/pack.json、各 Skill version、变更记录与许可、平铺资源闭包；stable annotated Tag；正式 Release、verify-tag、latest=false；不改模型清单与根 VERSION；Pages 并发隔离先上线。

## Hub 切换适用项

新源版本已公开；删除旧 owner entry 和添加新 entry 在同一 PR；catalog_dir 无重复；镜像只含 Skills；插件/README/安装注册生成物同步；上游直接安装用户迁移说明；notifier 权限/标签与重复事件演练。

## 实际发布记录（只有执行后填写）

批准的命令、Tag 对象/commit、Release URL、发布时间、非草稿/非预发布、Latest 核对、Pages/Hub 事件结果、安装验证。未执行字段使用“未执行”，不生成看似真实 URL。

## 失败回退

合入前：撤回候选改动，不动旧 Tag。Tag 已推：保留，只能补建精确缺失 Release 或发新补丁。Hub 注册出错：获批准后修正/回指已存在合法版本，通过新 PR 与生成校验；不强制移动源 Tag。Latest 错误先报告，经授权恢复目标模型 Latest 并检查根因。
