<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 模型、Skills 与 Hub 的独立发布契约

## 版本边界

| 对象 | 版本位置 | Tag | 说明 |
|---|---|---|---|
| X5 模型 | 目标 X5 ref 的根 VERSION（常见 `rdk_x5`） | x5-vMAJOR.MINOR.PATCH | 保留目标 ref 的模型流程 |
| S100 / S100P / S600 模型 | 目标 S ref 的根 VERSION（常见 `rdk_s`） | s-vMAJOR.MINOR.PATCH | 按 Manifest 的 SoC 范围分别核对 |
| X3 模型 | 目标 X3 ref 的根 VERSION（常见 `rdk_x3`） | x3-vMAJOR.MINOR.PATCH | 保留目标 ref 的历史布局兼容 |
| legacy 模型 | 用户指定的 legacy ref/tag | 该 ref 已有的模型 Tag | 不把历史交付重命名为当前平台版本 |
| Model Zoo Skills | skills/VERSION | vMAJOR.MINOR.PATCH | 独立 Pack 版本；不是全模型统一版本 |

Tag 标识整个提交，不是子目录。Hub 只镜像注册目录。Skills Pack 1.0.1 中的 rdk-model-zoo 为 1.1.1，以区分旧入口 1.0.0；其他六个 Skill 为 1.0.1。Pack 与单 Skill 版本可以不同，不应校验成永远相等。

裸 v* 的占用须发版前重新核验。Hub 当前 canonical stable 解析只接受严格 vMAJOR.MINOR.PATCH，不接受 skills-v*、branch、预发布或带数字前导零的版本。不能为了方便改变既有模型 Tag。

## 发布模型

从用户指定的目标 ref 读取 `docs/RELEASE*` 流程，优先检查当前 `docs/manifests/`；历史目标 ref 可能使用 `docs/release/` 或根 `release/`。有 Manifest 时，模型清单和 benchmark 保持 `source_ref`、平台、SoC 范围和版本一致，不能用维护源 `rdk_x5` 的 Manifest 代替 S100/S100P/S600、X3 或 legacy 目标；无 Manifest 时按目标 README、下载脚本和实际 sample 记录缺失。模型下载哈希缺失继续披露；不把新 Skill 检查当全量板测。目录聚合按各目标已批准固定 Tag 取数。

## 发布 Skills

发布前：skills/VERSION 与 Pack 元数据一致；技能各自 version 与变更一致；完整单元/结构/引用检查；每个 Skill 独立镜像后能读齐必需资源；核心 Agent 行为验收及已知限制审阅。root VERSION 和模型 Manifest 不因 Skills 改动。

发布物：已批准提交上的附注 stable Tag + 正式 GitHub Release。遵循当前 Hub 的标题/说明规范；按当前规范标题为 `RDK Skills vX.Y.Z`，正文首段标明 Component: RDK Model Zoo Skills。显式 `--verify-tag --latest=false`，既不创建隐式 lightweight Tag，也不抢 X5 Latest。非草稿、非 prerelease；不能用 prerelease 绕过 Latest。

Tag 与 Release 不是原子操作。半失败时记录已经存在的 Tag 对象、commit 和批准说明摘要；只允许为同一不可变身份补建缺失 Release。来源变了就停止，不能删 Tag 重来。

## Pages 与通知隔离

发布前读取候选 ref 中实际的 Pages workflow、触发器和并发配置；不能把某次观察到的 `release.published` 行为当成永久约定。模型事件只在确认的模型范围内进入 Pages，Skills 使用带 `run_id` 的独立忽略组且不取消模型任务。Skills notifier 仅接收 canonical `v*`；模型、S/X3、legacy/archive 事件不派发组件升级，具体过滤条件以当前 workflow 和批准变更为准。

仅 Git Tag 发布不等于 Hub 自动同步：正式源 Release、组件注册、同步/生成、源通知 App/权限、必要标签、CI 与审批都要验证。首次 onboarding 可在人工 Hub PR 中同时变更注册和生成物；后续 notifier 另建受控集成。不要声称现有小时任务会自动完成所有新组件上线。

## 维护源迁移

Hub 已注册 components.d/rdk-model-zoo.yml；新版本只更新该组件的 ref、镜像和生成物。catalog_dir 全局唯一；一个 repo 只能映射一个 component。源头保留上游署名、许可与历史。直接从旧仓库安装的用户需处理旧副本；Hub 同名入口不变。

平铺 Skills 不需要 OE workspace 安装器。不能把本 Pack 放入 OE 升级会删除的 `.drobotics-x5/` 或 `.drobotics-s/`。跨平台读取目标 checkout 的代码和规范；安装源维护分支不代表运行平台，独立安装的 Skill 必须携带自身的 references/assets。

## 来源

- [Model Zoo 模型发版规范](https://github.com/D-Robotics/rdk_model_zoo/blob/529aece791b8f6a21cf93a14e1caa3edbeb11995/docs/RELEASE_cn.md)
- [Pages 原工作流](https://github.com/D-Robotics/rdk_model_zoo/blob/529aece791b8f6a21cf93a14e1caa3edbeb11995/.github/workflows/model-catalog-pages.yml)
- [Hub component 约定](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/components.d/README.md)
- [Hub 版本校验源码](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/.github/scripts/release_contract.py)
- [Hub 发布与授权](https://github.com/D-Robotics/rdk-skills/blob/131d3048d5b1b8012b1383dc70be4f8264e25918/docs/RELEASING.md)
- [GitHub Release CLI](https://cli.github.com/manual/gh_release_create)
- [GitHub 并发](https://docs.github.com/en/actions/concepts/workflows-and-actions/concurrency)
