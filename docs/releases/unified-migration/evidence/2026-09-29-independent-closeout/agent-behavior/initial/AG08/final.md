以下为按 `rdk-model-zoo-repo` 技能（SKILL_ROOT：`/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/.coordination/20260928-skills-behavior/AG08/skill`，版本 1.1.1）完成的只读盘点。未修改任何文件。

## 结论速览

X3 目录（`platforms/x3/`）是**冻结的历史分发快照**，登记在案但**不属于本轮 X5/S 统一迁移的新增适配范围**；本轮**不应新增 X3 后端**。唯一与 X3 相关的前瞻项是 ADR 0007 的“有限可行性评估”，且明确不是已批准的完整后端。

## 一、地位（已观察事实）

| 事实 | 来源 |
|---|---|
| “X3 remains archived in its original layout. Historical tags retain their original layouts.” | `platforms/registry.json:3`（description 字段） |
| “X3 保留为历史分发，不属于本轮新增适配范围。” | `platforms/README_cn.md:5`（英文版 `platforms/README.md:5` 同义） |
| X3 在注册表中是正式登记的平台：`id: x3`、`path: platforms/x3`、交付线 `rdk_x3`、标签 `x3-v1.1.2`（目录内 `platforms/x3/VERSION` = `1.1.2` 一致）、runtime `hobot_dnn / bpu_infer_lib_x3`；`guidelines: null`、`license: null` | `platforms/registry.json:37-52` |
| “X3：保留 `demos/`、`resource/`、`release/` 的历史结构，不强套 X5/S 新目录规范。” | `platforms/README_cn.md:34` |
| “X3 的 20 个 Notebook 是历史资源，不能在本表中改动或算入 X5/S 迁移完成度。” | `docs/releases/unified-migration/x5-s-migration-map.md:130` |
| “X3 上游没有随附许可文件，未在迁移中补造。” | `platforms/README_cn.md:44` |
| 当前统一硬件身份注册只含 x5/s100/s100p/s600，**无 X3 条目** | `docs/release/platforms.json:4-9` |
| 版本线规则：X5/S/X3 三线独立 SemVer，X3 为 `rdk_x3` / `x3-vMAJOR.MINOR.PATCH` | `platforms/x5/docs/RELEASE.md:7,13` |

即：X3 是“登记在册、冻结归档”的历史平台树，不是统一迁移的开发目标；这正符合技能 SKILL.md 第 4 条“旧 X3 demos 或历史 release 路径在其版本上合法”及 context-policy 中“不能为迁移方便改写历史路径”的边界。

## 二、资料入口（均实际核对存在）

- **平台总入口**：`platforms/README_cn.md` / `platforms/README.md`（第 13 行表格行给出两个链接）；仓库根目录**没有**根 README，平台说明即此文件。
- **注册表**：`platforms/registry.json`（x3 条目，含清单/发布说明/README 路径）。
- **原平台资料**：`platforms/x3/README.md`、`platforms/x3/README_cn.md`（硬件背景、模型列表、FAQ、社区入口；注意其内部链接指向 `demos/` 旧布局）。
- **历史清单**：`platforms/x3/release/`（`models.yaml`、`benchmarks.yaml`、`schemas/`、`README.md`）——按旧布局放在平台树内，而非 `docs/release/`。
- **发布说明**：`platforms/x3/docs/releases/x3-v1.0.0.md` 至 `x3-v1.1.2.md` 共 4 份；另有 `platforms/x3/CHANGELOG.md`。
- **历史 demo 与资源**：`platforms/x3/demos/`（分类/检测/实例分割等，约 195 个文件，含 ipynb）、`platforms/x3/resource/`；**无** `platforms/x3/samples/`。
- **X3 相关决策记录**：`docs/adr/0007-evaluate-native-x3-hbm-api-subset.md`。

## 三、本轮是否应新增 X3 后端：**不应**

1. **范围文明确排除**：`platforms/README_cn.md:5`“不属于本轮新增适配范围”；registry description 亦写明 X3 仅归档。
2. **迁移台账不接纳**：`x5-s-migration-map.md:130` 明确 X3 历史资源“不能在本表中改动或算入 X5/S 迁移完成度”。
3. **ADR 0007 只是评估项，不是后端批准**（`docs/adr/0007-evaluate-native-x3-hbm-api-subset.md`）：状态为 “accepted-for-evaluation”——仅用户认可的**有限可行性验证**，“并非已批准完整 SDK 复刻或认定技术可行”；若语义或维护成本不满足，回退为“pyeasy_dnn 薄适配”；首版明确不含线程安全/异步/零拷贝/完整调度，“不覆盖系统 hbm_runtime 模块”，且“X3 仍使用适合自身架构的资产”。
4. **若未来真要 X3 支持**：按版本线规则属 `rdk_x3` / `x3-v*` 交付线（`platforms/x5/docs/RELEASE.md:13`），需在该线自身规范与实机证据下立项，而不是在 X5/S 统一迁移里顺手加；也不得为此改写 `platforms/x3/` 的历史布局。

## 四、权限边界与局限（如实说明）

- **未执行**技能第 2 步的 `scripts/inspect_repo.py`：本环境未暴露 shell 工具，改用 Glob/Grep/Read 做等效只读盘点。
- **仓库身份未核验**：无法运行 `git rev-parse`/`git status`，HEAD、branch/Tag、dirty 状态记为“未知/未读取”。目录名 `rdk-b7-board-integration` 及文档自述“在 `develop` 进行、尚未完成客户发布验收”（`platforms/README_cn.md:5`）仅为文本线索，非 git 观测。
- 按任务要求**未读取 fixture.json**（任务未引用它）；未逐个打开 x3 全部 195 个文件，结构经 glob 与定点抽读确认。
- 本轮（AG08）无文件编辑授权；实际也未做任何写入、分支或清理操作。未生成任何模板性“已验证/已上板”结论——本盘点为纯静态只读分析。
