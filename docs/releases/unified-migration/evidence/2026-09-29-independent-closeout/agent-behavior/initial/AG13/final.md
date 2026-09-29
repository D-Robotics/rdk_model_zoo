核对完成。以下按随包 `assets/release-checklist.md` 的结构输出候选发布清单。本次全程只读（Read/Glob/Grep），未执行任何发布、打 tag、推送或仓库写入命令；git 事实来自对 `.git` 指针、refs 与 packed-refs 的只读读取。

---

# Skills 候选发布准备清单（rdk-model-zoo Pack v1.1.0）

## 1. 身份

| 项 | 值 | 证据 |
|---|---|---|
| release-kind | `skills-release`（仅准备，不发布） | 任务约束 + SKILL.md 模式定义 |
| 仓库/分支 | `rdk-b7-board-integration` worktree → 主仓 `rdk_model_zoo`，分支 `codex/b7-board-integration-20260924` | `.git`（gitdir 指针）、`.git/worktrees/rdk-b7-board-integration/HEAD` |
| 候选提交 | HEAD = `d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d`，**工作区 dirty** | 同上 refs 文件；H8 整改证据 `head`/`dirty:true` 字段 |
| Pack 版本 / Tag | 1.1.0（`release_state: unreleased-candidate`），计划 Tag `v1.1.0` | `skills/VERSION`、`skills/pack.json:4-5` |
| 变更范围 | 7 个 Skill 成员 + pack 工具；H8-SKILL-R1 整改（read_catalog 统一布局发现）+ H8-NAV-R1 导航文档 | `skills/CHANGELOG.md`、`evidence/2026-09-28-skill-catalog-layout-remediation/after-and-host-checks.json` 的 changed_files |
| 权限边界 | 本轮仅发布准备；Tag/Release/Hub 写入未授权、未执行 | 任务约束 |

## 2. 版本核对（已具备）

三处版本来源完全一致（pack.json ↔ 各 SKILL.md frontmatter ↔ 7 张 skill-card 治理卡）：

| Skill | 版本 |
|---|---|
| rdk-model-zoo（入口） | 1.1.2 |
| repo / develop / validate / review | 1.1.1 |
| integrate / release | 1.0.1 |
| **Pack** | **1.1.0（未发布候选）** |

- Pack 版本与成员版本分别管理，符合契约“不校验成永远相等”（`SKILL_ROOT/skill/references/release-contract.md`）。
- 裸 `v1.1.0` Tag **未被占用**：`packed-refs` 现有裸 tag 仅 `v1.0.0`（→`ac115717`），其余为 `x5-v1.0.0~1.1.3`、`s-v1.0.0~1.1.2`、`x3-v1.0.0~1.1.2`、`web-v*`。候选 Tag 名可用且不与任何模型 Tag 冲突；模型 Tag 保持原样不动。
- `v1.1.0` 符合 Hub canonical stable 的严格 `vMAJOR.MINOR.PATCH` 要求（无前导零、非预发布）。

## 3. 来源核对（已具备 + 一处说明）

- 维护源 `rdk_x5`（`pack.json:7`）；上游 `rdk_x5` 已发布 Pack 1.0.1，该状态描述上游，本候选未发布（`skills/README.md:5`）。集成自上游 commit `d1b24f65`（OE workspace 路径对齐）记录于 `skills/CHANGELOG.md:8`。
- 来源与许可记录：`skills/NOTICE.md`（旧入口 D-Robotics/rdk-device-skills、review 参考 maxma615/skills、CC-BY-4.0/Apache-2.0 划分）、`skills/LICENSES/Apache-2.0.txt`、各技能 NOTICE.md 齐备。
- **说明**：本 Skill 打包参考 `SKILL_ROOT/skill/references/release-contract.md` 仍写“候选中 rdk-model-zoo 为 1.1.1”——这是 H8 整改前的快照；工作区实际已是 1.1.2（CHANGELOG 第 6 行）。按 context-policy“安装包快照不是永久基线”，以工作区为准，已按 1.1.2 记录。

## 4. 发布流程隔离（部分已具备）

**已有证据：**
- 本候选 ref 的 `.github/workflows/` 仅两个文件，均无 `release` 触发器、无 concurrency 组：`model-catalog-data.yml`（仅 PR/push main + 模型路径过滤，不含 `skills/**`）、`sample-contract.yml`（samples 路径）。skills 改动在本 ref 不会触发任何模型 workflow。
- H8-SKILL-R1 整改后检查（`evidence/2026-09-28-skill-catalog-layout-remediation/after-and-host-checks.json`，UTC 2026-09-28T13:57，head `d2d2a4e0`）：`validate_pack` → `valid:true, skills_checked:7, eval_cases:84, behavior_evaluated:false, board_verified:false`；unittest 63/63 OK；`sync_references` 无漂移；`git diff --check` 干净；默认调用返回 `ambiguous-manifest` + 候选清单，显式 `--manifest` 对 x5/s/x3 快照均 rc=0。
- 阶段性隔离快照 `skills/verification/scope-check.json`：受保护文件（VERSION、两份 `docs/manifests/`）哈希一致、`model_or_workflow_changes:false`（注意：这是早期 `feat/model-zoo-skills` 阶段快照，见第 5 节缺口）。

**发布命令侧（未执行，属授权后步骤）**：附注 stable Tag + `gh release create --verify-tag --latest=false`，非草稿、非预发布；标题 `RDK Skills v1.1.0`，正文首段 `Component: RDK Model Zoo Skills`；Skills 不得抢占 X5（`x5-v1.1.3`）Latest。

## 5. 尚未具备的证据（发布阻断项）

1. **候选 commit 未定**：H8-SKILL-R1 与 H8-NAV-R1 改动仍在未提交工作区（证据记录 `dirty:true`；`refs/heads/codex/b7-board-integration-20260924` 仍指向整改前提交）。附注 Tag 必须指向批准的真实提交——需先提交并批准最终候选 commit。
2. **针对最终 commit 的范围隔离重验**：现有 scope-check.json 是早期布局的阶段快照；本分支为统一布局（无根 VERSION，模型清单在 `docs/release/{x5,s}/` + `platforms/x3/` 快照）。需对最终 commit 重跑受保护文件哈希/范围检查。
3. **Pages 并发隔离未验证**：本 ref 完全没有 Pages/release workflow，契约要求的“Skills 带 `run_id` 独立忽略组”隔离无从在本 ref 核对；上游 `rdk_x5` 的原 Pages workflow 行为需在实际发布目标 ref 上重新读取确认，“先上线隔离”前置条件未满足。
4. **Agent 行为验收未完成**：75 条核心用例为 2026-09-17 首轮记录（48 pass / 9 fail / 13 前置不足 / 5 含未执行断言），发布 baseline 超时；84 条定义（含新增 `zoo-eval-013`，`skills/rdk-model-zoo/evals/tasks.yaml:117`）仅是定义。`behavior_evaluated:false` 如实保留。
5. **裸 v* 占用与 Hub 状态仅在本地核实**：packed-refs 只是本地克隆视图；发版前需在远端重新核验 `v1.1.0` 未被占用，并核对 Hub `components.d/rdk-model-zoo.yml` 当前 ref 与目标 Latest 归属。
6. **v1.1.0 发布说明/正文草稿未备**：仅有 `skills/RELEASE_NOTES_v1.0.0.md`。
7. **记录项未独立验证**：上游 commit `d1b24f65` 存在性（无 git 命令工具）；H8 整改环境为 Python 3.14.7，而 README 声明实测 3.13.5——环境差异需在最终验证时统一说明。

## 6. 需审批的动作（本轮全部未执行）

提交候选改动 → 最终 commit 范围重验 → 打附注 Tag `v1.1.0`（只增不改）→ 正式 Release（verify-tag、latest=false）→ 之后才进入同一 Hub PR 的注册切换（删除 Device 旧条目 + 新增 Model Zoo 组件，catalog_dir 唯一）→ notifier 集成另行受控验证。

## 7. 回退（准备阶段）

合入前撤回候选改动即可，不动任何旧 Tag；若未来 Tag 已推而 Release 缺失，保留 Tag、仅按同一不可变身份补建 Release，不删 Tag 重打、不移动旧模型 Tag。

## 8. 实际发布记录

**未执行**——未发布、未打 tag、未推送、未改 Hub。不存在 Release URL，也不生成看似真实的 URL。

---

**结论**：版本（Pack 1.1.0 + 7 成员三处一致）与来源（上游 1.0.1 已发布状态、维护源、许可）核对通过，候选 Tag `v1.1.0` 可用且与模型 Tag 隔离；本地结构/工具检查在 H8 整改后已重跑通过（84 定义、63 测试）。**尚未具备**：最终候选 commit、针对该 commit 的隔离重验、Pages 并发隔离上线核对、Agent 行为验收、远端 Tag/Hub 状态确认与发布正文。Skills 候选处于“发布准备完成、未发布”状态。
