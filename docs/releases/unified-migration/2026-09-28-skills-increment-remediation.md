# H8 — X5 上游 Skills 增量整合（作者报告）

Author: Claude Code + GLM（用户指定的实现分工）。Reviewer: Codex（独立评审 pending）。
工作树 `codex/b7-board-integration-20260924`，起始 HEAD `ecaa8ca8`。上游增量
`e3f9fa3fb5a795b2531bdb84fa60d03768af5956..d1b24f65b7307e38e747fe39829e794c455a6a22`
（origin/rdk_x5，单提交 `d1b24f65` "fix(skills): align Model Zoo routes with current
OE workspaces (#175)"，31 文件，+104/−53），按评审报告
[2026-09-28-x5-skills-increment-review.md](2026-09-28-x5-skills-increment-review.md)
的 H8 约束执行。**H8 保持 open**；本报告只声明本包完成的事项，不关闭该任务，
也不声明任何 Agent 行为验收。

范围：仅 `skills/`、本报告、以及
`evidence/2026-09-28-skills-increment-remediation/`。根 README、总计划、台账、
其他 sample、manifest、reviewer 报告均未改动；并行 MiniCPM/PointNet 工作未触碰。
未 fetch/checkout/merge 上游分支，全部合并决策来自 read-only `git show/diff`。

## 1. 上游增量逐项映射

| 上游改动（d1b24f65） | 本分支处置 |
|---|---|
| `_shared/context-policy.md`：`.drobotics/`、`.horizon/` → `.drobotics-x5/`、`.drobotics-s/` | 采纳（共享编辑源；随后用 `tools/sync_references.py --apply` 再生成 9 个登记副本：7×context-policy、2×toolchain-handoff） |
| `_shared/toolchain-handoff.md`：S 路由 `S/horizon router` → S 系列 `drobotics-router` | 采纳 |
| `_shared/toolchain-handoff.md`：S 部署范围来源链接 → `D-Robotics/oe-skills-s@v1.1.0/drobotics-s/skills/drobotics-router/...` | 采纳；HTTP HEAD 探测返回 200（证据 green-checks log） |
| 7 个成员的 `references/context-policy.md`、2 个 `references/toolchain-handoff.md` 副本 | 不手工改；由 sync 工具再生成，与上游 blob 逐字节一致（11/11 文件 `git diff d1b24f65` 为空） |
| `rdk-model-zoo/SKILL.md`：S router 交接行 `horizon-router` → `drobotics-router` | 采纳 |
| `rdk-model-zoo-release/references/release-contract.md`：工作区路径两处 | 采纳 |
| 同文件："Skills Pack 候选 1.0.0…其他新 Skill 为 1.0.0" 版本句 | 适配为本分支真实版本（候选 1.1.0；入口及 repo/develop/validate/review 1.1.1，integrate/release 1.0.1；并注明上游 1.0.1 已发布、本候选未发布）。直接照抄上游句子会写谎本分支成员版本 |
| 同文件："Hub 已注册 components.d/rdk-model-zoo.yml；新版本只更新 ref、镜像和生成物" | 采纳上游事实修正（Hub 注册状态是上游已完成的事实，与分支无关）；"来源"小节的历史 pin 链接一律未动 |
| `tests/test_pack.py` 两个新回归测试 | 适配采纳（见 §4） |
| pack/成员版本、`release_state: released`、README/CHANGELOG/卡片发布声明 | **不采纳发布声明**；版本按 §3 决策适配（这是评审明令防止的降级/伪发布风险点） |

未从上游带入的内容：无 —— 上游 diff 中其余全部为上述版本/状态文件的对应改动，
已逐项处置。上游 diff 全文存于
`evidence/.../upstream-increment.diff`；本包工作树 diff 存于
`evidence/.../skills-working-tree.diff`。

## 2. Q5（0a1ba609）保留

Q5 的全部行为约束原样保留，零回退：develop（README/inference-contract 顶层条款、
file↔contract↔evidence 映射）、review（四项强制检查 + 报告模板）、validate（代码块
四分类、host 执行绑定 SHA/cwd/工件身份）、repo（inspect_repo.py 的
platform_manifests/unified_layout/integration-role 事实）、8 条新增 eval 定义、
`tests/test_tools.py`、assets。`git diff e3f9fa3..d1b24f65` 与工作树的交集仅发生在
上游也改动的版本/共享文件上；Q5 独有文件不在本包 diff 内。维护源字段
`source_repository`/`maintenance_branch: rdk_x5` 保持"仅来源"语义；平台判断规则
（`rdk_x5` 不触发 X5 工具链、不默认 X5）在上游修正后原文保留。

## 3. 版本/状态决策

| 对象 | 原值 | 新值 | 依据 |
|---|---|---|---|
| Pack（pack.json `version`、`skills/VERSION`） | 1.0.0 candidate | **1.1.0 candidate** | 树内现含 1.1.x 成员（Q5 + 本次补丁）。取 1.0.1 会与上游已发布的 1.0.1 同号不同物，制造假等价；候选跳次版本号表达成员能力级（Q5 新增门禁/eval），semver 允许 |
| `release_state` | unreleased-candidate | **unreleased-candidate（不变）** | 上游 `released` 描述上游 Pack；本分支适配候选未发布、未打 tag。不伪称发布，不新建 tag |
| rdk-model-zoo | 1.1.0 | **1.1.1** | 与上游同基（1.1.0）同补丁（router 交接行），对齐上游 1.1.1 |
| repo/develop/validate/review | 1.1.0 | **1.1.1** | Q5 功能级保留；不降级到上游的 1.0.1。补丁号吸收本次共享内容变更，使成员版本可见地推进 |
| integrate/release | 1.0.0 | **1.0.1** | 无 Q5 增量，与上游映射一致 |

一致性修复（本包发现并一并处理）：Q5 曾把 repo/develop/validate/review 的
pack.json/SKILL.md 升到 1.1.0，但其 skill-card 停在 1.0.0 —— 既有的元数据漂移。
本次七张卡片统一为真实 Skill 版本 + `| Pack 候选版本 | 1.1.0（未发布） |`（保留
"候选/未发布"措辞以区别于上游卡片的 "Pack 版本"）。`skills/README.md` 交付状态段、
CHANGELOG 标题与新增条目、eval 定义计数（75→83：原始 70 + 跨平台 5 + Q5 8，
与 validator `eval_cases: 83` 一致；`evals/README.md` 中"已执行 75 条"是历史执行
事实，不改写）。历史 provenance：`evals/codex-evidence-2026-09-17.zip` 的删除是
既有分支适配（HEAD 已然），本包未追加删除；`docs/releases/unified-migration/
2026-09-21-phase05-q5-skills.md` 中"pack 保持 candidate 1.0.0"是冻结的历史记录，
按规不改。

## 4. 测试（host，Python 3.14.7 venv，cwd=仓库根）

先加测试后整合，红→绿：

1. **Red**（整合前，仅新增 2 测试应失败）：`Ran 57 tests, FAILED (failures=2), rc=1`
   —— 失败恰为 `test_platform_workspace_and_router_handoffs_match_hub` 与
   `test_candidate_pack_and_member_versions_are_consistent`。证据
   `red-new-tests-before-integration.log`。
2. **Green**（整合后）：
   - `sync_references.py --pack-root skills`（只读）：`valid: true, changed_files: []`，rc=0
   - `validate_pack.py --pack-root skills`：`valid: true, skills_checked: 7, eval_cases: 83, behavior_evaluated: false`，rc=0
   - `validate_pack.py --skill-root skills/rdk-model-zoo`（平铺资源闭包抽查）：`valid: true`，rc=0
   - `python -m unittest discover -s skills/tests -v`：`Ran 57 tests … OK`，rc=0（既有 55 + 新增 2）
   - 证据 `green-checks-after-integration.log`、`baseline-checks.txt`（整合前 55 全绿基线）。

上游两个回归测试的适配说明：路由/目录测试按上游原样采纳，另加两条本分支断言
（S 新链接 pin `oe-skills-s@v1.1.0` 存在、旧 `rdk-skills`/oe-skills-s 路径消失）；
版本测试改名为 `test_candidate_pack_and_member_versions_are_consistent`，断言
候选 1.1.0、`unreleased-candidate`、七个成员精确版本、README/CHANGELOG 的候选措辞
（含 `assertNotIn('Pack 1.1.0 已发布')`）、frontmatter↔pack.json↔卡片三方一致。
两测试仍属结构/元数据检查（文件头注释保留"not Agent behavior tests"声明），
不把作者自检当 Agent 行为验收。

## 5. 未验证 / 边界（交 Codex 评审）

- **未做**：Agent 行为矩阵、真实路由/安装交接、板卡、量化、导出、工具链安装、
  Hub 操作 —— 按包界定全部留给行为验收阶段。
- `~/.claude/skills/` 已安装副本**未**重新同步（本次明令禁止安装）：源与安装副本
  现存在 workspace/router 路径漂移，需另行授权的同步步骤。
- Python 3.14.7 上验证；`skills/README.md` 的"Python 3.13.5 实测"是原始交付的
  历史陈述，未改写（如需更新请评审定夺）。
- 上游 S 链接仅做了 HTTP 200 可达性探测，未校验目标文档内容语义。
- release-contract 的 Hub 注册状态句采纳自上游维护源；如上游事实有误应在此回退。
- 并行 MiniCPM 包正在修复 `docs/release/s/models.yaml`；skills 套件全部使用合成
  fixture，不读 live manifest，基线与绿跑均不受其影响（55/57 两次全量运行一致）。
- 证据 hash：`changed-files.sha256`（before 见 `baseline.txt`）。
