# Phase 0.5 — Q5 Skills 补强（2026-09-21）

Q5 的任务是不新增第八个入口，对既有 develop/review/validate 三个技能补细化约束与检查
证据，扩展 repo 技能的统一布局事实发现，并把修订同步到执行端已安装副本。仓库规范保持
唯一权威；Skills 读取目标 ref 的规范，不把 X5 维护源规则无条件施加到旧 ref。

## ① develop：README 契约关联与检查表强化

`skills/rdk-model-zoo-develop/SKILL.md`（1.0.0→1.1.0）：

- 步骤 2 增补：目标 ref 存在样例标准契约时（如 develop 的
  `docs/sample-standards/readme-contract.md` 与 `inference-contract.md`）一并读取并列为
  最高层适用条款；旧 ref 没有这些文件时不虚构，按该 ref 实际规范执行。
- 步骤 7 增补：**改动前**先按目标 ref 的 README 内容契约列出本次将触及的文件与各级
  README 章节（含父级索引），明确每个文件的职责；**改动后**在检查表中逐条关联
  "文件 ↔ 契约章节 ↔ 检查证据"，不允许以一行"文档已更新"代替逐项关联。

`assets/development-checklist.md` 同步新增行：改动前列 README 契约触及面与文件职责；
推理三阶段职责边界（pre/forward/post/predict，forward 无下载/NMS/解码/绘图/文件输出）；
改动后逐条文件↔检查证据关联；enriched 模型获取行要求"CLI 默认值与 README 表格逐项核对"。

## ② review：四个必查项与报告模板

`skills/rdk-model-zoo-review/SKILL.md`（1.0.0→1.1.0）步骤 5 增补四个必查项：

- （a）**README 可操作性**——按目标 ref 的 README 契约逐章核对必答问题；标题齐全但缺
  输入/执行目录/结果解释、命令与代码默认值漂移均为 finding，不得因章节标题齐全静态
  pass；
- （b）**接口职责**——forward/infer 混入下载、NMS、任务解码、绘图或文件输出按推理契约
  定为越界 finding（契约在目标 ref 不存在时按其接口规范判定）；
- （c）**旧能力保留**——相对 base 的能力、变体、语言或平台支持消失须列明并要求需求
  来源；
- （d）**数值回归**——预处理数值、raw output、任务结果的对照证据或明确 not-run，不接
  受"应无影响"。

`assets/review-report.md`：Delivery Specification 增加 README 可操作性走查 block；
Technical Correctness 必含接口职责越界检查、旧能力保留核对、数值回归证据状态。

## ③ validate：README 命令分类与证据绑定

`skills/rdk-model-zoo-validate/SKILL.md`（1.0.0→1.1.0）步骤 2 增补：验证 README 命令
与 API 示例时，先把文档代码块按 **说明 / 主机 / 板端 / 转换** 分类——主机类在授权内实际
执行并绑定代码 SHA、cwd 与制品/输入身份；板端与转换类只形成待执行计划交对应执行者；
纯说明文字不冒充已验证命令；**结构校验通过不是命令验证**。

## ④ 行为评测定义新增（8 case，仅定义未执行）

按协议 tasks.yaml 记录"要测试的行为"，不是执行结果；本步不声称任何行为通过：

| 技能 | 新增 ID | 行为场景 |
| --- | --- | --- |
| develop | develop-eval-011 | CLI 与文档漂移：应指出漂移、以代码事实修正、逐项关联证据 |
| develop | develop-eval-012 | forward 混入下载/NMS：应拒绝越界、按契约拆分、predict 层封装 |
| review | review-eval-011 | 标题齐全但不能运行：按契约逐章核对、出 finding、不静态 pass |
| review | review-eval-012 | forward 违规 PR：越界 finding、不以功能可用豁免契约 |
| review | review-eval-013 | OCR 合法多阶段：认定合法、核对阶段边界、不要求合并阶段 |
| review | review-eval-014 | 作者声称板端通过：不采纳、not-run、不变 ready |
| validate | validate-eval-011 | README 命令分类验证 |
| validate | validate-eval-012 | 不可绑定历史板测声称：拒绝 board passed |

case 数 12/14/12（develop/review/validate），全 pack 83；ID 唯一、YAML 可解析
（validate_pack 计数与 unittest 一致）。断言检查"含义而非关键词"。

## ⑤ inspect_repo.py 统一布局事实发现（repo 1.0.0→1.1.0）

`skills/rdk-model-zoo-repo/scripts/inspect_repo.py`：

- 新增 per-platform 清单模式：`docs/release/{platform}`（unified）、
  `platforms/{platform}/docs/release` 与 `platforms/{platform}/release`（snapshot）；
  同平台两者并存时 unified 胜出（snapshot 是迁移期冻结副本）。
- 输出新增 `platform_manifests`（platform→manifest 路径）与 `unified_layout`
  （`samples/` 存在且有 `docs/release/` 前缀的 per-platform 清单）。
- `branch_role: integration` 仅在分支名 **与** 布局事实同时成立时报告——develop 分支名
  单独不证明任何事（fixture 测试固定该行为）；分支名与硬件推断继续分离
  （`platform_hint` 对 develop 为 null）。
- warnings：unified 布局注记（硬件按 sample 选择，绝不从分支或清单存在性推断）；flat
  单平台清单与 per-platform 并存时报 "Migration dual state"；flat 歧义警告仅在
  `len(flat)>1` 时触发（两个 per-platform 清单是统一架构常态，不是歧义）。

新增 5 个合成 fixture 测试（`skills/tests/test_tools.py`，共 55）：统一布局上报双平台
清单且不误报歧义；仅 snapshot 不声称 unified；同平台 unified 胜出；integration 角色需
事实；dual state 旗标。历史 ref 行为（maintenance/target/unknown、detached、脱敏、
traversal/symlink 拒绝）全部保持。

## ⑥ 版本与 changelog

- 4 个被触及技能 member 版本 1.0.0→1.1.0（SKILL.md frontmatter + `pack.json` 成员行）；
  pack 版本保持 candidate 1.0.0、`skills/VERSION` 不动（成员与 pack 版本分别管理，
  validate_pack 双向校验通过）。
- `skills/CHANGELOG.md` Unreleased 段新增 Q5 六条 bullet（含"行为评测是定义、非执行
  结果"的明示）。
- `_shared/` 未动 → `sync_references.py` 校验无漂移（changed_files=[]）。

## ⑦ 已安装副本更新（源码与安装端分开记录）

- 更新前核对：4 个技能的安装副本与源码差异**恰好**为 Q5 触及文件集（develop 3 文件、
  review 3、validate 2、repo 2），其余 3 个未触及技能逐字节 SAME——确认无用户改动后
  才执行 rsync 镜像更新（`--delete --exclude=__pycache__`）。
- 更新后内容一致：7 个技能目录 `diff -r` 全部 SAME；4 个技能 frontmatter 均为 1.1.0。
- 发现一致（repo 技能实测）：更新前安装副本对 develop 返回
  `branch_role=unknown`、`manifest_candidates=[]`、无 `unified_layout` 键（Phase 0 已知
  边界）；更新后返回 `branch_role=integration`、`unified_layout=true`、
  `platform_manifests={x5,s→docs/release/*, x3→platforms/x3/release/*}`，与源码脚本
  输出逐字节一致。
- 版本记录（分开）：源码 = 本 commit 后的 `skills/`（member 1.1.0 ×4、1.1.0 ×1、
  1.0.0 ×2）；已安装 = `~/.claude/skills/` 同内容（安装端无 pack 级文件，flat install）。

## 验证

- 三件套：`sync_references.py`（bare check）`valid:true, changed_files:[]`；
  `validate_pack.py` `valid:true, skills_checked:7, eval_cases:83,
  behavior_evaluated:false, board_verified:false`；`unittest discover -s skills/tests`
  55/55 OK。
- 行为评测套件本身未执行（`behavior_evaluated:false` 如实输出）——8 个新 case 是行为
  定义；Codex 75 案例结果为 2026-09-17 历史记录，不因本步改写。
- 源码与安装端的发现一致性以实际脚本输出比对（JSON 逐字节相等），非文件存在性冒充。

## 未运行项

- 行为评测执行（定义已入库；执行需独立会话/执行者，本步不冒充）。
- 板端（无涉及）；Hub/发布动作（不在计划内）。

## 结论

Q5 全部子项完成：三个技能补强、8 个行为评测定义、repo 统一布局事实发现、版本/changelog
规范推进、安装端副本更新并重验。**Phase 0.5（Q1–Q5）至此全部完成**：Q1/Q2 契约与检查
器正反例（2026-09-20 记录）、Q3 检查器（同日）、Q4 双参照验收（2026-09-21）、Q5（本
记录）。下一步：Phase 1.5 架构硬化 H1–H4（H5/H6 随 B1），之后进入 B1 批次迁移。
