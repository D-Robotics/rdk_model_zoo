# Agent 入口文档修正（ENTRY-R1 作者记录）

Author: Claude Code + GLM。Reviewer: Codex（独立评审 pending，**ENTRY-R1 不由本包关闭**）。
依据 [2026-09-28-agent-entry-independent-review.md](2026-09-28-agent-entry-independent-review.md)
的 ENTRY-R1。起始 HEAD `ecaa8ca8`，分支 `codex/b7-board-integration-20260924`。

**改动范围**：仅根 `CLAUDE.md`（working tree，before `c153ba81…` → after `c550b7b2…`）、
本记录、`evidence/2026-09-28-agent-entry-remediation/`。AGENTS.md、spec、ADR、
sample-standards、samples、skills、计划台账、旧 reviewer 记录均未改动；并行
Gemma/MiniCPM 包未触碰。纯文档修正，无产品运行时改动，无量化/下载/板测。

## 逐条处置（评审发现 → 修正）

| ENTRY-R1 发现 | 修正 |
|---|---|
| "skills 不在本树、验证要去找 rdk_x5/installed copy"（原 15–19 行） | 删除整个 "NOT on develop yet" 中的 skills 条目；`skills/` 按本树现状描述（7 技能候选包、`_shared/` 单一编辑源、sync 漂移检查、maintainer-only 目录、运行期脚本不联网），并明确"在本树存在且可跑主机检查**不是发布或独立验收声明**"；未安装授权不得装 skill 或改 `~/.claude/` 副本 |
| "platforms manifests 当作 active、等未来 A4"（原 53 行） | 改为：active = `docs/release/{x5,s}/models.yaml` + `docs/release/platforms.json`；`platforms/{x5,s,x3}/docs/release/*.yaml` 是交付分支的**冻结归档快照**（`platforms/README.md`/`registry.json`），只用于历史事实、文件名键可能不同；`docs/manifests/` 是 `rdk_x5` 布局 |
| "每个 sample 必须 Python、固定 dict/tuple 接口、utils/py_utils"（原 29–42 行） | 删除整段旧 Python/C++ 处方；改为指向 `docs/sample-standards/inference-contract.md`（四方法公开接口、显式 per-call context、校验后 RawOutputs、owned typed Result、阶段职责、binding/runner/task 拆分、多阶段/流式豁免）及两个参照实现（resnet `classification.py`、paddle_ocr `pipeline.py`）；C++ 按 sample runtime README 声明的结构，不逐方法镜像 Python；共享组件在 `samples/_shared/`，`utils/py_utils` 仅余兼容位置；"runtime/python (always)" 删除，语言覆盖按 README 支持矩阵声明（`samples/llm/gemma4-e2b`、`samples/llm/minicpm5-2b` 为 C++/legacy 反例，路径已核实） |
| "S/X3 release before X5"、x3-v* 当活跃约定（原 63 行） | X3 从活跃发版叙述移除（仅 ADR-0007 `accepted-for-evaluation` 的有界评估）；发版指向 ADR-0006 统一源码发版模型；交付线约定（`x5-v*`/`s-v*`、各 ref `VERSION`、CHANGELOG 唯一发布说明位置）标注为**交付 ref 上的约定、本分支的历史参考**（已核实 `rdk_x5:VERSION`=1.1.3、`rdk_s:VERSION`=1.1.2 仍在各自 ref 上）；本工作分支不作任何发布声明 |
| 泛化 spec 优先声明代替可操作指引 | 新增 "Authoritative documents" 段：AGENTS.md、活跃 spec（`2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md`，迁移期冲突裁决）、规范基线（`docs/Model_Zoo_Repository_Guidelines.md` develop 统一架构版——原 CLAUDE.md 称其不在本树，已过时）、两契约+模板、ADR 索引、迁移地图。指向权威文档而非复制会漂移的长处方 |
| 工作分支 vs 交付分支不分 | 新增 "Where you are" 段：本检出是**迁移整合工作分支，未合入 develop、未发布**；develop 为整合线（`--target` 四态、板卡身份解析、无静默回退）；rdk_x5/rdk_s 为客户交付线（历史参考）；rdk_x3 仅归档；人/Agent 同一原生命令（ADR-0003）；AGENTS.md 的禁切分支/禁 reset 规则保留 |

**保留并核实的有用内容**：AGENTS.md 全部主机命令照实保留（`samples/_shared/tests`、
resnet、ultralytics_yolo、paddle_ocr、VLA 集成测试 `-p test_vla_integration.py`），并
补上实际存在的 `tools/sample_contract/check.py`（`--sample`/`--scope migration`）与其
自带测试、skills 三件套（`sync_references.py`/`validate_pack.py`/`skills/tests`——本会
话早前已实跑通过）、YOLO 目录对比所需的 `npm --prefix tools/catalog-publisher run
build`。真正仍不在本树的条目如实保留为"rdk_x5 专属"：`docs/catalog/`、
`docs/RELEASE.md`（均验证 absent）。双语、README 层级、manifest 同 commit 更新、
`sha256: null` 纪律、benchmark 不可推断、目录数据边界、已发布 tag 不可移动等约束
全部保留。原"量化配方可信来源、不重跑导出/校准/OE 验证"的用户范围声明由 AGENTS.md
承载，CLAUDE.md 通过指向 AGENTS.md 不再复述。

## 验证（静态，按评审 acceptance 执行）

1. **路径静态解析**：44 个正向路径全部存在、4 个负向声明（`docs/catalog`、
   `docs/RELEASE.md`、`docs/manifests`、根 `VERSION`）全部 absent —— 完整清单
   `evidence/path-verification.txt`。
2. **内容级核对**：`.gitignore` 确有 `*.pt`/`*.onnx`/`*.bin`/`*.hbm`；`samples/_shared/
   platforms.py` 在 82/86/94/97 行对未知 target/不可识别板卡/目标不匹配 `raise`（无静默
   回退）；resnet `model/download.sh` 委托 `download.py --target/--variant`（清单驱动），
   `archive.d-robotics.cc` 见于 model 层 README；CMake SoC 检测存在于 resnet
   `runtime/cpp/CMakeLists.txt`；inference-contract §1–4 章节锚点与所引条款一致。
3. **无新测试**：按任务约定"静态核对路径与规范足够"，未为文档改动编写测试；本包不改
   代码，既有主机套件不受影响（本会话早前 skills 三件套 57 全绿为最近一次实跑记录，
   不在本包重复声明为验收）。
4. **证据**：`path-verification.txt`、`before-after.sha256`、`claudemd.diff`（工作树 diff
   快照，亦可用 `git diff -- CLAUDE.md` 复核）。

## 未做 / 边界

- ENTRY-R1 保持 open，待 Codex 独立评审本 diff；本包不自行关闭。
- skills 增量整合候选（H8 路径）仍待评审：CLAUDE.md 中仅客观描述 skills/ 结构与主机
  检查，未写发布、未写独立通过、未写 H8 完成。
- 未访问/修改 `~/.claude/` 自动 memory 或已安装 skills；未跑板测/量化/下载/依赖安装；
  未用子 agent；未 commit/push/merge/reset/stash/checkout。
- 交付 ref 事实（`rdk_x5:VERSION` 等）经只读 `git show <ref>:VERSION` 核对，未切换分支。
