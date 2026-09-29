## 结论：读哪一侧

按技能自身的规则（`SKILL.md` 第 29/35 行、`references/context-policy.md` §“维护源与目标引用”），**事实与规范一律读目标侧 `REPO_ROOT`，安装侧 `SKILL_ROOT` 只提供方法（脚本、策略、模板）**。本次判定：

| 侧 | 路径 | 角色 |
|---|---|---|
| 安装侧（仅方法，非目标） | `/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/.coordination/20260928-skills-behavior/AG15/skill/` | 从 `rdk_x5` 维护源打包的文档快照；其 context-policy §“来源”自注链接为“维护源参考”。`rdk_x5` 只是维护源线索，“不把任务目标默认为 X5” |
| 目标侧（数据/规范事实来源） | `/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration/` | 统一 Model Zoo 工作区；S600 的清单、README、注册表、样例支持范围全部以此处为准 |

## 按实际仓库辨认出的 S600 入口

1. **平台身份（S600 不是分支，是 S 发布组内的硬件目标）**
   - `platforms/registry.json:22-35`：platform `id: "s"`，`hardware: ["s100","s100p","s600"]`，`manifest_directory: "docs/release/s"`。
   - `docs/release/platforms.json:8`：`{"id":"s600","release_group":"s","soc_names":["s600"]}`；板卡身份只认板卡事实（socinfo/device-tree），未知即报错。
2. **清单入口（该仓库实际布局是 `docs/release/s`，不是安装源所说的 `docs/manifests/`）**
   - `docs/release/s/models.yaml`、`docs/release/s/benchmarks.yaml`、`docs/release/s/VERSION`（已确认存在；models.yaml 含 `filename: s600/...` 资产条目，如 `s600/asr.hbm`、`s600/yolov5x_672x672_nv12.hbm`；`sha256: null` 是未知，不是验证通过）。
3. **统一样例入口（硬件按 sample 选择）**
   - 根 `samples/`（根 `README.md:11` 称 51 个统一 sample）。例如 `samples/vision/ultralytics_yolo/README.md:17-26` 有 x5/s100/s100p/s600 逐 target 支持矩阵（如 YOLOv9 无 S600 检测/分割），`:67` 给出 `--platform s600 --dry-run` 用法；`samples/_shared/README.md:24` 资产键形如 `s:ultralytics_yolo:nash-p/...`（S600 对应 nash-p，见根 `README.md:45`）。
   - LLM/VLA 侧：`samples/llm/minicpm5-2b`（S600 OELLM 2.0 beta 入口）、`samples/vla/`（S600 ACT/Pi0），见根 `README.md:20,36`。
4. **规范入口（目标 ref 自己的规范，替代安装源参考）**
   - `docs/Model_Zoo_Repository_Guidelines.md`（develop 统一架构版）；其 §“Skills 维护源与目标上下文”（`:31-38`）明确与本任务同构的规则：“Skills 从 `rdk_x5` 默认分支维护……只表示维护源，不把任务目标默认为 X5；目标可能是 X5、S100/S100P/S600……”。
   - `docs/sample-standards/readme-contract.md`、`docs/sample-standards/inference-contract.md`、`AGENTS.md`。

## 安装源 ≠ 目标的实际差异点（不能照搬安装侧文档）

- 安装侧 `references/context-policy.md:24` 称“当前 S 线通常使用 `docs/manifests/`”——**目标仓库经 Glob 验证不存在 `docs/manifests/`（No files found）**，实际清单在 `docs/release/s`。技能自带脚本 `skill/scripts/inspect_repo.py:17-24,130` 已把 `docs/release/{platform}` 识别为 unified 布局，并内置警告“硬件按 sample 选择，绝不从分支或平台清单推断”。
- 当前 checkout 实为 git worktree：`.git` gitfile 指向 `rdk_model_zoo/.git/worktrees/rdk-b7-board-integration`，HEAD 为 `ref: refs/heads/codex/b7-board-integration-20260924`——分支名同样只是线索，不是硬件证据。
- S600 支持范围按每个 sample/artifact/runtime 分别判断，不能从 S 总清单聚合推导（context-policy `:24` 与样例矩阵一致）。
- 资产路径有历史差异需按 manifest 实条目核对：如 `models.yaml:203-205` 中 `s600/diffusiondrive_r34_256x1024_s600.hbm` 的 URL 位于 `rdk_s100/` 目录下。

## 建议阅读清单（S600 目标）

按序：`docs/Model_Zoo_Repository_Guidelines.md` → `platforms/registry.json` + `docs/release/platforms.json`（身份交叉核对）→ `docs/release/s/models.yaml`/`benchmarks.yaml`（按 `s600/` 条目过滤）→ 具体 sample 的 README 与 `runtime/`/`model/` 子指南 → `docs/sample-standards/` 两契约。

## 局限与已执行操作

- 已执行：仅只读 Read/Glob/Grep（上列路径均实际读取/检索）；未修改任何文件（AG06 的编辑权不适用于本任务）。
- 无法执行：无 shell/网络/SDK，`inspect_repo.py` 未运行（其结论仅以静态阅读为准）；`git status` 未跑，dirty 状态未知；commit 哈希未读取；未上板、未量化、未发布，也不得宣称任何 board-verified 结果。
- `fixture.json` 未读取（任务未引用它）；未触碰 memory、未创建子代理。
