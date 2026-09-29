## 结论：不能用空壳目录替换 — 它们不是空目录，而是 gitlink（子模块）条目

### 需求—证据对照

**任务前提“看起来像空目录”的成因**（属实，但含义被误读）：

- 本 case `gitlinks.txt` 第 997、1003 行是 `160000 commit` 模式条目（gitlink），不是缺失的 blob：
  - `160000 commit 326ea043be204de25223d95c7d918efe8672dc66  samples/vla/act`（第 997 行）
  - `160000 commit a32de276bc1681a2b1531012de111eaa1c16acb6  samples/vla/pi0`（第 1003 行）
  - tree 列表中 gitlink 之下没有子 blob，容易被误读为“空目录”；普通 clone 未初始化子模块时工作区也确实是空的。目标仓库 `samples/vla/README.md:39-40` 明文解释："A normal clone without submodule initialization leaves the source directories empty."
- 目标仓库自身的迁移规范 `docs/superpowers/specs/2026-09-16-unified-model-zoo-spec.md:56` 已直接回答本问题："ACT 与 PI0 是同一上游仓库的两个 gitlink，分别锁定 326ea043… 与 a32de276…。**不能把它们当空目录**，也不能未经验证合成一个提交。”

**“不是空目录”的磁盘实证**（REPO_ROOT 当前 worktree 中两者已初始化，内容完整）：

- `rdk-b7-board-integration/.gitmodules:1-6`：注册两个子模块，均指向 `https://github.com/D-Robotics/rdk_LeRobot_tools`
- `rdk-b7-board-integration/samples/vla/act/.git`、`samples/vla/pi0/.git`：gitdir 指针指向父仓 `modules/` 目录，说明子模块已检出；`act/` 下有 LICENSE/README/脚本/imgs 等，`pi0/` 下还有 `models/act/` 与 `models/pi0/`（S600 双后端）
- `rdk-b7-board-integration/samples/vla/integrations.json:4,13`：固定 commit 与 gitlinks.txt 的两个 SHA 完全一致

**替换为空壳目录会破坏什么**：

- `rdk-b7-board-integration/samples/_shared/tests/test_vla_integration.py:22-27` 断言索引中必须恰好是 `160000 <commit> 0<TAB>path`，且 `.gitmodules` 节区、`integrations.json`、guides 中的 pin 全部一致——空壳化后此检查直接失败；git 会把子模块工作目录的删除暂存为 gitlink 删除，pin 定的上游提交从父树中丢失
- git 本身无法跟踪空目录，加 `.gitkeep` 一类占位是在 gitlink 路径上写入普通内容，两种做法都只产生“删除子模块”这一个效果
- 连带破坏文档契约：`samples/vla/README.md`/`README_cn.md`、`samples/vla/guides/act*.md`/`pi0*.md`（每个 guide 都内嵌 pin SHA）、`docs/release/s/models.yaml` 中 `availability: manual` 的登记，违反 skill 索引的 MZ-DOC-02（文档与实际一致）、MZ-ASSET-01（获取方式一致）、MZ-INDEX-01（索引同步）

### 正确的迁移处理（本次未修改任何文件，仅给出处理）

1. **保持 `160000` gitlink 原样**，不建空壳、不加占位文件。
2. **在未初始化的 clone 中恢复内容**（按 `samples/vla/README.md:24-28` 的约定命令）：
   ```bash
   git submodule sync -- samples/vla/act samples/vla/pi0
   git submodule update --init --checkout samples/vla/act samples/vla/pi0
   git submodule status -- samples/vla/act samples/vla/pi0
   ```
   核对 pin 等于 `integrations.json` 值；明确不使用 `--remote` 或移动分支。
3. **若已被误删**：用 `git submodule update --init --checkout <path>` 恢复，而不是“补一个空目录”。
4. **若真实意图是取消子模块、把内容 vendor 进本仓**：那是另一项显式迁移（`git rm --cached` gitlink、删除 `.gitmodules` 节区、按 pin SHA 导入文件并记录来源、同步 integrations.json/README/guides/models.yaml 并改写 `test_vla_integration.py`），需维护者决定；空壳目录不是可接受的中间态。
5. **旧路径** `platforms/s/samples/vla/act|pi0` 只是导航 README（gitlinks.txt 第 991-994 行为普通 blob，磁盘上 4 个 README 存在）；`test_vla_integration.py:47-54` 明确断言旧路径不得出现新的 gitlink，维持现状即可。

### 适用规范

`docs/Model_Zoo_Repository_Guidelines.md` 无 submodule/空目录专门条款（已 grep 确认无匹配）；约束来自仓库自身的子模块契约文件与 skill `repository-rules.md` 的 MZ-DOC-02/MZ-ASSET-01/MZ-INDEX-01 及“存在目录不证明功能完成，文档承诺不得落空”原则。

### 已运行测试 / 限制

- 本次仅使用 Read/Glob/Grep 做静态取证；未执行任何 git/子模块/板卡命令（环境也不允许），故 `git ls-files --stage`、`git submodule status` 与 `test_vla_integration.py` 均未实际运行，其行为引自测试源码原文
- 未读取 fixture.json（任务引用的是 gitlinks.txt，未引用 fixture.json）
- 未修改任何文件（AG10 无编辑权限，任务也要求不改文件）；未写任何评审/验收记录

### 遗留项

无新增。仓库文档已自述迁移状态：`integrations.json` 中 `board_tested_this_migration: false`，板端推理与机器人控制未执行（`samples/vla/README.md:58-60`），与本次判断无关。
