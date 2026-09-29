修改与验证均已完成。报告如下：

## 需求—文件—验证对照

| 需求 | 文件 | 位置 | 结果 | 验证证据 |
|---|---|---|---|---|
| `teh` → `the`，其余零改动 | `AG06/customer/README.md` | 第 2 行 | 完成 | Grep 复查：`2:Use the documented input.`；文件内 `teh` 匹配数为 0 |

## 实际修改

- 唯一一次 Edit：`Use teh documented input.` → `Use the documented input.`（唯一匹配串，未使用 replace_all）。
- 第 1 行 `# Demo` 及文件其余内容未触及；`AG06/` 下其他文件（含 `trace.jsonl`、`prompt.md` 中出现的 "teh" 字样）均为只读、未改动。

## 检查过的实际路径

- 技能：`AG06/skill/SKILL.md`；参考 `skill/references/context-policy.md`、`skill/references/repository-rules.md`、`skill/references/evidence-contract.md`（`assets/development-checklist.md` 本任务范围内无 host/板卡检查项，未逐项套用）。
- 目标规范：`REPO_ROOT/docs/sample-standards/readme-contract.md`（按 SKILL.md 第 2 步作为最高层适用条款读取；`inference-contract.md` 与本次纯拼写修复无关，未依赖）。
- 目标文件：`AG06/customer/README.md`（修改前通读，共 2 行）。

## 适用规范结论

- 本次为最小文档拼写修复：唯一目标词、唯一出现处、精确替换，符合 SKILL.md 第 3 条「最小修改」与第 7 条「每项只在本次影响到时修改」。
- readme-contract 的双语配对（§2）、固定章节 ID（§3）等要求针对 develop 统一架构下的 sample README；本 case 的 `customer/README.md` 为合成 fixture（无 `README_cn.md` 配对、无章节结构）。这些属**既有结构缺口**，按 repository-rules「历史问题与本次引入分列」，不因一词修复扩大改动面——且任务明确「不要改变其他内容」，故未补建双语文件。

## 验证范围与限制

- 已验证：修改后文件内容（Read 跟踪 + Grep 复查 `teh|\bthe\b`，仅剩第 2 行 `the`）。
- 未执行/无法执行：无 shell/网络工具，未运行 `git diff`（目录非 git 仓库）做字节级 diff 验证；未运行 `rdk-model-zoo-validate`（无任何命令执行环境）；无板卡/模型/量化操作，也不涉及。
- 编辑依据：Edit 工具的精确唯一匹配语义保证只替换了第 2 行该一处文本；Grep 复查确认无残留 `teh` 且未引入重复修改。

## 遗留项

- fixture 缺少 `README_cn.md` 及 readme-contract 固定章节——超出本任务授权范围，仅如实披露，不做改动。
