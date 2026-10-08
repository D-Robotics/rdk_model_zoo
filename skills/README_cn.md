[English](README.md) | 简体中文

# RDK Model Zoo Skills

本目录提供 Model Zoo 工作流 Skills，涵盖模型查询、目标仓库检查、集成、开发、验证、评审和发布准备。使用时以用户指定的目标仓库、分支和硬件为准，并从该检出的 sample、清单与运行接口读取支持信息。

## 能力

| Skill | 职责 |
|---|---|
| [rdk-model-zoo](rdk-model-zoo/SKILL.md) | 现成模型选择、查询发布数据、按真实命令使用样例 |
| [rdk-model-zoo-repo](rdk-model-zoo-repo/SKILL.md) | 目标仓库/版本/规范上下文与工作流导航 |
| [rdk-model-zoo-integrate](rdk-model-zoo-integrate/SKILL.md) | 自有模型接口接入与 OE 工具链交接 |
| [rdk-model-zoo-develop](rdk-model-zoo-develop/SKILL.md) | 新增、维护、修复与规范化交付 |
| [rdk-model-zoo-validate](rdk-model-zoo-validate/SKILL.md) | 分范围功能/精度/性能验证与证据 |
| [rdk-model-zoo-review](rdk-model-zoo-review/SKILL.md) | sample-audit 和 change-review，默认只读 |
| [rdk-model-zoo-release](rdk-model-zoo-release/SKILL.md) | 模型/Skills 发布准备与分发 |

量化、编译与调优使用对应 OE Skills；本 Pack 提供 Model Zoo 仓库和交付工作流。

## 版本

| 组件 | 版本 |
|---|---|
| Pack | 1.1.0 |
| rdk-model-zoo | 1.1.2 |
| rdk-model-zoo-repo | 1.1.1 |
| rdk-model-zoo-integrate | 1.0.1 |
| rdk-model-zoo-develop | 1.1.1 |
| rdk-model-zoo-validate | 1.1.1 |
| rdk-model-zoo-review | 1.1.1 |
| rdk-model-zoo-release | 1.0.1 |

版本以 [`pack.json`](pack.json) 清单为准。

## 目标仓库与平台

先解析 `SKILL_ROOT`（当前 Skill 安装目录）和 `REPO_ROOT`（用户指定的 Model Zoo 工作区）。常见目标包括 X5 的 `rdk_x5`/`x5-v*`、S100/S100P/S600 的 `rdk_s`/`s-v*`、X3 的 `rdk_x3`/`x3-v*`，以及用户指定的 `rdk_x5_legacy` 或历史 ref。按用户指定的目标 ref 检查 README、代码、清单和 metadata；S100、S100P、S600 的支持信息应定位到具体 sample、artifact 和 runtime。

清单按目标 ref 中实际存在的路径解析。常见位置为 `docs/manifests/`、
`docs/release/`、`release/` 和按平台组织的 `docs/release/{x5,s,x3}/`；
部分检出还包含 `platforms/` 下的按平台清单。`inspect_repo` 输出候选清单、
平台清单、分支角色、工作区状态和可选 sample 文件清单。`read_catalog` 使用
PyYAML 读取实际清单：单个候选可直接选择，多个候选时用 `--manifest` 指定路径；
`--benchmark-manifest` 可指定基准数据，`--model` 按模型 ID 或名称筛选。
例如 `--manifest docs/release/s/models.yaml`。

## 目录与资源

每个技能包含 SKILL.md、治理卡、按需参考、模板（适用时）和 evals/tasks.yaml。三个运行期辅助工具分别随拥有它的 Skill 独立分发，不依赖同级 Skill 文件；每个 Skill 目录复制到受支持的发现目录后都能独立解析自身 references/assets。

单独分发 Skill 时，复制完整的 `skills/<name>/` 目录，其中包括 `SKILL.md`、references、适用的 assets、脚本和模板。共享参考副本随各 Skill 目录一起提供。

## 命令行工具

```bash
python3 skills/rdk-model-zoo-repo/scripts/inspect_repo.py --repo "$REPO_ROOT"
python3 skills/rdk-model-zoo/scripts/read_catalog.py --repo "$REPO_ROOT" --model "$MODEL_QUERY"
python3 skills/rdk-model-zoo-validate/scripts/validate_evidence.py "$RECEIPT" --evidence-root "$EVIDENCE_ROOT"
```

变量分别是目标 checkout、查询词、receipt 文件和 evidence 目录。`read_catalog`
需要 PyYAML，`validate_evidence` 需要 jsonschema，`inspect_repo` 使用 Python 与 Git。

## 安装与发布

将完整的 `skills/<name>/` 目录复制到 Agent 支持的 Skill 发现目录，然后重新加载会话以使用该 Skill。规范、模型清单和源码信息来自用户指定的 Model Zoo 检出版本。

## 来源与许可

[来源与许可](NOTICE.md) 记录本 Pack 的参考来源、文档与脚本许可。
