# RDK Model Zoo Skills

本目录是 Model Zoo 专属 Skills 的维护源。Hub 只镜像注册的 `skills/<name>/`，不是维护源，也不安装模型仓库本身。Pack 从 `rdk_x5` 默认分支维护，但该分支只是文档来源，不能把用户目标默认为 X5；目标平台和 ref 必须按用户约束及实际目标仓库事实核对。

**交付状态：候选源码，尚未合入/发布。** Pack 目标版本为 1.0.0；迁移入口 `rdk-model-zoo` 的 Skill 版本为 1.1.0，其余新技能为 1.0.0。Pack 与成员版本分别管理，既有模型版本文件不改变。

## 能力

| Skill | 职责 |
|---|---|
| [rdk-model-zoo](rdk-model-zoo/SKILL.md) | 现成模型选择、查询发布数据、按真实命令使用样例 |
| [rdk-model-zoo-repo](rdk-model-zoo-repo/SKILL.md) | 目标仓库/版本/规范上下文与工作流导航 |
| [rdk-model-zoo-integrate](rdk-model-zoo-integrate/SKILL.md) | 自有模型接口接入与 OE 工具链交接 |
| [rdk-model-zoo-develop](rdk-model-zoo-develop/SKILL.md) | 新增、维护、修复与规范化交付 |
| [rdk-model-zoo-validate](rdk-model-zoo-validate/SKILL.md) | 分范围功能/精度/性能验证与证据 |
| [rdk-model-zoo-review](rdk-model-zoo-review/SKILL.md) | sample-audit 和 change-review，默认只读 |
| [rdk-model-zoo-release](rdk-model-zoo-release/SKILL.md) | 模型/Skills 发布准备、Hub 迁移与隔离 |

量化、编译与调优继续使用既有 OE Skills。声明 Skill 名称不代表它已安装；能力缺失时走明确交接，不自行重造工具链。

## 目标仓库与平台

每次使用先解析 `SKILL_ROOT`（当前 Skill 安装目录）和 `REPO_ROOT`（用户指定的 Model Zoo 工作区）。目标可能是 X5 的 `rdk_x5`/`x5-v*`、S100/S100P/S600 的 `rdk_s`/`s-v*`、X3 的 `rdk_x3`/`x3-v*`，或用户指定的 `rdk_x5_legacy`/历史 ref。分支名和模型文件名只是线索；以目标 ref 的 README、实际代码和（有清单时）Manifest/metadata 交叉核对。S100、S100P、S600 的支持必须落实到具体 sample、artifact 和 runtime，不能从 S 版本的总清单推断全仓兼容。用户给出的平台、版本、路径与候选 ref 冲突时保留约束并报告，不切换分支来消除冲突。

当前目标 ref 优先查 `docs/manifests/`；历史 ref 可能保留 `docs/release/` 或根 `release/`。多个清单要显式选择，缺少清单要保持未知，不能拿维护源的模型数据代替目标 ref。

## 目录与资源

每个技能包含 SKILL.md、治理卡、按需参考、模板（适用时）和 evals/tasks.yaml。三个运行期辅助工具分别随拥有它的 Skill 独立分发，不依赖同级 Skill 文件；每个 Skill 目录复制到受支持的发现目录后都能独立解析自身 references/assets。

`_shared/` 是共享规则的唯一编辑源；`tools/sync_references.py` 按 pack.json 生成各 Skill 内的参考副本。提交这些小型生成参考是为平铺安装提供资源闭包；CI 检查不漂移。不要人工修改副本，不通过 ../ 引用另一个 Skill 的资源。

`tools/` 和 `tests/` 是维护者的 Pack 开发工具，不要求单独安装某个 Skill 的用户拥有它们。`pack.json` 是本项目维护清单，不是声称 Hub 已支持的新 schema。

## 本地校验

Python 3.10+ 是代码和依赖的最低声明；本交付实际在 Python 3.13.5 上测试，其他版本及宿主会话需要另验。依赖安装只在获批准的隔离虚拟环境执行：

```bash
python3 -m venv .venv-skills
. .venv-skills/bin/activate
python -m pip install -r skills/requirements.txt
python skills/tools/sync_references.py
python skills/tools/validate_pack.py
python -m unittest discover -s skills/tests -v
```

上面的 cwd 是 Model Zoo 仓库根；仅检查不必重新创建环境。若修改 `_shared/`，明确执行写操作 `python skills/tools/sync_references.py --apply`，然后重跑只读检查。

运行期脚本：

```bash
python3 skills/rdk-model-zoo-repo/scripts/inspect_repo.py --repo "$REPO_ROOT"
python3 skills/rdk-model-zoo/scripts/read_catalog.py --repo "$REPO_ROOT" --model "$MODEL_QUERY"
python3 skills/rdk-model-zoo-validate/scripts/validate_evidence.py "$RECEIPT" --evidence-root "$EVIDENCE_ROOT"
```

变量为已确认的目标路径/查询，不由脚本猜测。read_catalog 需 PyYAML；validate_evidence 需 jsonschema；inspect_repo 只需 Python 与 Git。脚本不自动安装、不联网、不执行 sample。

## 安装与发布

开发期：将需要的完整 `skills/<name>/` 复制或链接到所选 Agent 官方支持的 Skill 发现目录，先确认目标无同名旧副本，不覆盖用户文件；重新加载会话后验证发现和触发。不能只复制 SKILL.md。把源码放在 skills/ 不代表 Agent 自动加载。

正式期：完成源 Release 和 Hub 切换后，才使用 Hub 的安装入口，例如 `npx skills add d-robotics/rdk-skills --skill rdk-model-zoo-review`。此命令是上线后的用法，不表示新技能现在已在 Hub 发布。

源头技能文件是发布物，但规范、模型清单、源码应从用户实际检出的目标 Model Zoo 版本读取。老 Tag 可以配合新安装 Skills 使用，不改历史 Tag。分支/platform profile 是线索，不是完整兼容矩阵；安装源的 `rdk_x5` 不会覆盖 S、X3 或 legacy 的目标身份。

## 评测与状态

75 条 eval 定义（原始 70 条及新增 5 条跨平台用例）覆盖五类：正确性、可发现性、安全、有效性、效率。它们不是已执行结果。参见 [行为评测协议](evals/README.md)；本地工具测试不能代替 Agent 基线/对照、真实板卡或生产 Hub 同步。

## 来源与许可

[来源与变更](NOTICE.md) 记录旧入口和 Review 的参考来源。新写文档采用 CC-BY-4.0，脚本采用 Apache-2.0；顶层 Skill license 字段按当前 Hub 兼容约定保留。维护者合入迁移时复核原有署名和许可，不自动给历史代码重新授权。
