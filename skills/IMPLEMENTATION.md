# 全平台 Skills 本地实施

本轮以 `rdk_x5` 为统一维护分支，目标覆盖 X5、S 系列、X3 及历史布局。维护分支不是任务的平台默认值。源码基线与原始交付包摘要见 [source-baseline.json](verification/source-baseline.json)。

## 实施范围

1. 导入七个候选技能，保留原许可、成员版本和独立资源目录。
2. 分离 SKILL_ROOT 与 REPO_ROOT；目标平台、版本、硬件和 runtime 根据用户约束及目标仓库事实判断。
3. 支持 `docs/manifests/`、`docs/release/`、`release/` 清单；多候选显式选择，缺清单不编造。
4. 通过工具回归、资源闭包、真实仓库清单检查与可执行的隔离 Codex 行为评测记录验证结果。

不执行远端发布、Hub owner 迁移、模型下载或板卡测试，不更改根 VERSION、模型清单或发布工作流。候选包中关于 Hub 与正式发布的说明是后续工作，不代表已经完成。

## 本地复验

在仓库外的隔离 Python 3.10+ 环境安装 `skills/requirements.txt`，从仓库根运行：

```text
python skills/tools/sync_references.py
python skills/tools/validate_pack.py
python -m unittest discover -s skills/tests -v
```

共享参考只修改 `skills/_shared/`，然后执行 `python skills/tools/sync_references.py --apply`。以上检查不执行模型，也不证明 Agent 行为或硬件兼容性。

行为评测方法见 [evals/README.md](evals/README.md)。本轮实际执行及未运行项以 [验证报告](verification/REPORT.md) 为准；保留原始用例，新增跨平台用例不能代替执行记录。
