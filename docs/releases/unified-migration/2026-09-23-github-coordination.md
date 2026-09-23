# GitHub 同步与 B7 下一轮任务

2026-09-23 用户恢复工作，授权使用板卡环境中的 HP Ubuntu，并要求每轮修改通过 GitHub 同步。本记录取代暂停交接中“不使用远程电脑、未提交”的当前操作约束；历史报告仍保持原样。

## 当前检查点

B3–B6 既有成果和 B7 中途实现入库用于协作，不是客户发布。B7 六套主机测试本轮全部返回 0；migration checker 为 36 samples / 14 violations（YOLOv5 C++ 双语 README 缺章节）/ 84 原 B9 豁免，返回 1。完整结果见 evidence/2026-09-23-github-sync-checkpoint.json。B7 独立评审仍 changes-required，板端新验证 not-run，不得宣称完成。

## 分工和同步规则

- Codex 负责 GitHub 同步、整合和独立评审；HP Ubuntu 的实际开发由 Claude Code 执行。
- 仅经 GitHub 交换代码。远端从已推送 develop 创建独立 B7 工作分支，不覆盖任何现有工作区。每个完整整改单元由协调者提交并推送；评审通过后再合并 develop。禁止 force push、清理他人文件或改写历史证据。
- 开发者不自行标记独立评审通过，不进入 B8。板测可在新环境具备准确板卡配置和所需资产时执行；缺设备/模型/SDK则明确 not-run，不猜地址或借旧记录声明通过。S600 MobileNetV2 C++ 的既有用户豁免保留。

## 下一轮：B7 整改

先读 AGENTS.md、X5/S Spec、台账、2026-09-23-claude-code-handoff.md 和 B7 独立初审。该交接中逐项技术问题仍是整改输入，本记录覆盖其中已过时的暂停/远程限制。固定源 SHA 不变，不随 develop 新提交漂移。

1. 逐项核对并解决 YOLOv5 C++ 的容量/stride/dtype/scale 校验、资源生命周期、源默认变体、NMS 边界、调度参数、构建目标身份及完整机器可比 dump；补行为测试，勿用字符串断言替代。
2. 独立复现并整改 FCOS、LPRNet、MODNet、YOLOWorld 的公开 API gate、绑定/参数/有限值/不可变 context、完整 source/unified evaluator。保留有意源兼容行为与缺失资产事实。
3. 审核全部 B7 双语 README，命令/API/参数/shape/结果/局限必须与实现一致；修复现有 14 项 checker 问题，禁止加豁免或降级规则。
4. 跑 B7、受影响 shared 和既有批次回归及 CI 同命令 checker；保存命令、返回码、版本、代码及输入 hash、完整日志。板端验证与主机验证分开。
5. 更新 B7 作者整改报告、evidence、台账，保留独立 changes-required 与 Closed=no，交 Codex 复审。每个单元交付准确修改范围、验证结果及待办。

外部原计划的可移植快照见 2026-09-23-execution-plan-snapshot.md；恢复后的操作规则以本记录及用户最新指示为准。
