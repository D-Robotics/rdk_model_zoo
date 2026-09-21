# B2 独立关闭确认（2026-09-21）

- Reviewer：Codex；范围：9097e3c → **21c833a3a8259ea9aee5b604e4020323897a739f**，仅 B2-R1-E 证据确认及 N1/N2 采纳验证；开审工作树 clean。
- **review_decision=pass；delivery_readiness=ready（B2 既定交付范围）；Closed=yes。** 无剩余必需整改项。B2-R1/R2/R3/R1-E 均 closed；N1/N2 已确认。B3 可由用户安排启动，本轮不执行。

## 证据确认

已读取 `evidence/2026-09-21-b2-r1e-default-entry/` 的 capture 脚本及两板完整 JSON，记录明确为 2026-09-21 08:20–08:21 UTC 的新执行，不是回填旧记录。

- S100/S600 身份、起止时间、cwd/argv、rc、完整 stdout/stderr 齐全。
- 两板各四个部署文件 SHA-256 均与当前 HEAD 工作树对应文件一致，包含 N1 后的共享 cls_binding；旧 bundle 哈希仅作为底座身份，不冒充 overlay 后身份。
- 两板 lite0 部署模型与 bundle 模型哈希相同，且与已持久化 models.sha256 相符；输入图及标签哈希与仓库相同。
- 两板 dry-run 均解析对应 lite0，rc=0；省略 variant/asset-id 的真实入口均 rc=0，完整五项类别及打印分数与原 lite0 对照记录匹配。
- S600 显式 lite2 对照 rc=0。它是额外控制用例，新增 capture 未单列其模型哈希；不将其扩展为重新完成该变体全量数值验证，原显式变体矩阵仍保留。

**数值边界：**本次完整捕获的是 CLI 输出，分数仍为六位小数。因此确认的是完整 Top-5 类别与打印精度下的分数一致，不是 float32 原始输出逐位一致。此边界不影响默认入口修复的验收，也不扩大数据集精度、性能或重新量化声明。板测由原执行者 Claude 运行；reviewer 独立核对记录和哈希，未连接板卡重跑。

## N1/N2 与针对性验证

N1 已将无默认映射时的候选设为空；单资产也明确拒绝，和 docstring 对齐。新增 EfficientViT 单资产空映射测试覆盖原 reviewer 反例。N2 中英 evaluator 均为 28 tests，并保留原 25+3 的来源说明。

独立运行 EfficientNet 28、EfficientViT 27、shared 71，共 **126 tests 全过**；CI 同命令 **11 samples / 0 violations / 84 exemptions / rc=0**。此前独立复审 436 tests 通过作为历史回归记录保留；作者本轮声明全量 437 tests 通过，本轮 reviewer 未重复全套，不混写成独立执行。

三维结论：Repository Standards 的当前文档状态/接口契约符合整改要求；Delivery Specification 的板端部署绑定和执行输出已补齐；Technical Correctness 的默认选择及无映射边界通过针对性验证。本范围无新增阻断 finding。

## 状态与交接

B1、B2 均已独立关闭。S100P 仍仅拒绝负例；B1 S600 MobileNetV2 C++ 依用户决定保持 not-run。B9 的 84 条精确文档欠账仍须在 B9 移除；B2 关闭不代表全部迁移完成，不改变当前 develop 不直接面向客户交付的约定。历史初审/复审原文保留，最新状态由本报告覆盖。
