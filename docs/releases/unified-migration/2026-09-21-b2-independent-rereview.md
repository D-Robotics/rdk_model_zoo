# B2 整改独立复审（2026-09-21）

> **最终独立确认（Codex，21c833a）：B2-R1-E 证据通过，N1/N2 已确认；B2 全部 findings closed，pass / ready（既定范围），Closed=yes。** 见 [关闭报告](2026-09-21-b2-independent-closure.md)。以下历史结论保留。


- Reviewer：Codex；change-review；base `bea64b1` → head `0a6deaafb55525cf00121f32d57f99b5ed97b0e4`；开审工作树 clean。
- **review_decision=changes-required；delivery_readiness=not-ready；Closed=no**。代码的原始 S 默认选择回归已修复，唯一关闭阻断为受影响板端复测的可追溯证据。B1 保持已关闭，不进入 B3。

## B2-R1-E / P2：overlay 后的默认入口板测缺少代码绑定及原始输出

- Axis：Delivery Specification / evidence；confidence：high；relation：整改验证证据缺口，继承 B2-R1 关闭条件，不是推理失败。
- 位置：`evidence/2026-09-21-b2-board-smoke-evidence.json:126–129`，尤其 `board_verification` 字段。
- 当前记录仅叙述“四个 fixed files overlaid”、S100/S600 rc=0 和部分六位小数结果；没有 overlay 文件哈希/可验证部署版本、执行时间、精确 cwd/argv、完整 Top-5 输出或日志链接。`d54d1cf...` 是修改前 bundle 的哈希，不能绑定 overlay 后执行的代码。
- 已保留的 review-inputs 五板 JSON 与 bea64b1 逐字节相同，是修复前显式 asset-id 路径的记录；本地 `/tmp/b2-board-results` 也只有原五板记录。不能用这些旧记录证明新增默认入口已经在修复版本执行。
- 影响：作者报告的复测成功可如实记录，但 reviewer 无法核对其与当前代码的对应关系；六位小数的部分输出也不足以独立确认“逐位一致”。不据此认定作者未执行或板测失败。
- 依据：既定 B2-R1 要求受影响 S 默认入口验证；review evidence-contract 要求 ref/代码身份、实际目标、输入及结果绑定。
- 最小补齐：优先持久化已有两板复测记录，包含实际时间/板身份、精确命令与 cwd、部署四文件 SHA-256（或可验证的部署树标识）、所用 lite0 模型/输入身份（可引用原证据中同一制品的哈希）、退出码和完整 stdout/stderr。将新记录链接到整改节。无保存的原记录时只重跑两板默认入口并留证，不重跑完整变体矩阵。不得事后把当前主机哈希冒充当时板端哈希。

## 非阻断建议

### B2-N1 / P3：共享接口缺失默认映射时的行为与文档不符

`samples/_shared/cls_binding.py:297–300`：无默认映射时设置 `matches=list(records)`；若 target 恰好只有一个资产，实际会选中它，尽管新接口 docstring 承诺无默认会报错。reviewer 用 `dataclasses.replace(efficientvit.BINDING_TABLE, default_variant={})` 复现：`default_variant_for('x5')` 返回 None，`resolve_selection(table,'x5')` 却成功选择 m5。

当前发布配置没有触发这个形态（EfficientNet 已覆盖所有有资产的 target，S100P 无资产），所以不阻断 B2。建议明确选择语义：要严格拒绝则保留空 matches/显式报错并加单资产反例；若允许唯一候选自动选择则修正文档契约。不要用 S100P 的零资产测试替代这个边界测试。

### B2-N2 / P3：EfficientNet evaluator 双语仍列 25 tests

根 README 已更新为 28，evaluator reference-results 仍是 25。属于同日旧自检数，建议标成历史或同步到 28；不影响本次实际验证结果，不作为关闭阻断。

## 原 finding 裁定

| Finding | 状态 | 独立复核 |
| --- | --- | --- |
| B2-R1 | 代码通过；证据关闭待补 | x5 默认 b2，s100/s600 默认 lite0；下载同样按 target；auto+S100/S600 返回 lite0；显式 lite2 与错误组合/无资产拒绝测试通过；板端摘要来源为作者 |
| B2-R2 | closed | 16 份客户根/evaluator README 已回填板测、比较口径与证据链接；未扩大 raw tensor、精度或性能声明 |
| B2-R3 | closed | 28=26+2；每侧精确平局与跨实现差异已区分，原始记录未改；gap=0 裁定不外推近似平局 |

## 独立执行验证

- 全部 13 个套件独立进程执行：B2 105 + B1/YOLO/OCR 233 + shared 71 + checker 27 = **436 tests 全部通过**。
- CI 同命令：`.venv/bin/python tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json`，**11 samples / 0 violations / 84 exemptions / rc=0**。
- 额外主机检查：运行与下载三 target 默认选择、auto 身份解析、上述单资产缺映射反例。
- 无 reviewer 板端执行、模型下载或 OE 重建。本次未将主机测试升级为板测结果。B1 S600 C++ 继续依用户决定不补测。
- 三维结论：Repository Standards 的主要文档整改通过；Technical Correctness 的原产品回归修复通过（另有不影响现有配置的 P3 边界建议）；Delivery Specification 尚缺整改板测可追溯记录。

## 关闭要求

补齐 B2-R1-E 后进行只针对证据的独立确认；如果同时采纳 P3 代码建议，再跑相应共享契约/消费者测试。不存在要求重新进行完整五板矩阵的新增条件。历史评审保留，本次状态由本报告更新。
