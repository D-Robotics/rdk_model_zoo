# B2 独立评审及 B1 关闭确认（2026-09-21）

> **最终独立确认（Codex，21c833a）：B2-R1-E 证据通过，N1/N2 已确认；B2 全部 findings closed，pass / ready（既定范围），Closed=yes。** 见 [关闭报告](2026-09-21-b2-independent-closure.md)。以下历史结论保留。


> **整改独立复审（Codex，0a6deaa）：R2/R3 closed，R1 代码验证通过，但 overlay 后 S 默认入口板测缺少可追溯记录（B2-R1-E）。** 436 tests 与 CI 通过；B2 仍 changes-required / not-ready / Closed=no。见 [复审报告](2026-09-21-b2-independent-rereview.md)。优先补存已有记录，无需完整重跑。


- Reviewer：Codex；change-review；B2 base `b835e7fe1113af0efbb2f2404566fb7b8088c57e` → head `716bdca92d704a1e7b7ce36d7c9ba04c3160b311`；开审工作树 clean。
- B2：**review_decision=changes-required；delivery_readiness=not-ready；Closed=no**。不进入 B3。
- B1：复核 `b835e7f` 的 R1a/R1b 文档修复通过，连同前次关闭的 R2–R6，**B1 Closed=yes / review_decision=pass / delivery_readiness=ready（既定批次范围）**。S600 MobileNetV2 C++ 按用户决定不补测，仍 not-run；不代表全仓迁移完成或可发布 develop。

## Findings

### B2-R1 — P2：EfficientNet 的 S 默认路径选择不存在的 b2 变体

- Axis：Technical Correctness & Regressions；confidence：high；relation：introduced regression。
- 位置：`samples/vision/efficientnet/runtime/python/model_binding.py:136`；`samples/vision/efficientnet/model/download.py:47,65,100`；消费方 `_shared/cls_binding.py:273`。
- 证据：统一表全局 `default_variant='b2'`；下载器也默认 b2。reviewer 实际执行 `main.py --dry-run --target s100` 与 `--target s600`，均 rc=2，报告无 b2 资产；直接 `asset_reference('s100')` / `('s600')` 同样失败。显式 lite0 正常。源 `rdk_s@380e1a2` 的 main.py 明确默认按 SoC 使用 lite0。
- 影响：在匹配 S 板卡且已准备模型时，省略 variant/asset-id 仍不能运行；`download.sh s100`/s600 的省略变体路径也不可用。已有板测全部显式给 asset-id，未覆盖该路径。
- 依据：迁移保留合理源默认行为及目标选择契约；可选 variant 不应隐式选择该 target 不存在的资产。
- 最小修正：按已解析 target 选择默认（x5 b2，s100/s600 lite0），保持显式 variant/asset-id 的精确匹配及 S100P 拒绝；下载器、CLI help、双语文档一致。增加省略变体、auto 身份、显式错误组合的回归测试；若改共享 API，验证其他消费者。板端只补受影响的 S 默认入口，不重复全矩阵。

### B2-R2 — P2：客户 README 未同步已完成的板测，验收口径也与实际不符

- Axis：Delivery Specification / Repository Standards；confidence：high；relation：introduced。
- 位置：四个 B2 sample 根 README 双语 support matrix / 其后段落，及 evaluator 双语 metrics/reference-results；例如 `efficientnet/README.md:29–39`、`efficientformer/evaluator/README.md:82`。
- 证据：16 份根/evaluator README 仍称 board smoke pending/not-run，而报告与台账已宣称完成；evaluator 写“类别 ID 与 raw scores 完全一致”，实际比较的是 softmax 后 Top-K、采用容差且存在精确平局。
- 影响：客户与 Agent 从正式入口得到错误状态和不可满足的比较条件；测试标题齐全不能证明文档语义正确。
- 依据：README 契约的验证状态、结果解释与可操作性；沿用 B1-R5 的交付要求。
- 最小修正：逐 target/variant 同步两语言状态并链接可追溯证据，区分 Top-K 冒烟、raw tensor、数据集精度和性能；明确容差和精确平局规则。保留 S100P 负例、转换/基准未重测边界。不要把未测 raw tensor 写成通过。

### B2-R3 — P2：板测汇总错误计数，并把平局误写为跨实现逐字节相同

- Axis：Delivery Specification / Technical Correctness of evidence；confidence：high；relation：introduced。
- 位置：`evidence/2026-09-21-b2-board-smoke-evidence.json:125`，同文件 s1 note 与 x5-4g result；批次报告 §6.5 第一个 X5 行和 tie 披露；台账 efficientformerv2 Board 列。
- 证据：原始记录共 9+9+5+5=28 次对照，26 次 ids_equal=true、2 次 tie_resolved=true（同一 s1 变体在两块 X5 各一次）。入库摘要却写“28 exact-id + 1 tie-resolved”及“ids 全等”。794/851 在各自实现内部相等，但 legacy 分数为 0.00418911874294281，unified 为 0.004189117345958948，跨实现并非逐字节相同。
- 影响：证据计数与数值结论不可直接用于验收；“一个变体”与“两次板位用例”被混用。
- 最小修正：修正摘要、报告、台账并区分严格 ID 一致与精确平局裁定；把原始完整 per-ID 记录作为证据入口。当前记录支持精确 gap=0 的裁定，不据此批准未来任意 gap<1e-6 的近似平局自动放行。
- reviewer 裁定：当前两次确切平局可接受为稳定 Top-K 排序的边界差异，不要求改回不稳定排序，不认定模型推理缺陷。S100P 原 harness 将负例当正例判 fail，按保留的 rc=2 和明确报错可裁定负例通过。

## 独立验证

| 项目 | 结果 |
| --- | --- |
| B2 efficientnet / efficientformer / efficientformerv2 / efficientvit | 25 / 25 / 26 / 26，102 tests 全过 |
| B1 + YOLO/OCR 回归 | 233 tests 全过 |
| shared / checker | 71 / 27 tests 全过 |
| 总计 | **433 tests 全过** |
| CI 同命令 checker（含原 B9 基线） | 11 samples / 0 violations / 84 exemptions / rc=0 |
| 转换材料对源 Git 字节比较 | efficientnet X5 3 + S 13；former 2；formerv2 3；vit 1，共 22 文件一致 |
| 本地板测 bundle 对当前仓库源文件 | 107 个可对照 py/sh/yaml/json 文件逐字节一致 |
| 本地 bundle 模型 SHA-256 | 19/19 与 models.sha256 相符；仅证明本地文件身份，不代表发布者背书 |
| EfficientNet S 默认选择反例 | 两 target 均可主机复现失败，显式 lite0 对照成功 |

测试命令为 `.venv/bin/python -m unittest discover -s <suite> -t <suite>`，各套件单独进程。完整输出与检查器 argv 见同名 evidence JSON。

三维结论：规范方面，布局/阶段职责/静态检查通过，但 README 内容未交付完整；交付方面，变体及转换材料保留、板端作者记录支持显式资产路径的冒烟，但状态与数值摘要需修正；正确性方面，共享前处理/推理/后处理被复用，forward 未混入下载、可视化或后处理，确认存在 S 默认选择回归。

## 板测复核及保留材料

reviewer 未连接板卡、未重跑推理、未运行 OE 或下载模型。板测来源是 Claude 原执行者。reviewer 从本机 `/tmp/b2-board-results` 找到五板原始 JSON、harness 和完整模型哈希表，已原样保留到 [review-inputs](evidence/2026-09-21-b2-review-inputs/)（来源及 SHA 见独立证据 JSON），避免临时目录清理后证据丢失。保留 raw status，不覆盖作者的原始记录。

X5 两板各 9 次对照+4 CLI；S100/S600 各 5 次对照+1 CLI；S100P 3 个拒绝负例。合计 28 对照+10 正向 CLI+3 负例。原始数值支持两次精确平局裁定及其余 26 次 ID 一致；不证明全量 raw output 等价、数据集精度或性能。客户文档中历史图表/截图未作为本次运行证据。

## B1 收尾

校准命令已改为 conversion cwd 下的 `python3 get_calibration_data.py`；相对 ONNX/校准目录/输出路径闭合；中英均明确 mean 相同、scale 不同，未擅自认定任一系数正确。三个源文件 SHA 不变，ResNet 52 tests 通过。原始初审/复审报告保留，当前关闭结论由本节更新。

## 下一轮

修复 B2-R1–R3，作者自检与独立评审分别记录。R1 如改变默认入口行为，补主机目标矩阵和受影响 S 板入口验证；已有显式资产板测可复用。R2/R3 仅文档/证据修改不重跑板测。用户免测决定仅适用于 B1 S600 MobileNetV2 C++。B2 复审通过前 Closed=no，不进入 B3。
