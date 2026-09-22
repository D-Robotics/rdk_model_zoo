# B3 独立评审（2026-09-22）

- Reviewer：Codex；change-review；base `d91ac899e92fce0bd1465694e4ca133d29f6cb3a` → head `f552c52da5c50214be51ba1db84a108ebb1cd4d5`。期间作者仅将上轮板测暂停策略文档提交为 f552c52；代码范围不变。
- **review_decision=changes-required；delivery_readiness=not-ready；Closed=no**。问题来自代码和客户 README，不以当前板测暂缓本身作为 finding。

## B3-R1 / P2：FasterNet/FastViT 下载 CLI 默认值非法

- Axis：Technical Correctness；confidence：high；relation：introduced。
- 位置：`samples/vision/fasternet/model/download.py:91`、`samples/vision/fastvit/model/download.py:91`，后者 `:38` 的 asset_reference 默认值也错误。
- 两个 parser 默认均为 `base`，不在各自 VARIANTS 中；主机执行 `download.py --target x5` 均 rc=2，尚未发起网络请求即失败。FastViT 的 `asset_reference('x5')` 又默认 s，同样无资产。运行时与 download_target 的默认分别是 s / s12。
- 影响：省略可选变体的下载入口不可用，API/CLI/help 相互冲突。现有默认下载测试直接调用 download_target，绕过 parser，因此全部测试仍通过。
- 依据：源默认保留、CLI 默认可用、下载与模型绑定一致契约。
- 最小修正：统一每个 sample 的 binding/API/parser/help 默认（FasterNet s、FastViT s12）；增加真实 parser→main→下载委托的无网络 fixture 测试，避免只验证底层函数。

## B3-R2 / P2：双语文档的下载参数与 FastViT 输入路径不可执行

- Axis：Delivery Specification / Repository Standards；confidence：high；relation：introduced。
- 位置：FasterNet 与 FastViT 的 `model/README(_cn).md`、`runtime/python/README(_cn).md`、`evaluator/README(_cn).md` 下载示例；FastViT evaluator EN `:46` / CN `:41` 图片参数。
- 文档使用 x5 S/T2；FasterNet parser 只接受小写 s/t2，FastViT 只接受 s12/sa12/t12/t8。独立主机验证 S/T2 均 rc=2。FastViT evaluator 英文传 zebra.JPEG，中文传 bittern.JPEG，两者均不存在；实际输入 bucket.JPEG。中英成功判据也不一致。
- 影响：客户按文档无法准备模型，即便绕过下载，图片路径仍失败。
- 最小修正：逐层同步实际合法变体与存在的测试图，统一中英命令/成功判据；验证 fenced code 与行内 prerequisite，而非只校验 Markdown 链接和表格默认值。

## B3-R3 / P2：FastViT evaluator 把 FasterNet 指标标成源发布数据

- Axis：Delivery Specification / Technical Correctness；confidence：high；relation：introduced。
- 位置：`samples/vision/fastvit/evaluator/README.md:93–96`、`README_cn.md:85–88`。
- 两表为 FastViT-S/T2/T1/T0 和 31.1/15.0/7.6/3.9 M 参数，实际逐项等于 FasterNet 表，仅换了名称。`ac11571:samples/vision/fastvit/evaluator/README.md` 列 SA12/S12/T12/T8，参数 10.9/8.8/6.8/3.6 M；例如 S12 延迟 5.86ms、FPS193.87，而非错表的 S 6.73ms/FPS162.83。当前根 README 已是正确表，层级之间冲突。
- 影响：客户获得错误模型性能和精度记录，不能用“源数据未重测”解释迁移时串表。
- 最小修正：依据固定源逐列恢复正确 FastViT 表并核对中英/根/evaluator；不要新编数据或在这轮要求重新测量历史基准。

## B3-R4 / P2：未执行板测却在 evaluator 声称已有 smoke

- Axis：Delivery Specification / evidence；confidence：high；relation：introduced。
- 位置：四个 B3 `evaluator/README.md:53` 附近，均写 `the recorded 2026-09-21 smoke used abs diff < 1e-5`。
- 同文 reference-results 和批次记录均 board=not-run。该句是从 B2 复制的已执行口吻，不能作为 B3 本批事实。
- 最小修正：改为拟采用的验收规则；若引用 B2 先例，明确批次和证据来源。真实测试后再填 B3 日期/结果。顺带同步 FasterNet/FastViT evaluator 仍为26的旧测试数（实际各27，非单独阻断）。

## 独立验证与范围

- B3 108、前批 sample 339、shared71、checker27，合计 **545 tests 全过**。各套件独立进程 `.venv/bin/python -m unittest discover -s <suite> -t <suite>`。
- migration checker：**15 samples / 0 violations / 84 exemptions / rc=0**。这不覆盖上述命令可执行性和数据语义错误。
- 四 sample 转换 YAML 对 ac11571 字节核对全部一致；13 个发布变体表/默认绑定已核对。ConvNeXt 的额外 femto/nano 配方无资产，按素材保留，未虚构支持。
- 共用 ClassificationTask/runner/tensor_io 职责沿用已审实现，未发现本批 forward 混入下载或后处理。下载/CLI helper 与推理分离。转换未运行，源配方缺口已披露；生成制品改名不证明与发布制品等价。
- 未运行板测、OE 重建或数据集评估；无本轮数值等价结论。保留历史图表但不作为实测证据。
- 三维：代码默认入口及文档交付存在确认缺陷；结构/阶段职责与有限源能力保留核对通过；实际板端正确性仍待验证。

## 远程板测协调

用户 2026-09-22 明确希望主评审统筹，通过办公室 xgs-hp-ubuntu 的 Codex 与 Git 同步执行板测。Mac 仍不尝试连接局域网板卡。

推荐固定 Git SHA → 远程独立 checkout/worktree → 单次核实已知板身份 → 同板源对照/统一入口 → 完整代码、模型、输入哈希和 stdout/stderr 证据。当前 develop 相对本地缓存 origin/develop ahead55，不能假设远程 clone 已包含 B3；发布同步前须明确分支和包含内容。板测结果可作为单独证据提交回传，主任务独立复核。

新建“B3 远程板测”任务 `01a0c52f-df05-7293-892b-864d2de67c48` 目前在本地等待；应用拒绝 projectless 任务远程交接（仅支持项目内 Git 任务）。现有保存项目是父目录，未识别为 Git。正在通过既有远程任务预检 git/gh/认证/仓库位置，不覆盖其原任务，不读取或转移 token。远程预检未完成前 Board 继续 not-run，不反复探测。

## 后续

先整改 R1–R4 并做针对性 host 检查；远程 Git 项目/认证/固定 SHA 就绪后，按用户授权恢复远程板测（无需 Mac 回局域网）。暂缓板测不是通过声明。B3 Closed=no，不推进 B4。
