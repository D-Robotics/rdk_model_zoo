# B1 整改独立复审（2026-09-21）

- Reviewer：Codex；模式：change-review，原评审基点 `c218e8622a2cb2025b6807ce2d36ef5c3df3d500`，整改 HEAD `dd609118205507c1302041387fe022ccef8875d7`。
- review_decision：**changes-required**；delivery_readiness：**not-ready**；**Closed=no，不进入 B2**。
- 原始独立评审保留；本报告更新 finding 状态，不覆盖历史结论。开审工作树 clean。

## 未关闭问题（findings first）

### B1-R1a / P2：ResNet152 校准命令与 cwd 不匹配

- Confidence：high；axis：Delivery Specification / Repository Standards；relation：introduced in R1 remediation。
- 位置：`samples/vision/resnet/conversion/README.md:124–128`、`README_cn.md:115–118`。
- 证据：文档明确 cwd 为 conversion 目录，随后执行 `python3 samples/vision/resnet/conversion/get_calibration_data.py`。该相对路径在指定 cwd 下不存在；实际文件就在 cwd 中。静态路径解析已确认 false，无需 OE 即可证明。
- 影响：客户或 Agent 按文档执行会在加载脚本前失败；改在仓库根目录执行又会把相对校准输出写到错误位置，不能与下一步 YAML 的 cwd 连通。
- 依据：README 契约的执行目录、输入输出和可操作性要求；原 R1 要求补齐变体转换文档。
- 最小修正：两种语言均使用明确的 conversion cwd 和 `python3 get_calibration_data.py`，核对下载、校准和编译的相对路径连续性。

### B1-R1b / P2：ResNet152 归一化一致性声明与源码矛盾

- Confidence：high；axis：Technical Correctness / Delivery Specification；relation：introduced documentation claim over preserved source inconsistency。
- 位置：`samples/vision/resnet/conversion/README.md:131–133`、`README_cn.md:121–123`；事实来源 `get_calibration_data.py:53`、`resnet152_config.yaml:18`。
- 证据：脚本统一乘 `0.017`，YAML 为逐通道 `0.01712475 0.017507 0.01742919`；两份 README 却称变换链与 YAML 条目一致。相同 README 后面的历史参数表也列出了不同系数。
- 影响：读者会把已有数值差异误认为已对齐，错误评估转换配方的重现能力。这里未实测量化影响，不推断精度损失或编译结果。
- 依据：README 契约要求参数与代码一致、已知缺口和验证边界明确。
- 最小修正：保持已迁入源文件及 SHA 不变，双语文档明确 mean 相同、scale 不同，属于保留的源配方差异，未经 OE 重建和数值对照确认；收紧“完整/可重现”表述，不擅自选择一套系数作为正确答案。

## 原 findings 复核

| Finding | 本次状态 | 依据 |
| --- | --- | --- |
| B1-R1 | **partial / open** | 三个转换文件与 `rdk_s@380e1a2` 原文件逐字节一致，ResNet50 未虚构配方；文档剩余 R1a/R1b |
| B1-R2 | **closed** | soc_name + board_type 别名拒绝、主机 24 tests、实板正负例记录；当前 launcher SHA 与证据一致 |
| B1-R3 | **closed** | V3/V4 逐 config 区分 RGB/BGR、224/256 和缺失 deploy 图；限制已披露，不代表转换重建通过 |
| B1-R4 | **closed** | S medium Y256/UV128 与 binding 对齐，V4 20 tests 通过 |
| B1-R5 | **closed** | README/台账区分板位、变体、语言与自评；S100P 仍仅负例，S600 C++ 仍 not-run |
| B1-R6 | **closed** | 精确 rule/path/line/message 基线及多 sample 修复；带基线 0 violations，无基线 84 violations；27 项检查器测试含消息不匹配、unused 和多 sample 回归 |

## 独立执行的验证

`.venv/bin/python -m unittest discover -s <tests目录> -t <tests目录>`，各套件独立进程：

| 套件 | 通过数 |
| --- | --- |
| mobilenetv1 / v2 / v3 / v4 | 17 / 24 / 17 / 20 |
| resnet / ultralytics_yolo / paddle_ocr | 52 / 59 / 44 |
| shared / sample_contract | 71 / 27 |
| 合计 | **331** |

CI 同命令：`tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json`：7 samples、0 violations、84 exemptions、rc=0。去除 exemptions：84 violations、rc=1。这些检查没有覆盖上述 README 语义错误。

三维度结论：Repository Standards 的结构与门禁检查通过，README 可操作性未过；Delivery Specification 的资产迁移和所需板测范围已覆盖，文档交付仍未过；Technical Correctness 未发现 R2/R6 实现的新阻断问题，但归一化一致性声明错误。

## 板端证据与边界

本次只审查原执行者（Claude Code）记录，不连接板卡、不重复运行。X5 4GB/8GB Python 各 5/5；S100/S600 Python 各 7/7；S100 MobileNetV2 C++ 正例；S100P Python 2 个拒绝负例及 R2 launcher 4 个用例（含一个 gate 放行对照）。S100P **无正向模型推理验证**；S600 C++ **not-run**，在已记录 B1 范围之外。

当前 launcher SHA-256 为 `01c0c6d4bb71a49f21046b0925db4b194a231ec24ade000e6275fcdccd6985bf`，与板端复测记录一致。其余 runtime 未在本轮整改变更，复用已有板测。部分原始日志仍引用执行者 `/tmp`，此处采纳的是入库结构化记录，不宣称 reviewer 独立重现板端结果。Top-K 冒烟不能替代全量 raw tensor 等价、数据集精度或重新量化验证。

## 下一次关闭条件

只修复 R1a/R1b 的双语文档，针对性核对路径、参数和配方限制；同步报告后独立复核。若仅文档修改，无需重复板测。B9 必须消除 84 条欠账并移除基线及 CI exemptions 参数。通过前 B1 保持 Closed=no，B2 pending。
