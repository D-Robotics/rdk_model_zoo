# README 源深度恢复 B1/B2 九 sample（DOC-DEPTH-R1 作者记录）

Author: Claude Code + GLM。Reviewer: Codex（独立评审 pending，**本包不自关闭
DOC-DEPTH-R1，不宣称 H1/H2/H9 整体完成**；DOC-DEPTH-R1 对其余十五个目的地仍然开放）。
依据 [2026-09-28-source-readme-image-independent-review.md](2026-09-28-source-readme-image-independent-review.md)
的 B1/B2 派发。起始 HEAD `f09b78aa`（base `5311f9c4` 之后的工作树），分支
`codex/b7-board-integration-20260924`。

**改动范围**：`samples/vision/{resnet, mobilenetv1..v4, efficientnet,
efficientformer, efficientformerv2, efficientvit}` 的根 `README.md` +
`README_cn.md`（18 个文件，纯增补 + 两处论文引用修正），以及 resnet
`test_data/` 新增 5 张从固定源 `git show` 恢复的图片。本记录与
`evidence/2026-09-28-readme-depth-b1b2-remediation/` 为新增。其它 sample、根
README、CLAUDE.md、AGENTS.md、skills、manifest、计划台账、旧 reviewer 报告、
代码与 model 资产均未改动；并行 MiniCPM/PointNet 工作未触碰。无板测、无下载、
无量化/OE/HMCT 执行；历史截图仅作源记录引用，不作为本仓新证据。

## 恢复内容（每 sample）

统一做法（双语同构）：在既有 `overview` 章节内新增 `### Algorithm
background` / `### 算法背景`（无新锚点，符合 readme-contract §3 固定锚点
规则），恢复源 README 的算法解释、特性条目（保留源措辞语义与论文/参考实现
链接）与架构图；架构图与历史截图带源 pin、路径与 SHA-256 前缀的图注。历史
推理截图放既有 `performance` 章节（源 README 的放置位置），resnet 的 S 侧
结果截图放 `expected-results`（S 源 README 的 "Inference Result" 对应位置）。
逐图映射（含 SHA 与舍弃/去重理由）见
[evidence/image-reference-mapping.md](evidence/2026-09-28-readme-depth-b1b2-remediation/image-reference-mapping.md)。

| Sample | 恢复的解释与图 | 历史结果截图 | 性能表 |
|---|---|---|---|
| resnet | 残差学习/快捷连接；resnet18 轻量、resnet50 瓶颈块（S50 源）、resnet152 深层容量（S152 源）；论文图 5 基础块+瓶颈块图（X5/S 同字节，`cebea796…`，`git show` 恢复） | X5 white_wolf（`16c9d04e…`，恢复）；S 侧 resnet18/50/152 三张 zebra 截图（`9e7a4c8a…`/`be58ba73…`/`229c2b23…`，按变体重命名恢复，注明源未说明板卡） | 新增根 `performance` 章节：X5 ResNet18 表（71.5/70.5、2.95 ms、449+）原样保留，注明线程条件源未说明并链接 evaluator；S 侧未发布数据的披露保留 |
| mobilenetv1 | 深度可分离卷积解释 + `depthwise&pointwise.png`（`48d3cb64…`，已在本树） | `inference.png` bulbul Rank-1（`6f07652b…`） | 既有表未动 |
| mobilenetv2 | 倒残差+线性瓶颈解释 + `mobilenetv2_architecture.png`（`7995faf5…`）；另将源树携带但源 README 未嵌入的 `seperated_conv.png`（论文图 2 演化图）恢复为正式引用插图 | `inference.png` Scottish deerhound（`7097e2e3…`） | 既有表未动 |
| mobilenetv3 | NAS+NetAdapt、SE、h-swish 解释 + `MobileNetV3_architecture.png`（`bc978181…`，论文图 4 SE 在残差路径） | `inference.png` kit fox（`03b15192…`） | 既有表未动 |
| mobilenetv4 | UIB 统一块设计与 mobile MQA 解释 + `MobileNetV4_architecture.png`（`944a191d…`，论文图 4） | `inference.png` great grey owl（`64930905…`） | 既有表未动 |
| efficientnet | 复合缩放解释（分辨率/深度/宽度同缩）、X5 B2-B4 与 S Lite0-4 关系及 224/240/260/300/380 几何；架构图单份去重（X5 `EfficientNet_architecture.png` = S `efficientnet_architecture.png`，`f0c7ccbe…`，图注注明两个源名与同一摘要） | `inference.png` redshank（`8ecf7529…`；注明输入是 redshank.JPEG 而非 Scottish_deerhound.JPEG） | 既有 X5/S 两张表未动 |
| efficientformer | 延迟驱动设计、维度一致 MetaBlock 解释；`latency_profiling.png`（`a3439462…`，图注明确这是论文 iPhone 12/CoreML 实验数据、**不是 RDK X5 实测**）+ `EfficientFormer_architecture.png`（`4fe4662f…`，MB4D/MB3D） | `inference.png` bittern（`7ddbce07…`） | 既有表未动；L1/L3 Float Top-1 同为 76.75% 系源表原样（见下"已知观察"） |
| efficientformerv2 | 细粒度联合搜索、统一 FFN、改进 MHSA、高分辨率注意力解释 + `EfficientFormerV2_architecture.png`（`0a3fd26e…`，论文图 2 (a)–(f)） | `inference.png` goldfish（`907925ac…`） | 既有表未动 |
| efficientvit | 访存受限开销、级联组注意力、BN 部署友好解释；三张图全恢复引用：`comparison_between_transformer_and_cnn.png`（`be1e2e39…`，论文图 2）、`mhsa_computation.jpg`（`4dda6352…`，实际内容为论文图 3 的 MHSA 占比-精度研究，图注按实际内容描述，不沿用源 alt 文本的"MHSA Computation"标题）、`efficientvit_msra_architecture.png`（`403d1c63…`，论文图 6 (a)(b)(c)） | `inference.png` hook（`2a23e138…`） | 既有表未动 |

图片恢复仅用 `git show ac11571/380e1a2`，落盘后 SHA-256 与
[图像审计 inventory](evidence/2026-09-28-source-readme-image-audit/inventory.json)
逐字节核对一致（`evidence/restored-images-sha256.txt`）；不外网取图。
八个 sample 的源引用图已在本树（摘要与 inventory 一致），本轮只补引用与
上下文，不改文件；resnet 缺失的 5 张为新恢复文件。

## 去重与舍弃处置（摘选，完整见 evidence 映射）

- resnet 架构图在 X5（`ResNet_architecture.png`）与 S 三个 resnet 交付
  （`resnet_architecture.png`）为同一字节文件，保留一份，图注披露双方来源
  名与摘要。
- efficientnet 架构图同理跨源去重（两个文件名，同一字节）。
- `ResNet_architecture2.png`（两 pin 的源 test_data 均有）在两个源 README
  正文均未嵌入、无源解释文本可携带，不恢复；处置记录于 evidence 映射，不作
  无说明丢弃。
- mobilenetv2 `seperated_conv.png` 源 README 仅在目录清单提及、未嵌入正文；
  现按其真实内容（论文图 2）恢复为带解释的引用图。
- S 侧 mobilenetv1-v4/efficientnet 源 README 无图片引用（inventory 0 项），
  无图可映射；其文字性差异已在映射表说明。

## 论文引用修正（披露）

- efficientformer：引文 arXiv `2206.00171`/"ImageNet Transformers at MobileNet
  Speed" 与固定源 pin（`ac11571`）的 `2206.01191`/"Vision Transformers at
  MobileNet Speed" 不一致，按源 pin 对齐（中英两处）。离线环境未做外部核对，
  依据为源 pin 的引用。
- efficientnet：引文 arXiv `1905.11942` 与源 pin 的 `1905.11946` 不一致，按
  源 pin 对齐（中英两处）。
- 全部 diff 中被删除且未原样重写的行只有上述两处（`evidence/command-blocks-and-diff-evidence.txt`）。

## 已知观察（不阻塞、未擅改）

- efficientformer 源性能表中 L1 与 L3 的 Float Top-1 同为 76.75%（上游论文
  中 L1 数值不同）；本仓规则是发布值不可推断/擅改，现按源表原样保留并在此
  提请 reviewer 注意。
- `mhsa_computation.jpg` 文件名与内容（MHSA 占比研究）不完全对应；按用户
  "不盲目继承错误标题"要求，图注按实际内容书写，文件名不改（避免无谓的
  历史名变更）。

## 复审修正（Codex 两项图文准确性发现的整改记录）

独立复审提出 DOC-B1B2-R1、DOC-B1B2-N1 两项修正；本包仅改这四处文字（每处
中英各一）并更新本记录与 evidence，其余内容、命令、性能表与图片文件一律
未动。逐字前后对照见
[evidence/review-fixes-r1-n1.md](evidence/2026-09-28-readme-depth-b1b2-remediation/review-fixes-r1-n1.md)，
复验输出见
[evidence/checker-after-review-fix.txt](evidence/2026-09-28-readme-depth-b1b2-remediation/checker-after-review-fix.txt)。

- **DOC-B1B2-R1（mobilenetv3 图注）**：原图注把 SE 门控写成"gates the 1×1
  output / 门控 1×1 输出"，与图不符。重看 `MobileNetV3_architecture.png`
  （`bc978181…`，未改文件）确认：SE 支路（Pool → FC ReLU → FC hard-σ →
  ⊗）作用于 NL depthwise 3×3 之后、最终 NL 1×1 投影之前的扩展通道。中英
  图注已按实际数据流改写（"在 NL depthwise 3×3 之后……门控作用于扩展通
  道，门控结果再经最后的 NL 1×1 投影输出"）。
- **DOC-B1B2-N1（efficientformerv2 背景句）**：原句"cut the cost that kept
  earlier hybrids off mobile devices / 削掉了早期混合架构无法登上移动设备
  的成本"属过度泛化（同包 EfficientFormer 根 README 本就携带 iPhone
  12/CoreML 移动端测量）。中英改为客观表述："相对 EfficientFormer 基线
  降低了注意力与下采样开销，同时保持 MobileNet 量级的尺寸和速度"，不再
  声称早期混合架构无法部署。
- 复验：mobilenetv3、efficientformerv2 两个 sample checker 0 violations；
  `git diff --check` rc=0；四个文件 fenced 命令块与 HEAD 仍 IDENTICAL。
  该两项发现是否关闭由 Codex 复核决定，本记录不自关闭。

## 保留核对（静态证据）

1. **命令块前后对照**：18 个 README 的全部 fenced 代码块（含围栏行）与 HEAD
   版本逐一 diff，全部 IDENTICAL；`git diff --check` rc=0。资产 id、参数默认
   值、支持矩阵、板测证据句均无改动 — `evidence/command-blocks-and-diff-evidence.txt`。
2. **图片引用覆盖**：两 pin 源 README 的每个图片引用在当前双语 README 中均有
   引用或显式处置；en/cn 每 sample 图片引用数一致（5/2/3/2/2/2/3/2/4）。
3. **checker 输出**：九个 sample 的 `tools/sample_contract/check.py --sample`
   全部 0 violations（每 sample 1–2 项既有 policy skip，与本轮无关）—
   `evidence/checker-after.txt`（含本地链接 R-README-LINKS 检查通过；恢复的
   5 张图使 resnet 的引用全部可解析）。venv：`rdk_model_zoo/.venv`。
4. **逐图人工核对**：每张图（含恢复前后）均实际查看后书写图注（白狼 Rank-1、
   bulbul、Scottish deerhound、kit fox、great grey owl、redshank、bittern、
   goldfish、hook；zebra 0.9985/0.9956/0.9649；三张架构/研究图内容逐一比对），
   未以文件名或字数测试代替。
5. **文件名面变化**：仅 `samples/vision/resnet/test_data/` 新增 5 文件；
   `git status` 中本轮范围外无本包改动（并行包的既有改动未触碰）。

## 边界与未验证项

- 板端、真实 SDK/OE/HMCT、量化精度：not-run（按 2026-09-28 用户裁定不在本包
  范围，不作为阻塞）；所有历史截图/性能表仅作为源 pin 记录引用。
- 独立评审、全分支回归、H1/H2/H9：开放，不由本包声明关闭。
- DOC-DEPTH-R1 其余十五个目的地（含 FCOS conversion、ByteTrack evaluator）
  待后续分组包处理。
