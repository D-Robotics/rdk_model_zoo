# README 源深度恢复专项包：Ultralytics YOLO / FCOS / ByteTrack / KWS（DOC-DEPTH-R1 + KWS-N1 作者记录）

Author: Claude Code + GLM。Reviewer: Codex（独立评审 pending，**本包不自关闭
DOC-DEPTH-R1 或 KWS-N1，不宣称 H1/H2/H9 整体完成**）。依据
[源图审计派发](2026-09-28-source-readme-image-independent-review.md)
（24 个目的地中的 Ultralytics/FCOS/ByteTrack）与
[KWS-N1](2026-09-28-kws-independent-review.md)。起始工作树基于
`bd06d2e9`（分支 `codex/b7-board-integration-20260924`），固定源 pin：
X5 `ac11571`、S `380e1a2`。

**改动范围**：上述四个 sample 的 10 份双语 README（12 个文件中的 FCOS
根 README 两份经核对已覆盖源内容、未改动）+ 新增 1 张固定源恢复图片
（`samples/vision/ultralytics_yolo/test_data/result_detect_yolo26.jpg`）+
本记录与 `evidence/2026-09-28-readme-depth-special-remediation/`。未改代码、
root 索引、skills、清单、计划台账、reviewer 文件及并行 worker 文件
（gemma/asr/classifiers 的既有改动未触碰）。无板测、无下载、无
量化/导出/OE/HMCT 执行；图片仅从固定 pin `git show` 恢复，不联网取图。

## 各 sample 恢复内容（双语同构）

### Ultralytics YOLO（根 README）

- 概览补回源"Algorithm Overview"的家族定位句：两个固定交付源 README 均将
  Ultralytics YOLO 描述为覆盖检测/分割/姿态/分类的实时视觉模型系列，
  YOLO26 作为同系列端到端家族并入本入口维护（仅补叙述，不动支持矩阵）。
- `expected-results` 补回两张源效果图（图注含 pin+SHA，均标注历史示意图、
  非本轮测量）：
  - S 源根 README 嵌入的 `test_data/result_detect.jpg`（`5d792a47…`，文件
    已在树中，本轮补回引用）；
  - S `ultralytics_yolo26` 交付自己的 `result_detect.jpg`（`2631c661…`，
    inventory `identical_files: []`），从 `380e1a2` 恢复为新文件
    `result_detect_yolo26.jpg` 并引用；与上一张非同字节、渲染不同
    （类别 ID+分数标签），不做去重合并。
- conversion 前次恢复的四张 dataflow 图及 DFL/YOLO26/pose-51 通道说明
  **未回退、未重写**；相关子目录（model/runtime/evaluator）经结构对照
  无源图缺失（inventory 计数为 0）。
- 未引用处置（记录于 evidence 映射，非静默丢弃）：`ultralytics_YOLO_Pose/
  Seg_demo.jpg`、`ultralytics_YOLO_CLS_demo.png` 为源树携带但源 README 从未
  嵌入的 IDE 截图（画面为标题 `RDK_S100_24GB` 的 SSH 会话，与 X5 树归属
  矛盾、无源图注可继承），维持不引用；`zebra_cls.jpg` 仅作为 CLI 输入图
  使用。

### FCOS（conversion README）

- 三张 `hb_perf` 快照从"正文提及但不引用"恢复为正式引用图（`toolchain-targets`
  内新增小节，双语），图注按逐图实际查看内容书写：NV12 → BPU `NV12TOYUV444`
  → `YUV444,NHWC,INT8` → `torch-jit-export_subgraph_0`（BPU）→ 15 个 INT32
  输出；B0 层级 64×64→4×4、B2 96×96→6×6、B3 112×112→7×7（stride 8–128）。
- 同段恢复源"Output Protocol"说明：5 路分类 + 5 路框回归 + 5 路 center-ness，
  Python runtime 按固定 shape 重排后解码；解码语义仍只在 runtime README
  （含已审阅的 `sqrt(sigmoid(cls_max)*sigmoid(center))` 公式），不重复。
- 根 README 经与源逐项核对：anchor-free 概述、五层级、l/t/r/b、论文与官方
  实现链接、demo 图及历史性能披露均已覆盖源内容，按"无因不改"未动。

### ByteTrack（根 README + evaluator README）

- 根 `overview` 恢复源算法解释：MOT 低分框丢弃问题、BYTE 两阶段关联四步
  流程，以及固定 S 源嵌入的检测示例条带 `image1.png`（`fdab9b40…`；初版图注
  误将其描述为论文三行动机图，已按 DOC-SPECIAL-R2 更正，见下）。
- 根 `expected-results` 恢复源"适用条件"：检出过少调低 `--score-thres`
  （初版曾并列 `--track-thresh`，DOC-SPECIAL-R1 已更正为与 tracker 代码一致
  的划分/代价语义），ID 频繁切换调 `--match-thresh`/`--track-buffer`；默认值
  按**当前** runtime 参数表核对（0.25/0.3/0.8/60，非照抄源文档缺省描述）。
- evaluator `reference-results` 恢复"MOT17-01/07-SDP"两段跟踪参考 GIF
  （`6b7a613f…`/`ff99c85a…`），标注为上游方法在该序列的历史可视化、非板端
  重跑；同节恢复源调参与多类跟踪说明（每类一个 tracker 或扩展 class_id）。
- `test_data/readme_img/image.png` 初版处置为"源从未嵌入、维持不引用"；
  DOC-SPECIAL-R2 更正后已按"源树携带、pinned 源 README 未嵌入"的显式身份
  补入根 `overview`（三行 (a)/(b)/(c) 关联说明图，`032728fb…`），见下。

### KWS（根 README，KWS-N1）

- 概览内新增 `### Algorithm and pipeline (MDTC)` / `### 算法与流程（MDTC）`，
  原地恢复算法与流程解释，不再只指向归档：
  - MDTC（Multi-Scale Dynamic Temporal Convolution）与 PaddlePaddle +
    PaddleAudio 框架来源，按固定源措辞**归属**（"固定源将……描述为"）；
    多尺度卷积/动态卷积/边缘友好/高精度等源特性句保留归属，并明确固定源
    只有编译制品、无训练代码，不把动态权重行为或精度措辞升级为本仓验证
    事实。
  - 时序模型消费什么、片段置信度如何产生：mono 16 kHz float32 → 前 60000
    采样（3.75 s，短补零）→ PaddleAudio fbank `[1,373,80]`（25 ms/10 ms/
    80 mel）→ BPU 概率 → 最大值归约为片段置信度、不叠加 sigmoid →
    `score >= threshold`（默认 0.5）。全部与 runtime README 及 KWS 主机
    评审记录一致。
- 保留不动：60000=3.75 s 修正、S100 唯一资产与显式拒绝矩阵、全部快速开始
  命令、历史 0.985/1.176 ms 证据边界、S 快照链接（降为补充历史入口）。

## 静态核对（证据见 evidence/ 目录）

1. **命令块**：12 份 README 全部 fenced 代码块（含围栏行）改动前后逐一
   diff，全部 IDENTICAL；`git diff --check` rc=0 —
   `command-blocks-and-diff-evidence.txt`（含被删除且未原样重写行的完整
   清单：仅 4 句被扩写的导语句，KWS 一句的语义已并入新流程段）。
2. **图片/链接**：四个 sample 的 `tools/sample_contract/check.py --sample`
   全部 0 violations / 0 exemptions（仅既有 policy skip）—
   `checker-after.txt`；新增引用全部手工核解析 — `image-reference-mapping.md`。
3. **双语一致**：新增行按 pin/SHA/参数/默认值/维度 token 逐对核对 en=cn —
   `bilingual-parity.txt`（42 token 全过）。
4. **逐图人工查看**：本轮引用的 9 张图（YOLO 两张 result、FCOS 三张
   dataflow、ByteTrack image1 + 两 GIF，另查看 3 张未引用截图与 image.png
   后作出处置）均实际查看后书写说明，未以文件名代替内容；其中 ByteTrack
   image1/image.png 初版的视觉身份写反，DOC-SPECIAL-R2 已更正（见下）。
5. **恢复文件哈希**：`result_detect_yolo26.jpg` 落盘后 SHA-256 与 inventory
   一致 — `restored-images-sha256.txt`。
6. **范围**：`git status` 中本包改动仅上述 10 份 README + 1 张图 + 本记录/
   evidence；并行包（gemma/asr/classifiers）文件未触碰。

## 边界与未完成

- 板端、真实 SDK/OE/HMCT、量化精度、数据集评测：not-run（按 2026-09-28
  用户裁定不在本包范围，不作为阻塞）；所有历史图/数值仅作源 pin 记录引用。
- 独立评审、全分支回归、H1/H2/H9、DOC-DEPTH-R1 其余目的地状态：不由本包
  声明关闭；KWS-N1 与 Ultralytics/FCOS/ByteTrack 的图源恢复整改已完成待审。
- Ultralytics 根概览的家族句为源措辞转述，未新增任何能力声明；YOLOv5/
  YOLOE/yolo26_depth 独立能力边界、退役重复系列不重新引入的裁定均未触碰。

## 整改追加：DOC-SPECIAL-R1（ByteTrack 调参表述与代码不符）

Reviewer（Codex）独立评审指出：本包初版把 `--track-thresh` 写成"调高会过滤
掉更多低分框 / 检出过少时调低"，与 tracker 实际实现不符；`--match-thresh`
方向与 `--track-buffer` 缩放未说明。整改前逐条核实代码
（`runtime/python/tracker_backend/byte_tracker.py:148` `det_thresh =
track_thresh + 0.1`、`:171-175` 两次关联按 `score > track_thresh` 与
`0.1 < score < track_thresh` 划分、`:260` 新轨迹门槛、`:149`
`buffer_size = int(frame_rate / 30.0 * track_buffer)`、`:204` +
`matching.py:41` `lapjv(cost_limit=match_thresh)`、`matching.py:87` 代价 =
1 − IoU、`:171-179` 分数融合；detector `--score-thres` 在 tracker 之前过滤）。

整改内容（仅文字，EN+CN，命令块零改动）：

- 根 `expected-results` 调参段与 evaluator "Tracker parameter tuning and
  applicability" 重写：`--score-thres` 在 tracker 前生效（检出过少调它，
  调低 `--track-thresh` 找不回 detector 丢弃的框）；`--track-thresh` 只划分
  tracker 输入（高于→第一次关联，(0.1, track-thresh)→与仍跟踪目标的第二次
  关联，新轨迹需 ≥ `track_thresh + 0.1`）；`--match-thresh` 为第一次关联
  接受的最大代价（1 − IoU 融合检测分数，越大允许越不相似的匹配，第二次
  关联保持固定 0.5 上限）；`--track-buffer` 按 30 fps 帧数计并以
  `frame_rate / 30` 缩放（`--frame-rate` 默认 30）。
- 逐条代码对照与边界说明见
  [evidence/doc-special-r1-code-verification.md](evidence/2026-09-28-readme-depth-special-remediation/doc-special-r1-code-verification.md)。
- 范围外观察（未越界修改）：`runtime/python/README*.md` 参数表的一行式
  flag 描述仍较粗略，留给维护者另行处理。

复核：四 sample checker 0 violations / 0 exemptions
（`checker-after-doc-special-r1.txt`）；12 份 README fenced 命令块与包前
快照仍全部 IDENTICAL；`git diff --check` rc=0；整改事实双语逐 token 一致。
DOC-SPECIAL-R1 整改完成待独立复审，本包不自关闭该 finding 或批次。


## 整改追加：DOC-SPECIAL-R2（ByteTrack 两张 PNG 视觉身份写反）

Reviewer（Codex）独立评审实际查看两个文件后指出：本包初版把 `image1.png`
与 `image.png` 的视觉身份写反。整改前用图像 Read 逐一重看工作树文件（均与
S pin `380e1a2` 逐字节一致，哈希自始正确，仅描述错误）：

- `test_data/readme_img/image1.png`（sha256 `fdab9b40…`，1,265,627 字节，
  **pinned 源 README 唯一嵌入的 PNG**）：一条横向三帧街区条带——彩色检测框
  带逐框置信度（左帧如 0.94/0.92/0.83，中间帧红三角旁低至 0.43），三角标记
  （左右帧黄色、中间帧红色）标注一名被跟随行人；无 (a)(b)(c) 行标题。
- `test_data/readme_img/image.png`（sha256 `032728fb…`，3,510,950 字节，
  **随源树携带、pinned 源 README 未嵌入**）：三行说明图，`Frame t1/t2/t3`
  表头，行标题 "(a) detection boxes"、"(b) tracklets by associating high
  score detection boxes"、"(c) tracklets by associating every detection
  box"；顶行中被跟踪的较小行人分数 0.8 → 0.4 → 0.1（初版误写 0.9 起，
  最终窄修更正，见下），(c) 行以虚线框（标注 0.4、0.1）重新关联其低分
  检测。

整改内容（EN+CN，仅根 README 图注段，命令块零改动，未改文件名、未互换
文件、不声称源嵌入过 image.png）：

- `image1.png` 图注改为按实际条带内容描述，保留源嵌入身份（pin+SHA）。
- 按派发的可选项同时嵌入 `image.png`：图注与 alt 文本显式写明"随固定源树
  携带、pinned 源 README 未嵌入"，用于解释低分恢复机制；描述仅限可见内容
  （行标题文字、0.8→0.4→0.1，初版 0.9 起点有误已由最终窄修更正、(c) 行虚线
  0.4/0.1），不添加不可见结论。
- `evidence/image-reference-mapping.md`：ByteTrack 两条 PNG 记录按正确身份
  重写（含各自字节数/哈希），顶部加 DOC-SPECIAL-R2 更正说明，去重条目 3
  同步更新；不删除初版错误记录的更正痕迹。

复核：`tools/sample_contract/check.py --sample samples/vision/bytetrack`
（其余三 sample 未受本轮影响亦复跑）0 violations / 0 exemptions
（`checker-after-doc-special-r2.txt`）；12 份 README fenced 命令块与包前
快照仍全部 IDENTICAL；`git diff --check` rc=0；两图引用 en/cn 均可解析且
身份句双语一致。DOC-SPECIAL-R2 整改完成待独立复审，本包不自关闭该
finding 或批次；R1 整改状态保持不变。

## 整改追加：ByteTrack 最终窄修（图注分数归属 + mot20 融合条件）

Reviewer 最终窄修意见，两项均先核实后改（仅文字，EN+CN，命令块零改动）：

- **image.png 分数归属**：重看该图（`032728fb…`，未变），(a) 行 t1 四个框为
  0.9（较高前景行人）、0.8（较小的被跟踪者）、0.9（黑衣男士）、0.1（右缘）；
  0.8 框位置与 t2 的 0.4、t3 的 0.1 一致。初版"walking woman 分数
  0.9→0.4→0.1"把两个框混为一谈，已改为：被跟踪的较小行人 0.8 → 0.4 → 0.1，
  0.9 框属于另一名较高前景行人（双语同改，仅用可见数字）。
- **match-thresh 融合条件**：核实 `main.py:17` `--mot20` 为受维护的
  store_true 开关（默认 `false`，传入 `TrackingConfig`，runtime 参数表有
  记录）；`byte_tracker.py:202-203`（首次关联）与 `:246-247`（unconfirmed
  阶段）均为 `if not self.args.mot20: fuse_score(...)`。初版"代价 = 1 − IoU
  并与检测分数融合"的无条件表述已改为"默认模式融合；`--mot20`（默认
  `false`）关闭融合，代价即 1 − IoU"（根 README 调参段 + evaluator
  match-thresh bullet，双语）。

逐条对照见
[evidence/bytetrack-caption-mot20-verification.md](evidence/2026-09-28-readme-depth-special-remediation/bytetrack-caption-mot20-verification.md)。
复核：四 sample checker 0 violations / 0 exemptions
（`checker-after-bytetrack-final-correction.txt`）；12 份 README fenced 命令
块与包前快照仍全部 IDENTICAL；`git diff --check` rc=0；修正事实双语逐
token 一致。R1/R2 修复保持不变；本窄修完成待独立复审，不自关闭任何
finding 或批次。
