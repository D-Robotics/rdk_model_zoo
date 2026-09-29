# README 深度恢复（12 个分类器 sample）— 作者报告

Author: Claude Code + GLM（实现）；待 Codex 独立评审。Base: 本工作分支
`codex/b7-board-integration-20260924`，含其他 worker 的并发未提交改动（本包未触碰）。
本报告落实 [源 README 插图独立评审](2026-09-28-source-readme-image-independent-review.md)
中 DOC-DEPTH-R1 分派给本包的 12 个目的地：
convnext、edgenext、fasternet、fastvit、googlenet、hgnetv2、mobileone、repghost、
repvgg、repvit、resnext、vargconvnet。

## 范围与源

- 只改了 12 个 sample 的根 `README.md` / `README_cn.md`（24 个文件，+763/−7 行）。
  未改其它 sample、根 README、CLAUDE.md/AGENTS.md、skills、manifest、计划台账、
  旧 reviewer 报告；未触碰任何代码、模型资产或 test_data 文件本体。
- 源 pin：X5 侧 `rdk_x5 @ac11571`（googlenet/hgnetv2/mobileone/repghost/repvgg/
  repvit/resnext/vargconvnet 八个 sample 的 README 现引用完整 SHA
  `ac115717197920355fc390bb04299b20e6436864`，同一提交）。适用 S 源 `380e1a2`
  经逐 sample 核对：这 12 个 sample 目录在 S pin **均不存在**（见
  [image-provenance.json](evidence/2026-09-28-readme-depth-classifiers-remediation/image-provenance.json)
  的 `s_pin_check`），因此每个 sample 只有单一源，**无跨源去重场景**；
  inventory 各行 `ref` 也均为 ac11571。
- 图片本体无需恢复：36 张被源 README 引用的图片（architecture/inference 等）
  在当前 `test_data/` 已存在且与 ac11571 逐字节一致（同文件 `image-provenance.json`
  的 36 条 SHA 对照全部 `identical_to_source: true`）。缺失的只是 README 中的
  引用与上下文文字。未从外网取图。

## 恢复内容（按 sample 的源→维护位置映射）

完整映射见
[source-to-maintained-map.json](evidence/2026-09-28-readme-depth-classifiers-remediation/source-to-maintained-map.json)。
共性做法（不是图片附录）：

1. **算法背景**：在根 README 既有 `overview` 锚点章节内、维护实现段落之前，
   恢复源 "Algorithm Overview / Features" 的特性列表（每 sample 3–4 条，按源
   内容翻译为双语），并把源架构图放回特性列表之后，配说明文字（图内容按实际
   逐图查看撰写，注明上游论文图号）。图片说明统一标注：恢复自 rdk_x5 @ac11571、
   图为上游训练结构、实际部署制品是 INT8 量化变体（224×224 NV12）——不把训练
   结构图冒充部署图。
2. **历史运行截图**：放回 `expected-results` 章节既有效果描述之后，逐图核实
   图中实际内容后撰写双语说明（如 convnext：cheetah，top-1
   `cheetah, chetoh, Acinonyx jubatus: 0.8048811`，标签按历史原样保留），统一
   声明这是源交付的历史截图（rdk_x5 @ac11571 旧版 Python 入口），不是本仓库
   当前入口的运行结果，也未跑板。
3. **性能条件**：各 sample 已发布历史性能表与条件全部保留原位（B4 批次
   googlenet/hgnetv2/… 的表或其 evaluator 委托均未动）。唯一数值变动见下节。
4. **目录树**：源 README 的 Directory Structure 树不再重复——现有 `directory`
   章节的逐目录职责清单是维护等价物（映射文件中逐项注明 "not re-added" 及
   理由）；无内容被无说明丢弃。

### convnext 性能记录修正（请评审重点复核）

当前 README 原注释"已发布表覆盖 nano/pico/femto——不含 atto"与两处固定证据
矛盾：(1) ac11571 源 README "Performance Data" 表本就含 atto 行
（73.25%/69.75%/1.96ms/732+）；(2) 归档不可变基准
`platforms/x5/docs/release/benchmarks.yaml` 条目 `convnext-atto-x5`（源 ref
`1e1c64d`）记录相同 atto 数值。本包恢复 atto 行并把注释改为说明四行同源。
这是对先前 B3 披露（"基准表不含 atto 如实披露"）的事实性修正：修正的是对
发布范围的错误描述，不新增任何本仓库实测。若 Codex 认定 B3 评审另有依据，
可回退此行，其余内容不受影响。

## 严格保留项核验

- 快速启动/默认模型/精确资产 id：全部命令块未动。diff 的删除行共 7 行
  （`grep '^-'` 排除文件头），全部为文字改写而非命令：
  convnext 中英"不含 atto"注释各 2 行（本包修正的对象）、1 个空行、以及
  hgnetv2 中英各 1 行概述句（原句保留为改写后长句的开头，仅扩写）。见
  [readme-depth-classifiers.patch](evidence/2026-09-28-readme-depth-classifiers-remediation/readme-depth-classifiers.patch)。
- 接口 API、支持矩阵、真实板测记录及其适用范围：未改写。B3 四个 sample 的
  "board smoke pending / not-run" 说明、B4 批次的 supported-not-run 三态矩阵
  均原样保留。
- 新增文字中引用的接口事实（`--img-save-path`、`result.jpg` 副作用移除、
  平局按 ID 升序）全部来自各 README 既有正文，未引入新接口声明。

## 检查与结果（主机静态；无板端/量化执行）

- 契约检查器（`tools/sample_contract/check.py --sample`，venv python）：
  编辑前后各跑一轮，12 个 sample 均为 **0 violations / 1 policy skip
  （R-STAGE-PURITY，CLI 层既定豁免类别）/ 0 exemptions**；after 输出留档
  [checks-after.log](evidence/2026-09-28-readme-depth-classifiers-remediation/checks-after.log)。
- 本地链接：检查器 R-README-LINKS 对 24 个 README 的全部引用通过；其中 48 个
  图片引用（含 fasternet `FLOPs%20of%20Nets.png` 的 URL 编码路径，检查器
  unquote 后解析为 `FLOPs of Nets.png`）逐条用检查器同款 LINK_RE 复核存在。
- sample 主机测试：12 个套件全部通过，共 194 项（convnext 28、edgenext 26、
  fasternet 28、fastvit 28、googlenet 10、hgnetv2 16、mobileone 10、repghost 8、
  repvgg 10、repvit 10、resnext 10、vargconvnet 10）；其中多个套件含解析
  README 命令行的用例，验证命令块未被破坏。
- 人工核对：每张恢复图片均实际打开查看后撰写说明（非文件名/字数.literal
  测试替代）；fasternet 截图文字（rank 1 class 97 drake）与
  vargconvnet 截图分数（rank 1 class 37, 0.8582）经放大裁剪逐字核对。
- 双语一致：每个 sample 的中英新增段落结构、图、说明逐条镜像；锚点 ID 集合
  未增删（新内容并入既有章节，未新增章节 ID，符合契约 §3）。

## 边界与状态

- 未运行：板卡/SSH、模型下载、导出/校准/OE/HMCT/量化验证（按用户 2026-09-28
  裁定不作为交付阻塞）；未读写 `~/.claude` 记忆；未用子 agent；未执行任何
  git commit/push/merge/reset/stash/checkout。
- 本报告是作者自证，**不构成独立通过声明**；DOC-DEPTH-R1 整体、H1/H2/H9
  保持 open，待 Codex 独立评审及其余分组（efficient*、mobilenet*、resnet、
  ultralytics_yolo、fcos、bytetrack 等）的整改包完成后统一裁量。
- 图片说明中的"部署制品为 INT8 量化变体"等表述继承自各 README 既有支持矩阵
  与 manifest 行，未新增验证声明。

## 整改轮次（2026-09-28，Codex 独立评审 2 项）

首轮交付后 Codex 复核发现 2 项失实，本包已整改；完整对照见
[codex-r1-r2-response.md](evidence/2026-09-28-readme-depth-classifiers-remediation/codex-r1-r2-response.md)。
改动仅限 fastvit 与 repvit 各自的根双语 README（4 个文件），命令块、默认
变体、支持矩阵、性能表未动。

- **R1 FastViT**：原 overview 把"stage 1–3 RepMixer、stage 4 自注意力"
  写成全家族一致。经拉取官方 `apple/ml-fastvit` 的 `models/fastvit.py`
  核对：t8/t12/s12 四个 stage 全部 `"repmixer"`，仅 sa12 为
  前三 `"repmixer"` + 末位 `"attention"`（stage 4 另带 RepCPE 位置编码）。
  已改中英 overview 正文与论文 Fig. 2 图注，明确论文图展示的是 sa12 型
  attention 配置，不代表已发布 t8/t12/s12；未声明量化制品经过实测。
- **R2 RepViT**：原 Fig. 3 图注把 "RepViTBlock (3×3DW stride-2, 1×1,
  FFN)" 误并成下采样模块；特性 bullet "placed in separate blocks" 暗示
  两个独立网络块。经放大重看 `test_data/RepViT_architecture.png`：橙色
  下采样单元是 RepViTBlock 后接 stride-2 3×3DW、1×1、FFN（分辨率减半、
  C_i→C_i+1）；stage 内黄色 block 为 3×3DW + 并行 1×1DW 残差分支后接
  FFN；绿色 SEBlock 在 token 混合器与 FFN 间加 SE。图注已按五部分逐块
  重写（中英），bullet 改为"同一 block 内解耦 token/通道混合部分，
  非两个独立网络块"。Figure 4（RepViT_DW.png）图注与图相符，未改。
- **验证**：fastvit/repvit checker 均 0 violations / 1 policy skip /
  0 exemptions；两 sample 测试套件 28 OK / 10 OK
  （[checks-after-r1r2.log](evidence/2026-09-28-readme-depth-classifiers-remediation/checks-after-r1r2.log)）；
  全量 12-sample patch 已刷新
  （[readme-depth-classifiers.patch](evidence/2026-09-28-readme-depth-classifiers-remediation/readme-depth-classifiers.patch)）。
- ConvNeXt atto 基准行恢复有固定源依据（ac11571 表 +
  归档 benchmarks.yaml `convnext-atto-x5`），按用户指示保留原样。
- 本轮不自行关闭任何 finding；R1/R2 整改待 Codex 独立复核。

### DOC-CLASS-R3（convnext evaluator 同步恢复 atto 行）

首轮交付只改了根 README，convnext `evaluator/` 中英仍保留"已发布表不含
atto / no benchmark row"的错误声明，与根 README 恢复的 atto 行相互矛盾。
经授权（仅 evaluator 两文件 + 本报告/证据），已核查
`git show ac11571:samples/vision/convnext/README.md`（Performance Data 表
含 atto：73.25%/69.75%/1.96ms/732+）与
`platforms/x5/docs/release/benchmarks.yaml` 条目 `convnext-atto-x5`（同值，
源 ref `1e1c64d`）后同步修复：两语表格恢复 atto 行，删除错误缺失声明，
改为四行同源 + 归档快照佐证；EN 顺带修正陈旧句中的 72.50%（源表任何行均
无此值，femto 量化值为 72.25%），中文版补齐对应双语说明。"not
re-measured / not-run" 范围声明与命令块（含 atto asset-id）原样保留。
convnext checker 0 violations / 1 policy skip、测试 28 OK；diff 仅
evaluator 两文件 +20/−9。完整对照见
[codex-r3-convnext-evaluator-response.md](evidence/2026-09-28-readme-depth-classifiers-remediation/codex-r3-convnext-evaluator-response.md)、
[checks-after-r3.log](evidence/2026-09-28-readme-depth-classifiers-remediation/checks-after-r3.log)。
R3 不自行关闭，待 Codex 复核。
