# R1/R2 整改对照（Codex 独立发现 → 作者整改）

整改时间：2026-09-28。范围仅 4 个文件：
`samples/vision/fastvit/README.md`、`README_cn.md`、
`samples/vision/repvit/README.md`、`README_cn.md`。
命令块、默认变体、支持矩阵、性能表均未改动。

## R1 FastViT stage-4 attention 以偏概全

**核查**：拉取官方
`https://raw.githubusercontent.com/apple/ml-fastvit/main/models/fastvit.py`
（Apple 官方仓库 main 分支）逐 variant 核对 `token_mixers` 定义：

- `fastvit_t8` / `fastvit_t12` / `fastvit_s12`：
  `token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")` —
  四个 stage 全部 RepMixer，无 attention。
- `fastvit_sa12`：`token_mixers = ("repmixer", "repmixer", "repmixer", "attention")`，
  且 `pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7), ...)]`
  — stage 1–3 RepMixer，仅 stage 4 attention（带 RepCPE 位置编码）。

Codex 判断属实：X5 发布的四个变体中只有 sa12 有 stage-4 attention；
原 overview"stages 1–3 use RepMixer, stage 4 self-attention"对
t8/t12/s12 不成立。

**整改**（overview 正文 + Fig. 2 图注，中英同步）：

- 正文改前（EN）：`The model combines convolutional stages and attention
  (stages 1–3 use RepMixer token mixing, stage 4 self-attention) ...`
- 正文改后（EN）：`The token mixers differ across the family: in the
  upstream model definitions, the published t8/t12/s12 variants use
  RepMixer token mixing in all four stages, while sa12 keeps RepMixer in
  stages 1–3 and uses self-attention (with a RepCPE positional encoding)
  only in stage 4 — the conv/attention hybrid the source README
  summarizes.`
- 图注增加：`The paper draws stage 4 with a self-attention token mixer —
  that is the configuration sa12 uses ...; it does not represent the
  published t8/t12/s12, whose upstream definitions use RepMixer in all
  four stages (see models/fastvit.py in apple/ml-fastvit).`
- 中文两处同步改写。源 README 的特性 bullet（Hybrid architecture）按源
  保留，由新增正文限定其适用面。
- 未声明量化制品经过任何实测；默认变体与支持矩阵未动。

## R2 RepViT Figure 3 图注混淆下采样单元 + "separate blocks"误述

**看图复核**（对 `test_data/RepViT_architecture.png` 放大裁剪逐块查看）：

- stem（粉色）：3×3 Stride=2 → Act → 3×3 Stride=2 两层堆叠。
- stage 间橙色下采样单元（如 stage1→stage2）：输入 B×C1×H/4×W/4 →
  **RepViTBlock**（分辨率不变）→ 3×3DW Stride=2（H/4→H/8）→ 1×1
  （C1→C2）→ FFN → 输出 B×C2×H/8×W/8。即下采样单元 = RepViTBlock 后接
  stride-2 3×3DW、1×1、FFN。
- stage 内黄色 RepViTBlock：3×3DW 与并行 1×1DW 分支经残差 ⊕ 相加（token
  混合部分），后接 FFN（通道混合部分，自带残差）。
- 绿色 RepViTSEBlock：同上结构，token 混合器与 FFN 之间多一个 SE。
- 底部：训练期 3×3DW + 1×1DW 并行分支 ⊕ →（Inference）→ 单个 3×3DW。

原图注把 "RepViTBlock (3×3DW stride-2, 1×1, FFN)" 混写成下采样模块；
特性 bullet "placed in separate blocks" 暗示 token/channel 混合器位于
两个独立网络块，均不准确。

**整改**（Figure 3 图注重写 + bullet 修正，中英同步）：

- bullet 改后（EN）：`inside a single block, the token-mixer part (3×3
  depthwise convolution, with SE in the SE variant) and the channel-mixer
  part (the 1×1 FFN) are decoupled and stacked one after the other,
  replacing the MobileNetV3 layout where both sit inside the
  inverted-bottleneck block — not two separate network blocks.`
- 图注改后按上述五点逐块描述（stem / 黄 RepViTBlock / 橙下采样单元 /
  绿 SEBlock / 推理期分支融合），中英镜像。
- Figure 4（RepViT_DW.png）图注本就与图相符，未改。

## 验证

- `tools/sample_contract/check.py --sample`：fastvit、repvit 均
  0 violations / 1 policy skip / 0 exemptions
  （[checks-after-r1r2.log](checks-after-r1r2.log)）。
- 两个 sample 测试套件：fastvit 28 OK、repvit 10 OK（README 命令解析用例
  未受影响；命令块未改）。
- 全量 12-sample patch 已刷新：
  [readme-depth-classifiers.patch](readme-depth-classifiers.patch)。
- 作者报告已追加"整改轮次"章节。
