[English](README.md) | 简体中文

# EfficientSAM-Tiny

EfficientSAM-Tiny 通过一个框提示对任意目标分割,使用蒸馏的 ViT-Tiny 图像**编码器**与固定提示掩码**解码器**,分别编译为独立 `.hbm`,适配 RDK-S 系列(S100/S100P/S600),板端用 `hbm_runtime` 推理。

## 算法概述

EfficientSAM 将 Segment Anything 蒸馏到 ViT-Tiny 主干。编码器将归一化 RGB 512×512 图像映射为 256×32×32 嵌入;解码器接收该嵌入(框提示已烤进解码器 ONNX),预测低分辨率掩码与 IoU。运行时将选中掩码上采样到 512×512 并叠加。

- 论文:<https://arxiv.org/abs/2312.00863>
- 项目网站:<https://yformer.github.io/efficient-sam/>
- 官方仓库:<https://github.com/yformer/EfficientSAM>

## 能力

- 单框提示 → 二值目标掩码 + IoU(提示烤进解码器 ONNX)。

## 平台兼容

| 板卡 | SoC | march | 模型目录 |
|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/` |
| S100P | s100p | nash-m | `model/nash-m/` |
| S600 | s600 | nash-p | `model/nash-p/` |

## 目录结构

```
efficient_sam/
├── conversion/          # ONNX 导出 + hb_compile 量化
│   ├── configs/         # 每个 march 一份 committed YAML(编码器 + 解码器)
│   └── scripts/         # quantize.py、export/prepare_*.py
├── evaluator/           # 板端数值评估说明
├── model/               # download_model.sh + 各 march .hbm
├── runtime/python/      # hbm_runtime 推理:main.py、efficient_sam.py、run.sh
└── test_data/           # dogs.jpg + 期望二值掩码
```

## 快速开始

在板端:

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
# -> 生成 test_data/efficient_sam_full_mask_result.jpg + efficient_sam_binary_mask_result.png
```

## 转换

见 [`conversion/README_cn.md`](./conversion/README_cn.md),包含 ONNX 导出、量化配置与 OE 工具链入口。

## 运行时

见 [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)。

## 评估

板端延迟测量见 [`evaluator/README_cn.md`](./evaluator/README_cn.md)。

## License

本 sample 遵循 RDK Model Zoo 许可。上游 EfficientSAM 权重与 ONNX assets 保留其原始许可。
