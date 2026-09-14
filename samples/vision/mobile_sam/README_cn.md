[English](README.md) | 简体中文

# MobileSAM

MobileSAM 通过一个框提示对图像中的任意目标进行分割。本 sample 将 TinyViT 图像**编码器**与框提示掩码**解码器**分别编译为独立的 `.hbm` 模型,适配 RDK-S 系列(S100 / S100P / S600),在板端用 `hbm_runtime` 完成整图掩码推理。

## 算法概述

MobileSAM 将 Segment Anything 蒸馏到轻量 TinyViT 主干,使编码器可在边缘 BPU 实时运行。编码器将归一化的 512×512 图像映射为 256×32×32 嵌入;解码器接收该嵌入与一个框提示 `(x1,y1,x2,y2)`,预测低分辨率掩码与 IoU。运行时将选中掩码上采样回 512×512 并叠加。

- 论文:<https://arxiv.org/abs/2306.14289>
- 官方仓库:<https://github.com/ChaoningZhang/MobileSAM>

## 能力

- 单框提示 → 二值目标掩码 + IoU。

## 平台兼容

| 板卡 | SoC | march | 模型目录 |
|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/` |
| S100P | s100p | nash-m | `model/nash-m/` |
| S600 | s600 | nash-p | `model/nash-p/` |

运行时从 `/sys/class/boardinfo/` 自动探测板卡;可用 `download_model.sh <march>` 或 `quantize.py --march <march>` 显式指定。

## 目录结构

```
mobile_sam/
├── conversion/          # ONNX 导出 + hb_compile 量化
│   ├── configs/         # 每个 march 一份 committed YAML(编码器 + 解码器)
│   └── scripts/         # quantize.py、export/prepare_*.py
├── evaluator/           # 板端数值评估说明
├── model/               # download_model.sh + 各 march .hbm
├── runtime/python/      # hbm_runtime 推理:main.py、mobile_sam.py、run.sh
└── test_data/           # dogs.jpg + 期望二值掩码
```

## 快速开始

在板端:

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
# -> 生成 test_data/mobile_sam_full_mask_result.jpg + mobile_sam_binary_mask_result.png
```

`run.sh` 自动探测板卡,缺 `.hbm` 时下载对应 march 的模型对,再执行 `python3 main.py`。

## 转换

见 [`conversion/README_cn.md`](./conversion/README_cn.md),包含 ONNX 导出、量化配置与 OE 工具链入口。

## 运行时

见 [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)。入口 `main.py` 解析板卡,加载两个 `.hbm`,先跑编码器再跑解码器,保存叠加图与二值掩码。

## 评估

见 [`evaluator/README_cn.md`](./evaluator/README_cn.md) 获取板端延迟测量。

## License

本 sample 遵循 RDK Model Zoo 许可。上游 MobileSAM 权重与 ONNX assets 保留其原始许可。
