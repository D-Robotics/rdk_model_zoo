[English](./README.md) | 简体中文

# PP-LiteSeg-STDC1 语义分割

PP-LiteSeg-STDC1 是 PaddleSeg 提供的轻量实时语义分割模型，已完成 PTQ 量化并部署至 RDK X5 BPU。

---

## 算法简介

PP-LiteSeg 是 PaddleSeg 提供的轻量实时语义分割模型。建议优先选择 `PP-LiteSeg-STDC1` 作为 RDK X5 新模型转换目标，因为它模型规模适中、分割效果直观，并且 CNN 结构相比 SAM 或大 Transformer 更适合先做 PTQ 量化验证。

- 论文：[PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)
- 官方实现：[PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

### 算法功能

- 输入单张 RGB 图像进行语义分割
- 输出逐像素类别预测，适用于道路场景或自定义分割数据集

### 算法特点

- 轻量实时分割网络，适合边缘部署验证
- CNN 结构为主，相比 prompt segmentation 或大型 Transformer 更易进行 PTQ
- 输出协议简单，对类别维度执行 `argmax` 得到分割图

---

## 目录结构

```bash
.
├── conversion
│   ├── onnx_export
│   │   └── export_pp_liteseg_stdc1_onnx.sh
│   ├── ptq_yamls
│   │   └── pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml
│   ├── prepare_calibration.py
│   ├── README.md
│   └── README_cn.md
├── evaluator
│   ├── README.md
│   └── README_cn.md
├── model
│   ├── download.sh
│   ├── README.md
│   └── README_cn.md
├── runtime
│   └── python
│       ├── main.py
│       ├── pp_liteseg.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data
│   └── street.jpg
├── README.md
└── README_cn.md
```

> 说明：当前示例仅提供 Python runtime 实现。

---

## 快速开始

在 RDK X5 板端执行以下命令：

```bash
# 1. 进入运行时目录
cd samples/vision/pp_liteseg/runtime/python

# 2. 一键运行（模型不存在时自动下载）
chmod +x run.sh
./run.sh

# 3. 或手动指定参数运行
python3 main.py \
    --model-path ../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
    --test-img ../../test_data/street.jpg \
    --output ../../test_data/result.jpg
```

参数说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

---

## 完整链路

本 sample 覆盖从模型转换到板端推理的完整链路：

```text
PaddleSeg 预训练模型 -> 导出推理模型 -> ONNX -> hb_mapper checker -> 校准数据 -> hb_mapper makertbin -> .bin -> BPU 推理
```

- **模型转换**：[conversion/README_cn.md](./conversion/README_cn.md)
- **下载预编译模型**：[model/download.sh](./model/download.sh)
- **板端推理**：[runtime/python/README_cn.md](./runtime/python/README_cn.md)
- **精度与性能验证**：[evaluator/README_cn.md](./evaluator/README_cn.md)

---

## 运行协议

生成的部署模型建议遵循以下协议：

- 模型：`PP-LiteSeg-STDC1`
- 输入分辨率：默认 `1024x512`
- 运行时输入类型：`nv12`
- 训练输入类型：`rgb`
- 训练输入布局：`NCHW`
- 归一化：通过 `hb_mapper` YAML 配置 ImageNet mean/std
- 输出：语义分割 logits，对类别维度执行 `argmax` 得到分割结果

---

## 注意事项

- 运行 `hb_mapper` 前请确保 ONNX 是静态输入 shape。
- 不要把 Python 后处理、resize 或 palette 渲染固化进 ONNX 图。
- 校准集使用有代表性的道路场景图像，建议从 20 到 50 张开始。
