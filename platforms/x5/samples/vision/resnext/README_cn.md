[English](./README.md) | 简体中文

# ResNeXt 模型说明

本目录给出 ResNeXt sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法概览

ResNeXt 在残差网络的基础上引入 split-transform-merge 设计，通过增加 cardinality 而不只是单纯增加网络深度或宽度，提升模型表达能力。该结构保留了简洁的残差主干，并通过组卷积提高表示效率。

- **论文地址**: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- **参考实现**: [facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)

### 算法功能

ResNeXt 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **Cardinality**：通过增加并行变换路径的数量提升模型表达能力。
- **组卷积**：使用 grouped convolution 在精度和计算效率之间取得平衡。
- **残差主干**：保留稳定的 residual learning 结构，便于扩展 CNN 设计。
- **分类输出**：模型输出 ImageNet-1k 的 Top-K 类别 ID 及其置信度。

![ResNeXt 架构](./test_data/ResNeXt_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- README.md
|   |-- README_cn.md
|   `-- ResNeXt50_32x4d_config.yaml
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download.sh
|   |-- README.md
|   `-- README_cn.md
|-- runtime
|   `-- python
|       |-- main.py
|       |-- README.md
|       |-- README_cn.md
|       |-- resnext.py
|       `-- run.sh
|-- test_data
|   |-- bee_eater.JPEG
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- ResNeXt_architecture.png
|-- README.md
`-- README_cn.md
```

## 快速开始

### Python

- 运行说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 预编译 `.bin` 模型请参考 [model/README_cn.md](./model/README_cn.md)。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

当前 sample 的维护运行链路为 Python。

- Python 运行说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

评测说明、性能数据和验证结论见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表给出 ResNeXt 在 `RDK X5` 上的公开性能数据。

| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | 浮点 Top-1 | 量化 Top-1 | 时延(ms) | 帧率(FPS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNeXt50_32x4d | 224x224 | 1000 | 24.99 | 76.25% | 76.00% | 5.89 | 189.61 |

![推理结果](./test_data/inference.png)

## 许可

遵循 Model Zoo 顶层 License。
