[English](./README.md) | 简体中文

# EdgeNeXt 模型说明

本目录给出 EdgeNeXt sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法介绍

EdgeNeXt 是面向移动视觉应用设计的高效 CNN-Transformer 混合架构。网络采用四阶段金字塔结构，并结合卷积编码器和 SDTA 编码器，以平衡分类精度、模型规模和推理速度。

- **论文**: [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)
- **参考实现**: [mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)

### 算法功能

EdgeNeXt 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **CNN-Transformer 混合设计**：结合 CNN 推理效率和 Transformer 风格的全局特征建模能力。
- **四阶段金字塔结构**：使用部署友好的层级特征提取结构。
- **SDTA 编码器**：通过通道分组和注意力机制编码多尺度特征。
- **高效部署**：提供 base、small、x-small、xx-small 四个 RDK X5 部署模型，使用 packed NV12 输入。

![EdgeNeXt 结构](./test_data/EdgeNeXt_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- EdgeNeXt_base_config.yaml
|   |-- EdgeNeXt_small_config.yaml
|   |-- EdgeNeXt_x_small_config.yaml
|   |-- EdgeNeXt_xx_small_config.yaml
|   |-- README.md
|   `-- README_cn.md
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
|       |-- edgenext.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- EdgeNeXt_architecture.png
|   |-- ImageNet_1k.json
|   |-- Zebra.jpg
|   `-- inference.png
|-- README.md
`-- README_cn.md
```

## 快速体验

### Python

- Python 详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验命令：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 预编译 `.bin` 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评估

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表为 `RDK X5` 上发布的 EdgeNeXt 性能数据。

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EdgeNeXt-base | 224x224 | 1000 | 18.51 | 78.21% | 74.52% | 8.80 | 113.35 |
| EdgeNeXt-small | 224x224 | 1000 | 5.59 | 76.50% | 71.75% | 4.41 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 1000 | 2.34 | 71.75% | 66.25% | 2.88 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1000 | 1.33 | 69.50% | 64.25% | 2.47 | 403.49 |

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
