[English](./README.md) | 简体中文

# EfficientNet-Lite 模型说明

EfficientNet-Lite 是面向边缘高效推理的图像分类模型系列。本示例使用 NV12 HBM 模型运行 EfficientNet-Lite ImageNet 分类，并打印 Top-K 结果。

![EfficientNet 网络结构](./test_data/efficientnet_architecture.png)

## 算法介绍

EfficientNet 通过复合缩放同时调整网络深度、宽度和分辨率。EfficientNet-Lite 是面向边缘推理优化的变体系列。

- **论文**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **参考实现**: [tensorflow/tpu EfficientNet-Lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)

### 算法功能

- ImageNet 1000 类图像分类
- Lite0 到 Lite4 多尺度模型推理
- Top-K 类别 ID 与置信度输出

### 算法特点

- **复合缩放**：平衡网络深度、宽度和输入分辨率。
- **边缘优化**：Lite 系列更适合移动和嵌入式推理。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
efficientnet/
|-- conversion/
|   |-- README.md
|   |-- README_cn.md
|   |-- efficientnet_lite*_config.yaml
|   |-- get_efficientnet_lite*_onnx.py
|   |-- get_calibration_data.py
|   |-- timm2onnx_local.py
|   `-- x86_inference.py
|-- evaluator/
|   |-- README.md
|   `-- README_cn.md
|-- model/
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime/
|   `-- python/
|       |-- README.md
|       |-- README_cn.md
|       |-- efficientnet.py
|       |-- main.py
|       `-- run.sh
|-- test_data/
|-- README.md
`-- README_cn.md
```

## 快速体验

下载默认模型。`download_model.sh` 会把模型写入板端共享模型目录
`/opt/hobot/model/<soc>/basic/`：

```bash
cd samples/vision/efficientnet/model
bash download_model.sh s100 lite0
```

S600 设备：

```bash
bash download_model.sh s600 lite0
```

运行 Python 示例：

```bash
cd ../runtime/python
bash run.sh
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path /opt/hobot/model/s100/basic/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

S600 设备：

```bash
python3 main.py \
  --model-path /opt/hobot/model/s600/basic/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 变体 | 输入 | 运行时输入类型 | 下载路径 |
| --- | --- | --- | --- |
| EfficientNet-Lite0 | 224x224 | NV12 Y/UV planes | `/opt/hobot/model/<soc>/basic/` |
| EfficientNet-Lite1 | 240x240 | NV12 Y/UV planes | `/opt/hobot/model/<soc>/basic/` |
| EfficientNet-Lite2 | 260x260 | NV12 Y/UV planes | `/opt/hobot/model/<soc>/basic/` |
| EfficientNet-Lite3 | 300x300 | NV12 Y/UV planes | `/opt/hobot/model/<soc>/basic/` |
| EfficientNet-Lite4 | 380x380 | NV12 Y/UV planes | `/opt/hobot/model/<soc>/basic/` |

本示例使用公开 S100 / S600 HBM 模型，模型下载到 `/opt/hobot/model/<soc>/basic/`（`<soc>` 取 `s100` 或 `s600`）。`run.sh` 会自动检测当前设备 SoC 并下载对应版本。

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `Scottish_deerhound.JPEG` 时，正确结果应在 Top-5 中包含与猎鹿犬相关的类别，且分数分布为合理的非零值。

## License

遵循 Model Zoo 顶层 License。
