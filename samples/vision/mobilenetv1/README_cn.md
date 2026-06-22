[English](./README.md) | 简体中文

# MobileNetV1 模型说明

MobileNetV1 是基于深度可分离卷积的轻量级 ImageNet 分类模型。本 sample
提供标准化 Python 推理入口，使用 `hbm_runtime` 运行 HBM 模型，模型输入为
NV12 的 Y/UV 双输入。

## 算法介绍

MobileNetV1 是一种面向嵌入式和移动端设备的轻量级卷积神经网络，用于高效图像分类。

- **论文**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- **参考实现**: [tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

### 算法功能

- ImageNet 1000 类图像分类
- Top-K 类别 ID 与置信度输出

### 算法特点

- **深度可分离卷积**：将标准卷积分解为 depthwise 卷积和 pointwise 卷积。
- **轻量级设计**：减少参数量和计算量，适合边缘设备部署。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
.
|-- conversion/             # 模型转换说明
|-- evaluator/              # 精度与结果验证说明
|-- model/                  # HBM 下载脚本与模型说明
|-- runtime/
|   `-- python/             # Python 推理入口与模型封装
|-- test_data/              # 测试图片与 ImageNet 标签
|-- README.md
`-- README_cn.md
```

## 快速体验

```bash
cd runtime/python
bash run.sh
```

脚本会将已发布的 S100 HBM 模型下载到 `model/s100/`，并使用
`test_data/zebra_cls.jpg` 执行分类推理。

直接运行入口：

```bash
cd runtime/python
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 任务 | 输入 | 类别数 | 已发布 HBM |
| --- | --- | --- | --- | --- |
| MobileNetV1 | 图像分类 | 224x224 NV12 (Y + UV) | ImageNet 1000 | S100 |

本 sample 使用公开 S100 HBM 模型，并下载到 sample 内 `model/s100/` 目录。

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `zebra_cls.jpg` 时，正确结果应在 Top-5 中包含 `zebra`，且分数分布为
合理的非零值。

## License

遵循 Model Zoo 顶层 License。
