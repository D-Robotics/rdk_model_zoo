[English](./README.md) | 简体中文

# MobileNetV4 模型说明

MobileNetV4 是 RDK Model Zoo 的 ImageNet 分类 sample，通过 SOC 自适应下载同时支持 S100 与 S600 平台。本目录提供基于
`hbm_runtime` 的标准 Python runtime、sample 内模型下载、保留的转换资产以及
验证说明。

## 算法介绍

MobileNetV4 引入 Universal Inverted Bottleneck (UIB) 搜索模块和面向移动端的
注意力设计，用于在移动和嵌入式加速器上实现高效图像分类。

资源：

- 论文：[MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
- timm 实现：[huggingface/pytorch-image-models MobileNetV4](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)

### 算法功能

- ImageNet 1000 类图像分类
- Small / Medium 两个模型变体推理
- Top-K 类别 ID 与置信度输出

### 算法特点

- **UIB 模块**：统一表达 inverted bottleneck、conv-next 风格和 FFN 风格结构。
- **移动端注意力设计**：面向移动和嵌入式推理优化精度与效率。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   |-- README_cn.md
|   |-- get_calibration_data.py
|   |-- get_mobilenetv4_onnx.py
|   |-- mobilenetv4_medium_config.yaml
|   |-- mobilenetv4_small_config.yaml
|   |-- timm2onnx_local.py
|   `-- x86_medium_inference.py
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime
|   `-- python
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- mobilenetv4.py
|       `-- run.sh
`-- test_data
    |-- imagenet_classes.names
    `-- zebra_cls.jpg
```

## 快速体验

```bash
cd runtime/python
bash run.sh
```

运行 medium 模型：

```bash
bash run.sh medium
```

直接入口（把 `<soc>` 替换为 `s100` 或 `s600`）：

```bash
python3 main.py \
  --model-variant small \
  --model-path ../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

随附测试图的预期结果：

```text
Top-5 Classification Results:
  [0] zebra: ...
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 版本 | 输入 | 运行模型 |
| --- | --- | --- |
| Small | 224x224 NV12 | `model/<soc>/mobilenetv4_small_224x224_nv12.hbm` |
| Medium | 256x256 NV12 | `model/<soc>/mobilenetv4_medium_256x256_nv12.hbm` |

`<soc>` 取 `s100` 或 `s600`，根据 `/sys/class/boardinfo/soc_name` 自动决定。

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `zebra_cls.jpg` 时，正确结果应在 Top-5 中包含 `zebra`，且分数分布为合理的非零值。

## License

遵循 Model Zoo 顶层 License。
