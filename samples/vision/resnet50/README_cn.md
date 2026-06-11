[English](./README.md) | 简体中文

# ResNet50 模型说明

ResNet50 是 RDK S100 Model Zoo 的 ImageNet 分类 sample。本目录提供 sample
内模型下载、Python runtime、保留的原始文档资产以及评测说明。

## 算法介绍

ResNet 通过 shortcut connection 进行残差学习，降低深层卷积网络的优化难度。
ResNet50 使用包含 `1x1`、`3x3`、`1x1` 卷积的 bottleneck 残差块，在控制计算量
的同时构建更深的网络。

![ResNet architecture](./test_data/resnet_architecture.png)

资源：

- 论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch 实现：[torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- TorchVision ResNet50 模型：[torchvision ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

### 算法功能

- ImageNet 1000 类图像分类
- Top-K 类别 ID 与置信度输出

### 算法特点

- **残差连接**：使用 shortcut connection 降低深层网络优化难度。
- **瓶颈结构**：使用 bottleneck block 提升特征表达效率。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   `-- README_cn.md
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
|       |-- resnet50.py
|       `-- run.sh
`-- test_data
    |-- resnet_architecture.png
    |-- resnet_architecture2.png
    |-- result.png
    `-- zebra_cls.jpg
```

## 快速体验

```bash
cd runtime/python
bash run.sh
```

Python 直接入口：

```bash
python3 main.py \
  --model-path ../../model/s100/resnet50_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 输入 | 运行模型 |
| --- | --- | --- |
| ResNet50 | 224x224 NV12 | `model/s100/resnet50_224x224_nv12.hbm` |

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

随附的测试图像：

![测试图像](./test_data/zebra_cls.jpg)

预期的 Top-5 分类输出：

```text
Top-5 Classification Results:
  [0] zebra: ...
```

分类结果可视化图：

![推理结果图](./test_data/result.png)

## License

遵循 Model Zoo 顶层 License。
