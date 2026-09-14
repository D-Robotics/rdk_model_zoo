[English](./README.md) | 简体中文

# ResNet152 模型说明

ResNet152 是用于图像分类的深层残差卷积网络。本示例使用 NV12 HBM 模型运行 ResNet152 ImageNet 分类器，并打印 Top-K 分类结果。

![ResNet 网络结构](./test_data/resnet_architecture.png)

## 算法介绍

ResNet 通过残差连接缓解深层网络训练退化问题。ResNet152 是 152 层深度残差网络，适用于 ImageNet 图像分类。

- **论文**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **参考实现**: [torchvision ResNet](https://pytorch.org/vision/main/models/resnet.html)

### 算法功能

- ImageNet 1000 类图像分类
- Top-K 类别 ID 与置信度输出

### 算法特点

- **残差连接**：使用 shortcut connection 降低深层网络优化难度。
- **深层特征提取**：152 层结构提升模型表达能力。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
resnet152/
|-- conversion/
|   |-- README.md
|   |-- README_cn.md
|   |-- get_calibration_data.py
|   |-- resnet152_config.yaml
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
|       |-- main.py
|       |-- resnet152.py
|       `-- run.sh
|-- test_data/
|   |-- resnet_architecture.png
|   |-- resnet_architecture2.png
|   |-- result.png
|   `-- zebra_cls.jpg
|-- README.md
`-- README_cn.md
```

## 快速体验

下载模型到当前 sample 的 `model` 目录（通过参数指定 `s100` 或 `s600`）：

```bash
cd model
bash download_model.sh s100   # 或：bash download_model.sh s600
```

运行 Python 示例：

```bash
cd ../runtime/python
bash run.sh
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path ../../model/s100/resnet152_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 输入 | 运行时输入类型 | 输出 | 下载路径 |
| --- | --- | --- | --- | --- |
| ResNet152 | 224x224 | NV12 Y/UV planes | ImageNet 1000 类 logits | `model/s100/resnet152_224x224_nv12.hbm`（S100）<br/>`model/s600/resnet152_224x224_nv12.hbm`（S600）|

本示例使用公开 RDK ResNet152 HBM 模型，模型下载到当前 sample 内的 `model/<soc>` 目录。

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
