# ImageNet 数据集说明

## 简介
ImageNet 是一个计算机视觉识别项目，是目前世界上图像识别最大的数据库。本目录提供了用于 ImageNet 模型验证的示例数据和标签文件。

## 算法介绍 (Algorithm Overview)
图像分类（Image Classification）是计算机视觉中的核心任务，旨在为输入图像分配一个或多个类别标签。
- 任务参考: [Image Classification on ImageNet](https://paperswithcode.com/task/image-classification)
- 相关模型: EfficientNet, ResNet, MobileNet 等。

## 目录结构
```bash
.
|-- imagenet_classes.names             # 包含 1000 类的索引与标签映射关系
`-- README.md                          # 本说明文档
```

## 使用说明
该数据集主要用于分类模型的推理示例和评估。用户可以通过加载 `imagenet_classes.names` 来获取模型输出索引对应的语义类别。

## 官方链接
- 官网: [image-net.org](https://www.image-net.org/)
