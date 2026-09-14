[English](./README.md) | 简体中文

# MobileNetV3 模型说明

MobileNetV3 是结合神经架构搜索、NetAdapt、SE 模块和 hard-swish 激活函数的
轻量级 ImageNet 分类模型。本 sample 提供标准化 Python 推理入口，模型输入为
NV12 的 Y/UV 双输入。

## 算法介绍

MobileNetV3 是面向移动端和嵌入式设备优化的轻量级 CNN，在 MobileNetV2 基础上引入 NAS 搜索、SE 模块和 hard-swish 激活。

- **论文**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **参考实现**: [torchvision MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html)

### 算法功能

- ImageNet 1000 类图像分类
- Top-K 类别 ID 与置信度输出

### 算法特点

- **NAS 搜索结构**：结合自动搜索和人工优化得到轻量网络。
- **SE 与 hard-swish**：改善移动端模型的表达能力和推理效率。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
.
|-- conversion/             # 原版转换 YAML 和辅助脚本
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

脚本会按当前板卡的 SOC 下载 S100 或 S600 HBM 模型到 `model/<soc>/`，并使用
`test_data/zebra_cls.jpg` 执行分类推理。

直接运行入口（把 `<soc>` 替换为 `s100` 或 `s600`）：

```bash
cd runtime/python
python3 main.py \
  --model-path ../../model/<soc>/mobilenetv3_224x224_nv12.hbm \
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
| MobileNetV3-Large | 图像分类 | 224x224 NV12 (Y + UV) | ImageNet 1000 | S100 / S600 |

本 sample 根据当前板卡 SOC 自动选择对应的公开 HBM 模型，并下载到 sample 内 `model/<soc>/` 目录。

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `zebra_cls.jpg` 时，正确结果应在 Top-5 中包含 `zebra`，且分数分布为
合理的非零值。

## License

遵循 Model Zoo 顶层 License。
