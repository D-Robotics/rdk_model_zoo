[English](./README.md) | 简体中文

# Vision Transformer 模型说明

Vision Transformer（ViT）通过将图片切分为 patch，并使用 self-attention 对 patch token 建模，将 Transformer 架构用于图像分类。本示例使用 NV12 HBM 模型运行 CIFAR-10 ViT 分类器，并打印 Top-K 分类结果。

![ViT 网络结构](./test_data/readme_img/vitnet.png)

## 算法介绍

ViT 将图像划分为固定大小 patch，并将 patch 序列输入 Transformer Encoder 进行分类。

- **论文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **参考实现**: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## 算法功能

- CIFAR-10 图像分类
- int8 / int16 两个模型变体推理
- Top-K 类别 ID 与置信度输出

## 算法特点

- **Patch token**：将图像划分为 patch 序列。
- **Self-attention**：使用 Transformer 建模全局关系。
- **NV12 输入**：runtime 使用 Y/UV 双输入适配 HBM 模型。

## 目录结构

```text
vit/
|-- conversion/
|   |-- README.md
|   |-- README_cn.md
|   |-- config_vit_nv12.yaml
|   `-- hb_compile.log
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
|       |-- run.sh
|       `-- vit.py
|-- test_data/
|-- README.md
`-- README_cn.md
```

## 快速体验

下载默认模型到当前 sample 的 `model` 目录：

```bash
cd samples/vision/vit/model
bash download_model.sh s100 int8
```

运行 Python 示例：

```bash
cd ../runtime/python
bash run.sh int8
```

直接运行入口脚本：

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

## 模型转换

- 预编译 HBM 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护 Python 推理路径，详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

| 模型 | 数据集 | 输入 | 运行时输入类型 | 下载路径 |
| --- | --- | --- | --- | --- |
| ViT CIFAR-10 int8 | CIFAR-10 | 224x224 | NV12 Y/UV planes | `model/s100/vit_cifar10_batch1_int8.hbm` |
| ViT CIFAR-10 int16 | CIFAR-10 | 224x224 | NV12 Y/UV planes | `model/s100/vit_cifar10_batch1_int16.hbm` |

本示例使用公开 S100 HBM 模型，模型下载到当前 sample 内的 `model/s100` 目录。

## 模型评估

评测说明和结果检查方法请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

使用 `airplane_0000.png` 时，正确结果应在 Top-5 中包含 `airplane`，且分数分布为合理的非零值。

## License

遵循 Model Zoo 顶层 License。
