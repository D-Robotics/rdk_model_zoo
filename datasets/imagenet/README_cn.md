[English](README.md) | 简体中文

# ImageNet 数据集资源

**ImageNet ILSVRC-2012（ImageNet-1k）** 是图像分类数据集：1,000 个类别，约 128
万张训练图和 5 万张验证图。分类 sample 使用该类别顺序。本目录随附标签文件和一张
示例图片；数据集需按官方条款另行获取（验证集需要注册）。

<a id="files"></a>
## 随仓文件

```text
imagenet/
├── README.md                    # 英文指南
├── README_cn.md                 # 本指南
├── imagenet_classes.names       # 1000 类标签文件（字典字面量格式）
└── asset/
    └── zebra_cls.jpg            # 一张分类示例图片
```

### imagenet_classes.names

按文件实测：这是一个覆盖全部 1000 类的 **Python 字典字面量**，键为模型输出类别
索引 0–999，按标准 ILSVRC-2012 顺序排列（索引 0 = `tench, Tinca tinca`，
索引 999 = `toilet tissue, toilet paper, bathroom tissue`）；值是逗号分隔的多个
同义英文名称；文件末尾没有换行符。摘录：

```python
{0: 'tench, Tinca tinca',
 1: 'goldfish, Carassius auratus',
 ...
 999: 'toilet tissue, toilet paper, bathroom tissue'}
```

它**不是**每行一个名称的列表，尽管运行时加载器也兼容那种格式。
`utils/py_utils/labels.py::load_labels`（各分类 sample 经此复用；旧版
`utils/py_utils/file_io.py::load_labels` 行为一致）检测开头的 `{` 并用
`ast.literal_eval` 解析为 `{索引: 名称}`；每行一个的显示名文本同样可用。每个值是
逗号分隔的多个同义英文短语——**仅作显示名**：它们不是 WordNet synset ID
（`n########`），也不能替代 synset ID。

键是**类别索引，不是 synset ID**。ImageNet-1k 没有 COCO 那样的稀疏 category-ID
空间；索引顺序和命名由本文件固定，精度比较必须基于同一文件。注意 `--label-file`
在不同使用方处含义不同：见下表两行消费者说明。

### asset/zebra_cls.jpg

随仓的斑马示例图片对应 ImageNet 类别索引 340（`zebra`）。它与
`samples/vision/resnet/test_data/zebra_cls.jpg` 和
`samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg` 逐字节一致。各 sample
指南说明支持的模型输入和标签文件用法。

<a id="usage"></a>
## 这些资源被谁使用

| 使用方 | 用途 |
| --- | --- |
| 全部分类 sample（如 [ResNet](../../samples/vision/resnet/README_cn.md)、[EfficientFormer](../../samples/vision/efficientformer/README_cn.md)、[RepViT](../../samples/vision/repvit/README_cn.md)） | 这里的 `--label-file` 指**运行时显示名**：本字典文件（或每行一个的显示名文本）经共享加载器解析，用于输出 Top-1/Top-5 类名（相对仓库根目录的路径） |
| [Ultralytics YOLO 分类评估器](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md) | 它的 `--label-file` 含义不同：按模型类别顺序排列的 `n########` synset ID 列表，与图片文件名中的标识匹配。本显示名文件**不能**传给它。真值来自 `--val-txt`（named 格式 `<相对图像路径> <从0开始的类别索引>`，或 ordered 变体）——确切约定见评估器的数据集章节 |

sample 指南中的命令均以**仓库根目录**为工作目录，标签路径即
`datasets/imagenet/imagenet_classes.names`。换目录执行时请传入修正后的相对或
绝对路径。

<a id="acquisition"></a>
## 获取数据集

本目录**没有下载脚本**（与 [COCO](../coco/README_cn.md) 不同）。验证集需自行
准备，例如：

```text
datasets/imagenet/val_images/   # .gitignore 已排除该路径
```

`.gitignore` 已排除 `datasets/imagenet/val_images/*`；数据集图片和真值列表
不得提交。Ultralytics 分类评估器在其[数据集章节](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md#dataset)
说明所需的图片/标签列表布局。

官方来源——请在各自条款下获取和使用：

- ImageNet：<https://image-net.org/>（下载需注册）
- Hugging Face 镜像：<https://huggingface.co/datasets/ILSVRC/imagenet-1k>
  （gated 数据集）

<a id="provenance"></a>
## 来源

数据来源：[ImageNet](https://image-net.org/)。`imagenet_classes.names` 将模型输出索引映射为显示名称。数据集计分还需每张图片对应的真实类别索引，顺序须与模型的 0–999 类别一致。
