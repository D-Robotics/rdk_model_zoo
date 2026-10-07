# ViT CIFAR-10 图像分类

<a id="overview"></a>
## 概述

ViT 把图像分块组织为序列，通过自注意力完成分类。本 sample 提供 S100 CIFAR-10 十分类的两个量化变体，不是 ImageNet 权重。

[Paper](https://arxiv.org/abs/2010.11929) · [Original ViT implementation](https://github.com/google-research/vision_transformer)

![ViT](test_data/readme_img/vitnet.png)

<a id="support-matrix"></a>
## 支持矩阵

| Target | Variant | Python | C++ |
| --- | --- | --- | --- |
| x5 | int8 | not-supported | not-supported |
| x5 | int16 | not-supported | not-supported |
| s100 | int8 | supported | not-supported |
| s100 | int16 | supported | not-supported |
| s100p | int8 | not-supported | not-supported |
| s100p | int16 | not-supported | not-supported |
| s600 | int8 | not-supported | not-supported |
| s600 | int16 | not-supported | not-supported |

使用支持矩阵所列的 S100 `int8` 或 `int16` 制品运行 Python 分类器。

<a id="prerequisites"></a>
## 前提

需要完整仓库。S100 推理需要板端镜像提供的 `hbm_runtime`；主机侧依赖
按下方 `requirements-host.txt` 安装。磁盘需容纳仓库、所选 HBM 和输出。
只有重新转换需要 OE 工具链。

```bash
# cwd: repository root
python3 -m venv .venv-vit
source .venv-vit/bin/activate
python3 -m pip install -r samples/vision/vit/requirements-host.txt
```

<a id="quickstart"></a>
## 快速开始

在 S100 上先显式准备模型再运行；成功时退出 0，打印五个类别 ID/分数/标签。推理不会下载模型。

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
python3 samples/vision/vit/runtime/python/main.py --target s100 --variant int8 --test-img samples/vision/vit/test_data/airplane_0000.png --label-file samples/vision/vit/test_data/cifar10_classes.names --top-k 5
```

<a id="expected-results"></a>
## 预期结果

使用随附 `airplane_0000.png` 时，Top-5 包含 airplane。默认 int8、
resize=0、Top-K=5。分数是十个 logits 的 softmax；完全平局按 ID 升序。
仅 `--img-save-path` 写标注图片。

<a id="performance"></a>
## 性能数据

已发布的 CIFAR-10 精度记录见[评估](evaluator/README_cn.md#reference-results)；


<a id="directory"></a>
## 目录

```text
model/          # HBM download and artifact references
runtime/python/ # CLI, binding, runner and shared classification API
conversion/     # 原始 PTQ YAML 与原始编译日志
evaluator/      # 评估说明与已发布精度记录
test_data/      # 10 张 CIFAR 图片、类别字典与原始插图
tests/          # 无 SDK 主机 unittest 套件
```

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python runtime](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

新集成使用 `ViTClassifier` 类（[classify.py](runtime/python/classify.py)；
共享 `ClassificationTask` 流程仍可从
[classification.py](runtime/python/classification.py) 导入）；
`--model-variant` 保留为 `--variant` 别名，本地 run.sh 接受位置参数
int8/int16。

<a id="license"></a>
## 许可

代码保留 Apache-2.0 声明。仓库 LICENSE 与上游实现/权重许可分别适用；下载发布不新增权重授权。
