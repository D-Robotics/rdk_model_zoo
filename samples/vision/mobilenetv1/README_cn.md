# MobileNetV1 图像分类

MobileNetV1 在 RDK 板卡上的 ImageNet-1k 分类：输入一张 BGR 图像，输出稳定的 Top-K `(类别 ID, 分数, 标签)`。源模型：[tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)，论文 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)。[English](README.md)

<a id="overview"></a>

## 概述

本样例为全部目标提供同一个 Python 运行时。
`MobileNetV1Classifier` 类执行由 `predict` 串联的 `preprocess → infer → postprocess` 流程：按检测到的板卡从平台发布 Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，返回带类型的 Top-K 结果（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。

### 算法背景

MobileNetV1 面向嵌入式与移动端设备的高效图像分类。其效率来自深度可分离
卷积：将标准卷积分解为逐通道的 depthwise 滤波和整合各通道输出的 1×1
pointwise 投影（[论文](https://arxiv.org/abs/1704.04861)、
[tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)）。

特性摘要：

- **深度可分离卷积**：将标准卷积分解为 depthwise 卷积和 1×1 pointwise 卷积。
- **轻量级设计**：降低计算量和参数量，适合嵌入式部署。
- **分类输出**：输出 ImageNet-1k 类别的 Top-K 类别 ID 及对应置信度。

![Depthwise 与 Pointwise 卷积](./test_data/depthwise&pointwise.png)

*深度可分离卷积：每个输入通道使用各自的 D_K×D_K
depthwise 核滤波，随后的 1×1 pointwise 卷积整合各通道结果。*

<a id="support-matrix"></a>
## 支持范围

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | mobilenetv1 | python | supported |
| s100 | mobilenetv1 | python | supported |
| s600 | mobilenetv1 | python | supported |
| s100p | 任意 | python、cpp | not-supported（按支持矩阵选择目标与变体） |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv1
source .venv-mobilenetv1/bin/activate
python3 -m pip install -r samples/vision/mobilenetv1/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv1:mobilenetv1_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv1/model/download.sh x5

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。X5 上使用内置测试图
`bulbul.JPEG` 时，Top-1 应与图像主体（一只黄鹎（鸟类））一致；
S100/S600 上使用 `zebra_cls.jpg` 时，Top-5 应包含 `zebra`。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下表为 (x5-v1.1.3) 发布的 MobileNetV1 在 `RDK X5` 上的公开数据：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV1 | 224x224 | 1000 | 4.2 | 71.7% | 65.4% | 0.58 | 2800+ |



![推理结果](./test_data/inference.png)

*X5 发布的参考推理结果：随仓 [bulbul.JPEG](test_data/bulbul.JPEG) 的
Rank-1 为 `bulbul`，其后依次为 junco/snowbird、robin、chickadee、
water ouzel。*

<a id="directory"></a>
## 目录职责

- [model/](model/README_cn.md) — Manifest 驱动的制品下载，不提交二进制
- [runtime/python/](runtime/python/README_cn.md) — canonical Python 入口与任务模块
- [conversion/](conversion/README_cn.md) — 转换记录与参考配置
- [evaluator/](evaluator/README_cn.md) — 公开基准与功能检查
- `test_data/` — 内置测试图（[bulbul.JPEG](test_data/bulbul.JPEG)、[zebra_cls.jpg](test_data/zebra_cls.jpg)）
- `tests/` — 主机 unittest 套件

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为 MobileNetV1 上游发布；
模型/权重许可以上游分发为准（见上方参考实现链接）。发布制品遵循平台发布
Manifest 中的模型文件按各自上游许可使用；再分发前请核对其适用条款。
