# MobileNetV4 图像分类

MobileNetV4 是采用通用倒瓶颈模块的图像分类模型系列。

来源：[timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py) · [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

[English README](README.md)

<a id="overview"></a>

## 概述

本样例为全部目标提供同一个 Python 运行时。
`MobileNetV4Classifier` 类执行由 `predict` 串联的 `preprocess → infer → postprocess` 流程：按检测到的板卡从平台发布 Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，返回带类型的 Top-K 结果（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。

### 算法背景

MobileNetV4 把移动端 CNN 的块设计空间统一到 Universal Inverted Bottleneck
（UIB）：按启用的 depthwise 层不同，同一块可表达 inverted bottleneck、
ConvNeXt 风格块、FFN 风格块或 ExtraDW 变体；移动端多查询注意力
（Mobile Multi-Query Attention）在移动加速器上有收益的位置引入注意力
（[论文](https://arxiv.org/abs/2404.10518)、
[timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)）。

特性摘要：

- **Universal Inverted Bottleneck**：统一 inverted bottleneck、ConvNeXt 风格模块、FFN 风格模块和 ExtraDW 变体。
- **Mobile Multi-Query Attention**：面向移动端加速器优化的注意力结构。
- **模型变体**：本 sample 提供 Conv-Small 和 Conv-Medium 两个部署模型。

![MobileNetV4 UIB 块](./test_data/MobileNetV4_architecture.png)

*Universal Inverted Bottleneck 块（论文图 4）：带两个可选 depthwise
层的 UIB 块、其
Extra-DW / Inverted Bottleneck / ConvNeXt / FFN 实例化，以及替代的
fused IB。*

<a id="directory"></a>
## 目录结构

```text
mobilenetv4/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # Python 依赖
```

<a id="support-matrix"></a>
## 支持范围

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | small | python | supported |
| x5 | medium | python | supported |
| s100 | small | python | supported |
| s100 | medium | python | supported |
| s600 | medium | python | supported |
| s600 | small | python | supported |
| s100p | 任意 | python、cpp | not-supported（按支持矩阵选择目标与变体） |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv4
source .venv-mobilenetv4/bin/activate
python3 -m pip install -r samples/vision/mobilenetv4/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv4/model/download.sh x5 --variant small

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。X5 上使用内置测试图
`great_grey_owl.JPEG` 时，Top-1 应与图像主体（一只灰林鸮）一致；
S100/S600 上使用 `zebra_cls.jpg` 时，Top-5 应包含 `zebra`。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下表为 (x5-v1.1.3) 发布的 MobileNetV4 在 `RDK X5` 上的公开数据：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Medium | 224x224 | 1000 | 9.7 | 76.8% | 75.1% | 2.42 | 572+ |
| MobileNetV4-Conv-Small | 224x224 | 1000 | 3.8 | 70.8% | 68.8% | 1.18 | 1436+ |


![推理结果](./test_data/inference.png)

*X5 发布的参考推理结果：随仓
[great_grey_owl.JPEG](test_data/great_grey_owl.JPEG) 的 Rank-1 为
`great grey owl`，其后依次为 ruffed grouse、partridge、meerkat、
prairie chicken。*

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为 MobileNetV4 上游发布；
模型/权重许可以上游分发为准（见上方参考实现链接）。发布制品遵循平台发布
Manifest 中的模型文件按各自上游许可使用；再分发前请核对其适用条款。
