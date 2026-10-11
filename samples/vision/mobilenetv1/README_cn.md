[English](README.md) | 简体中文

# MobileNetV1 图像分类

MobileNetV1 使用深度可分离卷积完成轻量图像分类。

来源：[tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) · [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

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
- **模型变体**：本 sample 提供 MobileNetV1-100、MobileNetV1-125 两个部署模型（timm 检查点，INT8）。

![Depthwise 与 Pointwise 卷积](./test_data/depthwise&pointwise.png)

*深度可分离卷积：每个输入通道使用各自的 D_K×D_K
depthwise 核滤波，随后的 1×1 pointwise 卷积整合各通道结果。*

<a id="directory"></a>
## 目录结构

```text
mobilenetv1/
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
| x5 | 100 | python | supported |
| x5 | 125 | python | supported |
| s100 | 100 | python | supported |
| s100 | 125 | python | supported |
| s100p | 100 | python | supported |
| s100p | 125 | python | supported |
| s600 | 100 | python | supported |
| s600 | 125 | python | supported |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy、OpenCV-Python
和 Pillow（已发布模型要求抗锯齿双三次的短边缩放，由 Pillow 完成）；
只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv1
source .venv-mobilenetv1/bin/activate
python3 -m pip install -r samples/vision/mobilenetv1/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv1/model/mobilenetv1_100_bayese_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv1/model/download.sh x5 100

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100、S100P、S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。使用内置测试图 `bulbul.JPEG` 时，Top-1 为
class 16（`bulbul`）；使用 `zebra_cls.jpg` 时 Top-1 为 class 340（`zebra`），X5、S100、S100P、S600 上的
两个变体都是如此（已用发布的构建在每块板上核对）。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下列数字均在真实板卡上用已发布制品测得（INT8，224x224，batch 1）。模型规模为
4.23 M 参数 / 1.14 GFLOPs（100）、6.27 M 参数 / 1.76 GFLOPs（125）；
GFLOPs 按每次 Conv 与 Gemm 乘加计 2 次运算。

**精度。** 在完整的 ImageNetV2 MatchedFrequency 集合（10,000 张图像，1,000
类）上的 Top-1 / Top-5。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果；
“板端”是编译后的模型在对应板卡上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV1-100 | X5 | 62.86% | 59.45%* | 84.37% | 81.53% |
| MobileNetV1-100 | S100 | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S100P | 62.86% | 62.00% | 84.37% | 83.51% |
| MobileNetV1-100 | S600 | 62.86% | 62.04% | 84.37% | 83.66% |
| MobileNetV1-125 | X5 | 64.25% | 63.14% | 85.23% | 84.17% |
| MobileNetV1-125 | S100 | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S100P | 64.25% | 63.19% | 85.23% | 84.27% |
| MobileNetV1-125 | S600 | 64.25% | 62.95% | 85.23% | 84.12% |

\* MobileNetV1-100 在 X5 上相对 FP32 的 Top-1 损失为 5.4%，超过 5% 的验收目标，
作为有记录的例外发布。MobileNetV1-100 的 S 系列构建使用工具链的权重偏差校正
（仍为 INT8），损失降到 1.3-1.4%；X5 工具链的偏差校正反而使 X5 构建变差。

**速度。** Runtime 数字来自 BPU 核 0 上的 `hrt_model_exec perf`（只含模型：
不含前处理与后处理）；FPS 为 3 次运行（每次 200 帧、先热身 20 帧）完成的
总帧数除以共同墙钟时间。C++ 流水线一列统计从内存中的 BGR 图像到 Top-5 列表的
单帧耗时，包括缩放、裁剪、NV12 打包、输入上传、推理和排序（不含读文件与
解码），分别为 1 路与 2 路独立流。

| 模型 | Target | Runtime 延迟，1 线程（ms） | Runtime FPS，1 / 2 线程 | C++ 流水线 FPS，1 / 2 路 | CPU / BPU（GHz） | CPU 线程数 |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV1-100 | X5 | 1.146 | 867 / 1,178 | 106 / 118 | 1.5 / 1.0 | 8 |
| MobileNetV1-100 | S100 | 0.431 | 2,217 / 3,762 | 371 / 432 | 1.5 / 1.0 | 6 |
| MobileNetV1-100 | S100P | 0.361 | 2,645 / 4,426 | 479 / 559 | 2.0 / 1.5 | 6 |
| MobileNetV1-100 | S600 | 0.311 | 3,081 / 5,980 | 744 / 942 | 2.1 / 1.5 | 18 |
| MobileNetV1-125 | X5 | 1.690 | 589 / 718 | 102 / 115 | 1.5 / 1.0 | 8 |
| MobileNetV1-125 | S100 | 0.485 | 1,993 / 3,366 | 371 / 433 | 1.5 / 1.0 | 6 |
| MobileNetV1-125 | S100P | 0.432 | 2,230 / 3,854 | 473 / 564 | 2.0 / 1.5 | 6 |
| MobileNetV1-125 | S600 | 0.335 | 2,879 / 5,502 | 729 / 972 | 2.1 / 1.5 | 18 |

每次测量时 CPU 调频策略为 `performance`，所有在线核心运行在表中所列频率；
各板卡的 CPU 与 BPU 频率不同，比较不同 target 时需谨慎。C++ 流水线使用对
Pillow 双三次缩放的标量、逐位一致的重新实现，它主导了前处理耗时，因此该列并非
优化流水线的上限。复现精度数字见[评测器](evaluator/README_cn.md)；C++ 计时工具
见[基准说明](../../../utils/tools/mobilenet/cpp/README.md)。

![推理结果](./test_data/inference.png)

*MobileNetV1-100 模型在 RDK X5 上的参考推理结果，由 `--img-save-path` 写出：随仓
[bulbul.JPEG](test_data/bulbul.JPEG) 的 Rank-1 为 `bulbul`（分数 0.872），其后依次为 junco、jay、robin、water ouzel。*

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
