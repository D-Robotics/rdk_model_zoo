[English](README.md) | 简体中文

# MobileNetV4 图像分类

MobileNetV4 是采用通用倒瓶颈模块的图像分类模型系列。

来源：[timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py) · [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

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
- **模型变体**：本 sample 提供 Conv-Small、Conv-Medium 和 Conv-Large 三个部署模型。

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
| x5 | large | python | supported |
| s100 | small | python | supported |
| s100 | medium | python | supported |
| s100 | large | python | supported |
| s100p | small | python | supported |
| s100p | medium | python | supported |
| s100p | large | python | supported |
| s600 | small | python | supported |
| s600 | medium | python | supported |
| s600 | large | python | supported |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy、OpenCV-Python
和 Pillow（已发布模型要求抗锯齿双三次的短边缩放，由 Pillow 完成）；
只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv4
source .venv-mobilenetv4/bin/activate
python3 -m pip install -r samples/vision/mobilenetv4/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv4/model/download.sh x5 small

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100、S100P、S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。使用内置测试图 `great_grey_owl.JPEG` 时，Top-1 为
class 24（`great grey owl`）；使用 `zebra_cls.jpg` 时 Top-1 为 class 340（`zebra`），X5、S100、S100P、S600 上的
三个变体都是如此（已用发布的构建在每块板上核对）。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下列数字均在真实板卡上用已发布制品测得（INT8，224x224、256x256，batch 1）。模型规模为
3.77 M 参数 / 0.37 GFLOPs（Small）、9.72 M 参数 / 1.66 GFLOPs（Medium）、32.59 M 参数 / 5.67 GFLOPs（Large）；
GFLOPs 按每次 Conv 与 Gemm 乘加计 2 次运算。

**精度。** 在完整的 ImageNetV2 MatchedFrequency 集合（10,000 张图像，1,000
类）上的 Top-1 / Top-5。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果；
“板端”是编译后的模型在对应板卡上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Small | X5 | 60.96% | 58.72% | 82.64% | 80.53% |
| MobileNetV4-Conv-Small | S100 | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S100P | 60.96% | 58.56% | 82.64% | 80.78% |
| MobileNetV4-Conv-Small | S600 | 60.96% | 58.57% | 82.64% | 80.94% |
| MobileNetV4-Conv-Medium | X5 | 67.35% | 66.25% | 87.86% | 87.54% |
| MobileNetV4-Conv-Medium | S100 | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S100P | 67.35% | 66.94% | 87.86% | 87.65% |
| MobileNetV4-Conv-Medium | S600 | 67.35% | 66.75% | 87.86% | 87.56% |
| MobileNetV4-Conv-Large | X5 | 70.79% | 69.85% | 89.15% | 89.13% |
| MobileNetV4-Conv-Large | S100 | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S100P | 70.79% | 69.66% | 89.15% | 89.04% |
| MobileNetV4-Conv-Large | S600 | 70.79% | 69.62% | 89.15% | 89.16% |

**速度。** Runtime 数字来自 BPU 核 0 上的 `hrt_model_exec perf`（只含模型：
不含前处理与后处理）；FPS 为 3 次运行（每次 200 帧、先热身 20 帧）完成的
总帧数除以共同墙钟时间。C++ 流水线一列统计从内存中的 BGR 图像到 Top-5 列表的
单帧耗时，包括缩放、裁剪、NV12 打包、输入上传、推理和排序（不含读文件与
解码），分别为 1 路与 2 路独立流。

| 模型 | Target | Runtime 延迟，1 线程（ms） | Runtime FPS，1 / 2 线程 | C++ 流水线 FPS，1 / 2 路 | CPU / BPU（GHz） | CPU 线程数 |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Small | X5 | 0.999 | 994 / 1,428 | 108 / 119 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Small | S100 | 0.407 | 2,359 / 4,203 | 362 / 416 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Small | S100P | 0.355 | 2,710 / 4,501 | 459 / 534 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Small | S600 | 0.311 | 3,078 / 5,991 | 741 / 911 | 2.1 / 1.5 | 18 |
| MobileNetV4-Conv-Medium | X5 | 2.066 | 483 / 564 | 101 / 115 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Medium | S100 | 0.607 | 1,595 / 2,831 | 366 / 431 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Medium | S100P | 0.530 | 1,829 / 3,174 | 461 / 551 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Medium | S600 | 0.407 | 2,381 / 4,605 | 707 / 980 | 2.1 / 1.5 | 18 |
| MobileNetV4-Conv-Large | X5 | 5.423 | 184 / 195 | 71 / 87 | 1.5 / 1.0 | 8 |
| MobileNetV4-Conv-Large | S100 | 1.147 | 858 / 1,120 | 285 / 359 | 1.5 / 1.0 | 6 |
| MobileNetV4-Conv-Large | S100P | 1.063 | 926 / 1,188 | 346 / 458 | 2.0 / 1.5 | 6 |
| MobileNetV4-Conv-Large | S600 | 0.631 | 1,553 / 2,829 | 582 / 1,017 | 2.1 / 1.5 | 18 |

每次测量时 CPU 调频策略为 `performance`，所有在线核心运行在表中所列频率；
各板卡的 CPU 与 BPU 频率不同，比较不同 target 时需谨慎。C++ 流水线使用对
Pillow 双三次缩放的标量、逐位一致的重新实现，它主导了前处理耗时，因此该列并非
优化流水线的上限。复现精度数字见[评测器](evaluator/README_cn.md)；C++ 计时工具
见[基准说明](../../../utils/tools/mobilenet/cpp/README.md)。

![推理结果](./test_data/inference.png)

*Small 模型在 RDK X5 上的参考推理结果，由 `--img-save-path` 写出：随仓
[great_grey_owl.JPEG](test_data/great_grey_owl.JPEG) 的 Rank-1 为
`great grey owl`（分数 0.957），其后依次为 vulture、cheetah、lynx、
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
