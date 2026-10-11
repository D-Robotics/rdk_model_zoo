[English](README.md) | 简体中文

# MobileNetV2 图像分类

MobileNetV2 使用倒残差模块和线性瓶颈完成轻量图像分类。

来源：[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py) · [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

<a id="overview"></a>

## 概述

本样例为全部目标提供 Python 运行时，另为 S 系列提供 C++ 运行时。
`MobileNetV2Classifier` 类执行由 `predict` 串联的
`preprocess → infer → postprocess` 流程：按检测到的板卡从平台发布
Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，
返回带类型的 Top-K 结果（见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。C++ 侧为
S 系列 `hbDNNInferV2` 实现（见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)）。

### 算法背景

MobileNetV2 引入带线性瓶颈的倒残差结构：每个块先用 1×1 卷积扩展通道，
再做 3×3 depthwise 卷积，最后经线性（不带 ReLU）的 1×1 瓶颈投影回低维；
stride-2 块去掉快捷连接。线性瓶颈保留了低维空间中会被 ReLU 丢弃的信息
（[论文](https://arxiv.org/abs/1801.04381)、
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)）。

特性摘要：

- **倒残差结构**：先扩展通道，再进行 depthwise 卷积，最后通过线性瓶颈投影回低维空间。
- **深度可分离卷积**：相比标准卷积显著降低计算量。
- **分类输出**：输出 ImageNet-1k 类别的 Top-K 类别 ID 及对应置信度。
- **模型变体**：本 sample 提供 MobileNetV2-100、MobileNetV2-140 两个部署模型（timm 检查点，INT8）。

![MobileNetV2 架构](./test_data/mobilenetv2_architecture.png)

*倒残差块：stride-1 块（左）保留相加快捷
连接；stride-2 块（右）无快捷连接直接下采样，且仅最后的 1×1 投影为
线性。*

随附论文的可分离卷积演化图：

![可分离卷积块的演化](./test_data/seperated_conv.png)

*MobileNetV2 论文图 2：从标准卷积 (a) 到可分离块 (b)、带线性瓶颈的可分
离块 (c)、带扩展层的瓶颈块 (d)；斜线纹理表示不含非线性层的层。*

<a id="directory"></a>
## 目录结构

```text
mobilenetv2/
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
| x5 | 140 | python | supported |
| s100 | 100 | python | supported |
| s100 | 140 | python | supported |
| s100p | 100 | python | supported |
| s100p | 140 | python | supported |
| s600 | 100 | python | supported |
| s600 | 140 | python | supported |
| s100 | 100 | cpp | supported |
| s100 | 140 | cpp | supported |
| s100p | 100 | cpp | supported |
| s100p | 140 | cpp | supported |
| s600 | 100 | cpp | supported |
| s600 | 140 | cpp | supported |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy、OpenCV-Python
和 Pillow（已发布模型要求抗锯齿双三次的短边缩放，由 Pillow 完成）；
只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv2
source .venv-mobilenetv2/bin/activate
python3 -m pip install -r samples/vision/mobilenetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

C++ 构建需要 CMake、C++17 编译器、OpenCV 与 gflags 开发包以及 Horizon DNN 头文件/库——见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)。

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv2/model/mobilenetv2_100_bayese_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv2/model/download.sh x5 100

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv2/model/mobilenetv2_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100、S100P、S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。
C++ 流程（S100、S100P、S600）使用 `bash samples/vision/mobilenetv2/runtime/cpp/run.sh`。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。使用内置测试图 `Scottish_deerhound.JPEG` 时，Top-1 为
class 177（`Scottish deerhound`）；使用 `zebra_cls.jpg` 时 Top-1 为 class 340（`zebra`），X5、S100、S100P、S600 上的
两个变体都是如此（已用发布的构建在每块板上核对）。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下列数字均在真实板卡上用已发布制品测得（INT8，224x224，batch 1）。模型规模为
3.50 M 参数 / 0.60 GFLOPs（100）、6.11 M 参数 / 1.16 GFLOPs（140）；
GFLOPs 按每次 Conv 与 Gemm 乘加计 2 次运算。

**精度。** 在完整的 ImageNetV2 MatchedFrequency 集合（10,000 张图像，1,000
类）上的 Top-1 / Top-5。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果；
“板端”是编译后的模型在对应板卡上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| MobileNetV2-100 | X5 | 60.18% | 59.55% | 82.05% | 81.48% |
| MobileNetV2-100 | S100 | 60.18% | 59.61% | 82.05% | 81.61% |
| MobileNetV2-100 | S100P | 60.18% | 59.61% | 82.05% | 81.61% |
| MobileNetV2-100 | S600 | 60.18% | 59.55% | 82.05% | 81.42% |
| MobileNetV2-140 | X5 | 63.71% | 63.11% | 84.68% | 84.44% |
| MobileNetV2-140 | S100 | 63.71% | 63.13% | 84.68% | 84.41% |
| MobileNetV2-140 | S100P | 63.71% | 63.13% | 84.68% | 84.41% |
| MobileNetV2-140 | S600 | 63.71% | 63.09% | 84.68% | 84.51% |

**速度。** Runtime 数字来自 BPU 核 0 上的 `hrt_model_exec perf`（只含模型：
不含前处理与后处理）；FPS 为 3 次运行（每次 200 帧、先热身 20 帧）完成的
总帧数除以共同墙钟时间。C++ 流水线一列统计从内存中的 BGR 图像到 Top-5 列表的
单帧耗时，包括缩放、裁剪、NV12 打包、输入上传、推理和排序（不含读文件与
解码），分别为 1 路与 2 路独立流。

| 模型 | Target | Runtime 延迟，1 线程（ms） | Runtime FPS，1 / 2 线程 | C++ 流水线 FPS，1 / 2 路 | CPU / BPU（GHz） | CPU 线程数 |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV2-100 | X5 | 1.039 | 957 / 1,347 | 108 / 119 | 1.5 / 1.0 | 8 |
| MobileNetV2-100 | S100 | 0.457 | 2,102 / 3,748 | 370 / 428 | 1.5 / 1.0 | 6 |
| MobileNetV2-100 | S100P | 0.373 | 2,588 / 4,254 | 464 / 551 | 2.0 / 1.5 | 6 |
| MobileNetV2-100 | S600 | 0.325 | 2,953 / 5,758 | 732 / 963 | 2.1 / 1.5 | 18 |
| MobileNetV2-140 | X5 | 1.642 | 607 / 742 | 101 / 113 | 1.5 / 1.0 | 8 |
| MobileNetV2-140 | S100 | 0.528 | 1,836 / 3,181 | 358 / 421 | 1.5 / 1.0 | 6 |
| MobileNetV2-140 | S100P | 0.452 | 2,132 / 3,589 | 451 / 541 | 2.0 / 1.5 | 6 |
| MobileNetV2-140 | S600 | 0.366 | 2,634 / 5,086 | 710 / 927 | 2.1 / 1.5 | 18 |

每次测量时 CPU 调频策略为 `performance`，所有在线核心运行在表中所列频率；
各板卡的 CPU 与 BPU 频率不同，比较不同 target 时需谨慎。C++ 流水线使用对
Pillow 双三次缩放的标量、逐位一致的重新实现，它主导了前处理耗时，因此该列并非
优化流水线的上限。复现精度数字见[评测器](evaluator/README_cn.md)；C++ 计时工具
见[基准说明](../../../utils/tools/mobilenet/cpp/README.md)。

![推理结果](./test_data/inference.png)

*MobileNetV2-100 模型在 RDK X5 上的参考推理结果，由 `--img-save-path` 写出：随仓
[Scottish_deerhound.JPEG](test_data/Scottish_deerhound.JPEG) 的 Rank-1 为 `Scottish deerhound`（分数 0.930），其后依次为 Irish wolfhound、Afghan hound、Bouvier des Flandres、hyena。*

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)
- C++ 运行（S100/S100P/S600）：[runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为 MobileNetV2 上游发布；
模型/权重许可以上游分发为准（见上方参考实现链接）。发布制品遵循平台发布
Manifest 中的模型文件按各自上游许可使用；再分发前请核对其适用条款。
