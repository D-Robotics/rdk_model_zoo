[English](README.md) | 简体中文

# ResNet 图像分类

ResNet 通过残差连接训练深层图像分类网络。

来源：[timm/models/resnet](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py) · [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) · [ResNet strikes back](https://arxiv.org/abs/2110.00476)

<a id="overview"></a>

## 概述

本样例为全部目标提供 Python 运行时，另为 S 系列提供 C++ 运行时。
`ResNetClassifier` 类执行由 `predict` 串联的
`preprocess → infer → postprocess` 流程：按检测到的板卡从平台发布
Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，
返回带类型的 Top-K 结果（见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。C++ 侧为
S 系列 `hbDNNInferV2` 实现（见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)）。

### 算法背景

ResNet 由 Kaiming He、Xiangyu Zhang、Shaoqing Ren 和 Jian Sun 提出。残差
学习通过快捷连接让每个块只学习残差而非完整映射，稳定深层网络优化，避免
普通层堆叠随深度增加出现的退化问题
（[论文](https://arxiv.org/abs/1512.03385)、
[timm/models/resnet](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py)）。
发布模型保持原始网络结构，使用 timm `a1_in1k` 权重，即按“ResNet strikes back”
的 A1 方案训练的检查点（[论文](https://arxiv.org/abs/2110.00476)）。

特性摘要：

- **残差学习**：恒等快捷连接把块的输入加到输出上，加深网络不会使优化退化。
- **基础块与瓶颈块**：ResNet-18/34 每个块堆叠两个 3×3 卷积；ResNet-50/101/152 使用 `1x1 → 3x3 → 1x1` 瓶颈块。
- **分类输出**：输出 ImageNet-1k 类别的 Top-K 类别 ID 及对应置信度。
- **模型变体**：本 sample 提供 ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152 五个部署模型（timm 检查点，INT8）。

![ResNet 残差块](./test_data/ResNet_architecture.png)

*ResNet-18/34 的残差基础块（左，两个 3×3 卷积）与 ResNet-50/101/152 的瓶颈构建块
（右，1×1 → 3×3 → 1×1），即 ResNet 论文图 5。*

<a id="directory"></a>
## 目录结构

```text
resnet/
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

下表列出本 Sample 已发布的模型；`—` 表示该组合未发布。全部已发布制品均为
INT8、224x224 的 ImageNet-1k 分类模型。

| 模型 | 输入 | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | 224x224 | ✓ | ✓ | ✓ | ✓ |
| ResNet-34 | 224x224 | ✓ | ✓ | ✓ | ✓ |
| ResNet-50 | 224x224 | ✓ | ✓ | ✓ | ✓ |
| ResNet-101 | 224x224 | ✓ | ✓ | ✓ | ✓ |
| ResNet-152 | 224x224 | ✓ | ✓ | ✓ | ✓ |

运行时覆盖：

- Python 覆盖上表全部组合。`--target auto|x5|s100|s100p|s600` 会解析对应的
  Manifest 制品（`--list-models` 打印精确引用）：X5 发布
  `resnet<size>_bayese_224x224_nv12.bin`；S 系列在 `s100/`、`s100p/`、`s600/`
  目录下发布 `resnet<size>_nashe/nashm/nashp_224x224_nv12.hbm`。默认变体为
  ResNet-18。
- C++（[runtime/cpp](runtime/cpp/README.md)）覆盖相同的五个变体，仅限
  S100、S100P、S600（`VARIANT=18|34|50|101|152`，默认 18）；X5 没有 C++
  运行时。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy、OpenCV-Python
和 Pillow（已发布模型要求抗锯齿双三次的短边缩放，由 Pillow 完成）；
只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 输出：输出 "host dependencies: ok"
python3 -m venv .venv-resnet
source .venv-resnet/bin/activate
python3 -m pip install -r samples/vision/resnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml, PIL; print('host dependencies: ok')"
```

C++ 构建需要 CMake、C++17 编译器、OpenCV 与 gflags 开发包以及 Horizon DNN 头文件/库——见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)。

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:resnet:resnet18_bayese_224x224_nv12.bin）
#    输出：samples/vision/resnet/model/resnet18_bayese_224x224_nv12.bin
#    输出：退出码 0 并打印观察 digest
bash samples/vision/resnet/model/download.sh x5 18

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    输出：退出码 0 且打印 Top-5 列表
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_bayese_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_bayese_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100、S100P、S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。
C++ 流程（S100、S100P、S600）使用 `bash samples/vision/resnet/runtime/cpp/run.sh`。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。使用内置测试图 `white_wolf.JPEG` 时，Top-1 为
class 270（`white wolf`）；使用 `zebra_cls.jpg` 时 Top-1 为 class 340（`zebra`），X5、S100、S100P、S600 上的
五个变体都是如此。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

下列数字均在真实板卡上用已发布制品测得（INT8，224x224，batch 1）。模型规模为
11.69 M 参数 / 3.63 GFLOPs（18）、21.80 M 参数 / 7.33 GFLOPs（34）、25.56 M 参数 / 8.18 GFLOPs（50）、44.55 M 参数 / 15.60 GFLOPs（101）、60.19 M 参数 / 23.03 GFLOPs（152）；
GFLOPs 按每次 Conv 与 Gemm 乘加计 2 次运算。

**精度。** 在完整的 ImageNetV2 MatchedFrequency 集合（10,000 张图像，1,000
类）上的 Top-1 / Top-5。它不是 ILSVRC2012 验证集，因此数值不能与 ImageNet-1k
验证集的结果相比。“FP32”是同一检查点的 ONNX 导出在相同裁剪图上的结果；
“板端”是编译后的模型在对应板卡上的结果。

| 模型 | Target | FP32 Top-1 | 板端 Top-1 | FP32 Top-5 | 板端 Top-5 |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | X5 | 59.35% | 58.91% | 80.94% | 80.56% |
| ResNet-18 | S100 | 59.35% | 58.79% | 80.94% | 80.46% |
| ResNet-18 | S100P | 59.35% | 58.79% | 80.94% | 80.46% |
| ResNet-18 | S600 | 59.35% | 58.88% | 80.94% | 80.58% |
| ResNet-34 | X5 | 64.92% | 64.30% | 85.06% | 84.71% |
| ResNet-34 | S100 | 64.92% | 64.26% | 85.06% | 84.75% |
| ResNet-34 | S100P | 64.92% | 64.26% | 85.06% | 84.75% |
| ResNet-34 | S600 | 64.92% | 64.23% | 85.06% | 84.75% |
| ResNet-50 | X5 | 68.84% | 66.90% | 87.59% | 86.25% |
| ResNet-50 | S100 | 68.84% | 67.01% | 87.59% | 86.40% |
| ResNet-50 | S100P | 68.84% | 67.01% | 87.59% | 86.40% |
| ResNet-50 | S600 | 68.84% | 66.94% | 87.59% | 86.41% |
| ResNet-101 | X5 | 70.32% | 68.59% | 88.35% | 87.66% |
| ResNet-101 | S100 | 70.32% | 68.64% | 88.35% | 87.67% |
| ResNet-101 | S100P | 70.32% | 68.64% | 88.35% | 87.67% |
| ResNet-101 | S600 | 70.32% | 68.65% | 88.35% | 87.56% |
| ResNet-152 | X5 | 70.87% | 69.34% | 88.16% | 87.78% |
| ResNet-152 | S100 | 70.87% | 69.27% | 88.16% | 87.74% |
| ResNet-152 | S100P | 70.87% | 69.27% | 88.16% | 87.74% |
| ResNet-152 | S600 | 70.87% | 69.22% | 88.16% | 87.74% |

**速度。** Runtime 数字来自 BPU 核 0 上的 `hrt_model_exec perf`（只含模型：
不含前处理与后处理）；FPS 为 3 次运行（每次 200 帧、先热身 20 帧）完成的
总帧数除以共同墙钟时间。C++ 流水线一列统计从内存中的 BGR 图像到 Top-5 列表的
单帧耗时，包括缩放、裁剪、NV12 打包、输入上传、推理和排序（不含读文件与
解码），分别为 1 路与 2 路独立流。

| 模型 | Target | Runtime 延迟，1 线程（ms） | Runtime FPS，1 / 2 线程 | C++ 流水线 FPS，1 / 2 路 | CPU / BPU（GHz） | CPU 线程数 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | X5 | 2.740 | 364 / 409 | 95 / 109 | 1.5 / 1.0 | 8 |
| ResNet-18 | S100 | 0.631 | 1,553 / 2,583 | 363 / 440 | 1.5 / 1.0 | 6 |
| ResNet-18 | S100P | 0.534 | 1,826 / 3,048 | 456 / 565 | 2.0 / 1.5 | 6 |
| ResNet-18 | S600 | 0.394 | 2,452 / 4,738 | 715 / 930 | 2.1 / 1.5 | 18 |
| ResNet-34 | X5 | 4.625 | 216 / 231 | 80 / 96 | 1.5 / 1.0 | 8 |
| ResNet-34 | S100 | 0.918 | 1,066 / 1,513 | 329 / 409 | 1.5 / 1.0 | 6 |
| ResNet-34 | S100P | 0.791 | 1,237 / 1,747 | 409 / 525 | 2.0 / 1.5 | 6 |
| ResNet-34 | S600 | 0.525 | 1,860 / 3,493 | 650 / 1,003 | 2.1 / 1.5 | 18 |
| ResNet-50 | X5 | 5.404 | 185 / 196 | 76 / 92 | 1.5 / 1.0 | 8 |
| ResNet-50 | S100 | 1.070 | 919 / 1,231 | 311 / 396 | 1.5 / 1.0 | 6 |
| ResNet-50 | S100P | 0.916 | 1,072 / 1,433 | 392 / 510 | 2.0 / 1.5 | 6 |
| ResNet-50 | S600 | 0.593 | 1,652 / 3,081 | 628 / 1,050 | 2.1 / 1.5 | 18 |
| ResNet-101 | X5 | 9.044 | 110 / 115 | 59 / 113 | 1.5 / 1.0 | 8 |
| ResNet-101 | S100 | 1.614 | 613 / 739 | 266 / 380 | 1.5 / 1.0 | 6 |
| ResNet-101 | S100P | 1.395 | 709 / 852 | 329 / 524 | 2.0 / 1.5 | 6 |
| ResNet-101 | S600 | 0.876 | 1,123 / 2,077 | 528 / 941 | 2.1 / 1.5 | 18 |
| ResNet-152 | X5 | 12.332 | 81 / 83 | 49 / 83 | 1.5 / 1.0 | 8 |
| ResNet-152 | S100 | 2.128 | 466 / 536 | 235 / 423 | 1.5 / 1.0 | 6 |
| ResNet-152 | S100P | 1.842 | 538 / 618 | 285 / 545 | 2.0 / 1.5 | 6 |
| ResNet-152 | S600 | 1.137 | 869 / 1,598 | 467 / 831 | 2.1 / 1.5 | 18 |

每次测量时 CPU 调频策略为 `performance`，所有在线核心运行在表中所列频率；
各板卡的 CPU 与 BPU 频率不同，比较不同 target 时需谨慎。C++ 流水线使用对
Pillow 双三次缩放的标量、逐位一致的重新实现，它主导了前处理耗时；表中延迟列为该实现的实测值。复现精度数字见[评测器](evaluator/README_cn.md)；C++ 计时工具
见[基准说明](../../../utils/tools/classification/cpp/README.md)。

![推理结果](./test_data/inference.png)

*ResNet-18 模型在 RDK X5 上的参考推理结果，由 `--img-save-path` 写出：随仓
[white_wolf.JPEG](test_data/white_wolf.JPEG) 的 Rank-1 为 `white wolf`（分数 0.998），其后依次为 timber wolf、ice bear、Arctic fox、Samoyed。*

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)
- C++ 运行（S100/S100P/S600）：[runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为 ResNet 上游发布；
模型/权重许可以上游分发为准（见上方参考实现链接）。发布制品遵循平台发布
Manifest 中的模型文件按各自上游许可使用；再分发前请核对其适用条款。
