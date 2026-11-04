[English](README.md) | 简体中文

# EdgeNeXt 图像分类

EdgeNeXt 将卷积与注意力结合，用于轻量图像分类。

来源：[EdgeNeXt: Efficiently Amalgamated
CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589) · [mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)

<a id="overview"></a>

## 概述

EdgeNeXt 是面向移动视觉的高效 CNN-Transformer 混合架构：四级金字塔把
卷积编码器与 SDTA（Split Depth-wise Transpose Attention）编码器组合在
一起，在分类精度、模型体积与推理速度之间取得平衡，面向 ImageNet-1k
1000 类分类。四项核心特性：

- **CNN-Transformer 混合设计**——结合卷积的推理效率与 Transformer 风格
  的全局特征建模。
- **四级金字塔**——采用对部署友好的层级化特征提取结构。
- **SDTA 编码器**——通过通道拆分（拆分 3×3 分支）与转置注意力编码
  多尺度特征。
- **高效部署**——提供 base、small、x-small、xx-small 四个 RDK X5
  部署模型，使用打包 NV12 输入。

![EdgeNeXt 架构：四级金字塔及 NxN 卷积编码器与 SDTA 编码器细节](./test_data/EdgeNeXt_architecture.png)

*图：EdgeNeXt 架构——四级金字塔（上）与 NxN 卷积编码器（左下）、含
拆分 3×3 分支和转置注意力的 SDTA 编码器（右下）。图中为上游训练结构，实际部署制品是
INT8 量化的 base/small/x_small/xx_small 变体（224×224 NV12，见
[支持范围](#support-matrix)）。*

Python 入口见 `runtime/python/main.py`，模型推理流程在 `classify.py`，参数和结果展示在 `cli.py`。
[runtime/python/README_cn.md](runtime/python/README_cn.md)

<a id="directory"></a>
## 目录结构

```text
edgenext/
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

下表列出本 Sample 已发布的模型；`—` 表示该组合未发布。

| 模型 | 输入 | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| EdgeNeXt-xx-small | 224x224 | ✓ | — | — | — |
| EdgeNeXt-x-small | 224x224 | ✓ | — | — | — |
| EdgeNeXt-small | 224x224 | ✓ | — | — | — |
| EdgeNeXt-base | 224x224 | ✓ | — | — | — |

运行时覆盖：Python CLI 通过 `--target auto|x5` 和
`--variant xx_small|x_small|small|base`（默认 `base`）覆盖全部已发布组合。
仅 X5 归档发布 `EdgeNeXt_<size>_224x224_nv12.bin`；`--target s100`、
`s100p`、`s600` 不会解析出任何资产。本 Sample 不提供 C++ 运行时。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发
主机可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 输出：输出 "host dependencies: ok"
python3 -m venv .venv-edgenext
source .venv-edgenext/bin/activate
python3 -m pip install -r samples/vision/edgenext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:edgenext:EdgeNeXt_base_224x224_nv12.bin）
#    输出：samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin
#    输出：下载器退出码 0 并打印实测摘要
bash samples/vision/edgenext/model/download.sh x5 base

# 2. 运行分类（输入：上一步制品 + 随附测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    输出：退出码 0 且打印 Top-5 列表
python3 samples/vision/edgenext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin \
  --model-path samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin \
  --test-img samples/vision/edgenext/test_data/Zebra.jpg \
  --label-file datasets/imagenet/imagenet_classes.names
```

`small`/`x_small`/`xx_small` 换用自己的引用与路径（见 `--list-models`）；缺省变体（未指定时）为 `base`。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非
指定 `--img-save-path`，不写任何输出文件。使用随附 `Zebra.jpg` 时
Top-5 含斑马相关 ImageNet 类别。按支持矩阵选择目标并准备对应制品，在对应板卡上运行。

下图为 X5 发布的参考运行效果：demo 将 Top-5 排名叠画在随附的
`Zebra.jpg` 上，rank 1 为 class 340（zebra）。

![X5 参考推理结果：zebra 测试图与 Top-5 叠加，rank 1 为 class
340（zebra）](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

RDK X5 上的已发布数值（X5 发布 x5-v1.1.3；Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果，
延迟为单帧单线程单核，FPS 为多线程）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeNeXt-base | 224x224 | 18.51 | 78.21% | 74.52% | 8.80 | 113.35 |
| EdgeNeXt-small | 224x224 | 5.59 | 76.50% | 71.75% | 4.41 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 2.34 | 71.75% | 66.25% | 2.88 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1.33 | 69.50% | 64.25% | 2.47 | 403.49 |

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行时：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为上游
EdgeNeXt 发行版；模型/权重许可由上游发行版约束（见上方论文
链接）。已发布制品遵循平台发布 Manifest。模型与权重按上游许可使用。

<!-- converted-batch-20261102 -->
## 已转换新增规格（2026-11-02）

以下 timm 权重单独转换，已通过严格权重加载、结构重参数化及两组输入的
PyTorch/ONNX 主机对照、checker 和编译。板端冒烟、精度、Runtime 1/2 并发和
C++ 1/2 流均为 **not-run**，历史指标和截图不适用于这些产物。原有默认规格保留。
新规格需要 Pillow，使用 bicubic 短边 256、四舍五入居中裁剪 224；Mapper 融合
该权重实际的归一化参数。输出为 float32 1,000 类 logits，由原生 Runtime 做 softmax。

| 规格 | 模型文件 | 权重版本 | 板端 |
|---|---|---|---|
| `timm-xx-small-in1k` | `edgenext_xx_small_in1k_bayese_224x224_nv12.bin` | [`e10d04b6e70f`](https://huggingface.co/timm/edgenext_xx_small.in1k/blob/e10d04b6e70f33cfb54ebd4bce2ede078f4c9ba2/README.md) | not-run |
| `timm-x-small-in1k` | `edgenext_x_small_in1k_bayese_224x224_nv12.bin` | [`8525b3ecc3c7`](https://huggingface.co/timm/edgenext_x_small.in1k/blob/8525b3ecc3c7a25c07406163de827d8814f5f1ef/README.md) | not-run |
| `timm-small-usi-in1k` | `edgenext_small_usi_in1k_bayese_224x224_nv12.bin` | [`81bd8f2ee563`](https://huggingface.co/timm/edgenext_small.usi_in1k/blob/81bd8f2ee56330bdf6a7a2e47e1b2b82eb34f978/README.md) | not-run |
| `timm-base-usi-in1k` | `edgenext_base_usi_in1k_bayese_224x224_nv12.bin` | [`ce0946556d0e`](https://huggingface.co/timm/edgenext_base.usi_in1k/blob/ce0946556d0ed93eaa61c686955e3118814573bb/README.md) | not-run |

```bash
python3 samples/vision/edgenext/model/download.py --target x5 --variant timm-xx-small-in1k
python3 samples/vision/edgenext/runtime/python/main.py --target x5 --variant timm-xx-small-in1k --dry-run
```

精确 URL、SHA-256 和大小登记在 `docs/release/x5/models.yaml`；来源及主机对照
记录见 `conversion/converted.json`。模型许可逐项保留在公开 release 清单中。
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

本轮新增的独立规格和显式命令见 [模型准备](model/README_cn.md)。板端精度与性能尚未实测。
