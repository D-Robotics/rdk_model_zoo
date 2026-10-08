# EfficientFormer 图像分类

EfficientFormer 在 RDK X5 上的 ImageNet-1k 分类：输入一张 BGR 图像，
输出稳定的 Top-K `(类别 ID, 分数, 标签)`。X5 发布交付 EfficientFormer-L1
与 L3 变体（论文 [EfficientFormer: Vision Transformers at MobileNet
Speed](https://arxiv.org/abs/2206.01191)）。[English](README.md)

<a id="overview"></a>

## 概述

本样例提供面向 X5 的 Python 运行时。
`EfficientFormerClassifier` 类执行由 `predict` 串联的
`preprocess → infer → postprocess` 流程：从平台发布 Manifest 解析唯一的
制品引用，核验板卡身份，懒加载 `hbm_runtime`，返回带类型的 Top-K 结果
（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。

### 算法背景

EfficientFormer 是面向移动端推理速度设计的视觉 Transformer 家族。其设计
从 ViT 类网络的延迟剖析出发，剔除在边缘硬件上表现不佳的算子；维度一致的
MetaBlock 在早期阶段保持 4D 卷积式 token 混合，只在有收益的阶段切换到
3D 全局注意力，从而在保持 Transformer 建模能力的同时维持部署友好性
（[论文](https://arxiv.org/abs/2206.01191)、
[snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer)）。

特性摘要：

- **延迟驱动设计**：通过延迟分析剔除低效的 ViT 算子，面向移动端推理。
- **维度一致的块**：保持部署友好的张量布局，保证高效执行。
- **边缘部署**：提供 L1、L3 两个 RDK X5 部署模型，输入为 packed NV12。

![延迟剖析](./test_data/latency_profiling.png)

*延迟剖析（论文图 2）：iPhone 12/CoreML 上
CNN 与 ViT 类模型的分算子延迟拆分，括号内为 ImageNet-1k top-1 —
这是引出维度一致块设计的研究依据，为论文实验数据，不是 RDK X5 实测。*

![EfficientFormer 架构](./test_data/EfficientFormer_architecture.png)

*架构总览（论文图 3）：卷积 stem 作为 patch
embedding，阶段 1–3i 为带局部池化的 4D MetaBlock，阶段 3j–4 为带全局
MHSA 的 3D MetaBlock，整体按维度一致方式组织。*

<a id="directory"></a>
## 目录结构

```text
efficientformer/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # 源码或数据文件
```

<a id="support-matrix"></a>
## 支持范围

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | l1、l3 | python | supported |
| s100 / s100p / s600 | 任意 | python | not-supported（按支持矩阵选择目标与变体） |

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发
主机可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-efficientformer
source .venv-efficientformer/bin/activate
python3 -m pip install -r samples/vision/efficientformer/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:efficientformer:EfficientFormer_l3_224x224_nv12.bin）
#    输出：samples/vision/efficientformer/model/EfficientFormer_l3_224x224_nv12.bin
#    成功判据：下载器退出码 0 并打印实测摘要
bash samples/vision/efficientformer/model/download.sh x5 l3

# 2. 运行分类（输入：上一步制品 + 随附测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/efficientformer/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientformer:EfficientFormer_l3_224x224_nv12.bin \
  --model-path samples/vision/efficientformer/model/EfficientFormer_l3_224x224_nv12.bin \
  --test-img samples/vision/efficientformer/test_data/bittern.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`l1` 换用自己的引用与路径（见 `--list-models`）；缺省变体（未指定时）
为 `l3`。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非
指定 `--img-save-path`，不写任何输出文件。使用随附 `bittern.JPEG` 时
Top-5 含麻鳽相关 ImageNet 类别。按支持矩阵选择目标并准备对应制品；运行时会在加载模型前核验板卡身份。

<a id="performance"></a>
## 性能数据

RDK X5 上的已发布数值（X5 发布 x5-v1.1.3；Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果，
延迟为单帧单线程单核，FPS 为多线程）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientFormer-L3 | 224x224 | 31.3 | 76.75% | 76.05% | 17.55 | 65.56 | 60.52 |
| EfficientFormer-L1 | 224x224 | 12.3 | 76.75% | 67.72% | 5.88 | 20.69 | 191.605 |

![推理结果](./test_data/inference.png)

*X5 发布的参考推理结果：随仓 [bittern.JPEG](test_data/bittern.JPEG) 的
Rank-1 为 `bittern`，其后依次为 partridge、European gallinule、bustard、
coucal。*

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行时：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为上游
EfficientFormer 发行版；模型/权重许可由上游发行版约束（见上方论文
链接）。已发布制品遵循平台发布 Manifest；Manifest 不含独立许可字段，
本文件不主张额外许可。
