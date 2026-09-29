# FasterNet 图像分类

FasterNet 在 RDK X5 上的 ImageNet-1k 分类：输入一张 BGR 图像，输出稳定
的 Top-K `(类别 ID, 分数, 标签)`。X5 发布交付 S、T0、T1、T2 四个变体
（论文 [Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks](https://arxiv.org/abs/2303.03667)，按源交付引用）。[English](README.md)

<a id="overview"></a>
## 概述

FasterNet 是围绕一个核心思想设计的轻量 CNN 系列：追求更高的*有效*
FLOPS（实际每秒计算量），而不是只压低理论 FLOPs。其核心算子部分卷积
（PConv）只对输入通道的一部分做空间卷积、其余通道保持不动，从而减少
冗余访存，提升边缘设备上的实际运行效率。面向 ImageNet-1k 1000 类分类。
源 README 提炼的四项特性：

- **高 FLOPS 设计**——强调实际计算效率，而不是只最小化理论 FLOPs。
- **部分卷积（PConv）**——减少冗余计算与内存访问。
- **轻量 CNN 骨干**——保持对部署友好的 CNN 结构，便于板端高效推理。
- **高效部署**——提供 S、T0、T1、T2 四个 RDK X5 部署模型，使用打包
  NV12 输入。

![与其他网络在 CPU 上的有效 FLOPS 与延迟对比](./test_data/FLOPs%20of%20Nets.png)

*图（上游论文 Fig. 2）：(a) CPU 上不同 FLOPs 对应的 FLOPS——许多网络的
有效 FLOPS 低于 ResNet50，FasterNet 保持更高；(b) CPU 上不同 FLOPs 对应
的延迟——同等 FLOPs 下 FasterNet 更快。恢复自 X5 源 README（rdk_x5
@ac11571，x5-v1.1.3）；图中条件为上游 CPU 测量，不是 RDK 板端数据（板端
数字见[性能数据](#performance)）。*

![FasterNet 架构：四级层级结构与含部分卷积的 FasterNet block](./test_data/FasterNet_architecture.png)

*图（上游论文 Fig. 4）：整体架构——四级层级堆叠 FasterNet block，前置
embedding/merging 层；PConv 细节（只对部分通道卷积）与 block 布局
PConv 3×3 → 两层逐点卷积，归一化与激活只放在中间层之后以保留特征多样性。
恢复自 X5 源 README（rdk_x5 @ac11571）；图中为上游训练结构，实际部署
制品是 INT8 量化的 s/t0/t1/t2 变体（224×224 NV12，见
[支持与实测矩阵](#support-matrix)）。*

统一实现是一条 Python 流程（仅 X5；本 sample 无 S 分支交付，两个源分支
也都没有 C++ 运行时）。Python 从平台发布 Manifest 解析唯一的制品引用，
核验板卡身份，懒加载 `hbm_runtime`，执行
`pre_process → forward → post_process` 任务（见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。迁移前的平台
分支入口在收尾前仍以兼容 shim 形式保留在 `platforms/x5/` 下，其审计记录
在迁移文档中，不在本 README 展开。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | s、t0、t1、t2 | python | supported（源契约已核实；B3 板端冒烟待执行，见下方说明） |
| s100 / s100p / s600 | 任意 | python | not-supported（S Manifest 未发布 FasterNet 资产；选择时显式报错，无跨平台回退） |

源基线：X5 侧 rdk_x5 @ac11571 (x5-v1.1.3)。统一 sample 的主机测试（28
项）全部通过。本批（B3）板端冒烟在四个 sample 主机侧全部落地后执行；
届时在此回填实测结果——在该条目出现之前，本 sample 的板端状态为
**not-run**，已验证的交付仍是源分支。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发
主机可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-fasternet
source .venv-fasternet/bin/activate
python3 -m pip install -r samples/vision/fasternet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:fasternet:FasterNet_S_224x224_nv12.bin）
#    输出：samples/vision/fasternet/model/FasterNet_S_224x224_nv12.bin
#    成功判据：下载器退出码 0 并打印实测摘要
bash samples/vision/fasternet/model/download.sh x5 s

# 2. 运行分类（输入：上一步制品 + 随附测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/fasternet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin \
  --model-path samples/vision/fasternet/model/FasterNet_S_224x224_nv12.bin \
  --test-img samples/vision/fasternet/test_data/drake.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`t0`/`t1`/`t2` 换用自己的引用与路径（见 `--list-models`）；缺省变体（未指定时）为 `s`，保持源入口的默认模型不变。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非
指定 `--img-save-path`，不写任何输出文件（源入口总会写
`test_data/result.jpg`——该副作用已移除）。使用随附 `drake.JPEG` 时
Top-5 含鸭相关 ImageNet 类别。无法识别的板卡或无匹配制品的目标（全部
S 目标）会显式报错退出。

供参考：X5 源 README（rdk_x5 @ac11571，x5-v1.1.3 旧版 Python 入口）用
下面的截图演示运行效果：旧版 `result.jpg` 绘制把 Top-5 排名叠在图上，
rank 1 为 class 97（drake）。这是源交付中的历史截图，不是本仓库当前
入口的运行结果。

![X5 源 README 的历史推理截图（rdk_x5 @ac11571）：drake 测试图与旧版
Top-5 叠加，rank 1 为 class 97（drake）](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

X5 源发布（rdk_x5 @ac11571，x5-v1.1.3）的已发布记录，未在本仓库重测
（源说明：Float Top-1 为量化前 ONNX 结果，Quant Top-1 为部署模型结果，
延迟为单帧单线程单核，FPS 为多线程）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 1135.13 |

已发布参数量 (M) 列与上游论文的模型尺寸并非全部吻合；按发布原样
记录。

<a id="directory"></a>
## 目录职责

- [model/](model/README_cn.md) — Manifest 驱动的制品下载，不检入二进制
- [runtime/python/](runtime/python/README_cn.md) — 统一 Python 入口与任务模块
- [conversion/](conversion/README_cn.md) — X5 参考 PTQ 配置（含已披露缺口）
- [evaluator/](evaluator/README_cn.md) — 发布的基准记录与功能检查
- `test_data/` — 随附测试图（[drake.JPEG](test_data/drake.JPEG) 及参考插图）
- `tests/` — 主机 unittest 套件

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行时：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为上游
FasterNet 发行版；模型/权重许可由上游发行版 govern（见上方论文
链接）。已发布制品遵循平台发布 Manifest；Manifest 不含独立许可字段，
本文件不主张额外许可。
