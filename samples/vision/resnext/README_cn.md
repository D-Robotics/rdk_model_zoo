# ResNeXt 图像分类

<a id="overview"></a>

## 概述

ResNeXt 在残差网络的基础上引入 split-transform-merge 设计，通过增加 cardinality 而不只是单纯增加网络深度或宽度，提升模型表达能力。该结构保留了简洁的残差主干，并通过组卷积提高表示效率。

- **论文地址**: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- **参考实现**: [facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)

源 README 提炼的特性：

- **Cardinality（分支数）**——通过增加并行变换路径数（32×4d 中的
  "32"：32 组）提升表达能力，而不是只加深或加宽网络。
- **组卷积**——在保持参数量/FLOPs 与对应 ResNet 接近的同时平衡精度与
  计算效率。
- **残差主干**——保留稳定的残差学习模式（block 内
  split-transform-merge）。
- **分类输出**——产出 ImageNet-1k 标签的 Top-K 类别 ID 与置信度。

![上游论文中 ResNeXt-50 32x4d 与 ResNet-50 的逐 stage 结构表](./test_data/ResNeXt_architecture.png)

*图（上游论文 Table 1）：逐 stage 的 block 结构表——每个 ResNeXt
bottleneck 把稠密 1×1/3×3/1×1 变换替换为分组 3×3（C=32），参数量
（25.0 对 25.5 M）与 FLOPs（4.2 对 4.1 G）相对 ResNet-50 几乎不变。
恢复自 X5 源 README（rdk_x5
@ac115717197920355fc390bb04299b20e6436864）；图中为上游训练结构，实际
部署制品是 INT8 量化的 `50_32x4d` 变体（224×224 NV12，见
[支持与实测矩阵](#support-matrix)）。*

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。统一 Python 任务复用已有分类实现，按前处理、推理、后处理组织；标签读取、绘图和文件输出由 CLI 负责。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | 50_32x4d | supported-not-run | not-supported |
| s100 | 50_32x4d | not-supported | not-supported |
| s100p | 50_32x4d | not-supported | not-supported |
| s600 | 50_32x4d | not-supported | not-supported |

`supported-not-run` 表示已有统一实现与发布制品，但本轮板测未执行；S 系列无对应制品，所有目标均无 C++ 实现。[主机验证记录](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md)不替代板端证据。

固定源：rdk_x5 @ac115717197920355fc390bb04299b20e6436864。此 sample 没有 C++ 运行交付。CLI 小写 ID 映射到准确发布文件名；文件名大小写保留。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机依赖按下方装入虚拟环境。SciPy 仅用于保留源实现的对照测试，统一推理不依赖它。

已验证的本地主机环境：Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3、SciPy 1.18.1。该组合仅用于主机回归，不是板端依赖版本承诺。计划验证 X5 4GB/8GB；板端系统镜像、Python、SDK 的准确版本及最低内存仍待实测登记。磁盘需容纳仓库、所选模型与输出，本轮未测最低容量。原生推理不需要 OE；重新转换的工具链和数据前提见 conversion 文档。

```bash
# cwd: repository root
python3 -m venv .venv-resnext
source .venv-resnext/bin/activate
python3 -m pip install -r samples/vision/resnext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速开始

在 X5 仓库根执行。下载成功时退出码 0 并打印观测哈希；推理成功时退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py \
  --target x5 --variant 50_32x4d \
  --test-img samples/vision/resnext/test_data/bee_eater.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

唯一发布变体为 `50_32x4d`，默认即选中它。推理打印 Top-5 ID、softmax 分数与标签。完全平局按 ID 升序排序。随附图片用于功能检查；板测 not-run。仅指定 `--img-save-path` 才写文件。

供参考：X5 源 README（rdk_x5
@ac115717197920355fc390bb04299b20e6436864，旧版 Python 入口）用下面的
截图演示运行效果：旧版 `result.jpg` 绘制把 Top-5 排名叠在图上，随附
`bee_eater.JPEG` 的 rank 1 为 class 92（bee eater）。这是源交付中的
历史截图，不是本仓库当前入口的运行结果。

![X5 源 README 的历史推理截图：bee eater 测试图与旧版 Top-5 叠加，
rank 1 为 class 92（bee eater）](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

历史源数据未在本轮重测。完整列与计时条件见[评测说明](evaluator/README_cn.md#reference-results)，单线程延迟与多线程 FPS 不能按倒数直接比较。

<a id="directory"></a>
## 目录

`model/`：制品与下载；`runtime/python/`：原生 CLI、任务与运行器；`conversion/`：1 份保留 PTQ 配置；`evaluator/`：检查与历史基准；`test_data/`：`bee_eater.JPEG` 输入及随附资源；`tests/`：主机回归。

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

旧 `platforms/x5/samples/vision/resnext` 入口仍是原始实现，供基线对照，未改成转发层。新集成使用本目录；不承诺所有旧内部 Python 导入兼容。

<a id="license"></a>
## 许可

Python 代码遵循 Apache-2.0。转换材料保留各文件原声明，上游模型/权重遵循其各自许可；这里不新增权重许可声明。
