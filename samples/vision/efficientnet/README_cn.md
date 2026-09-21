# EfficientNet 图像分类

EfficientNet 在 RDK 板卡上的 ImageNet-1k 分类：输入一张 BGR 图像，输出稳定的 Top-K `(类别 ID, 分数, 标签)`。X5 侧发布 EfficientNet B2/B3/B4 变体（论文 [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11942)）；S100/S600 侧发布 EfficientNet-Lite lite0..lite4 系列（源交付引用的 [TensorFlow TPU EfficientNet-Lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) 实现）。[English](README.md)

<a id="overview"></a>
## 概述

统一实现是一条 Python 流程（全部目标；两个源分支均未提供该 sample 的 C++ 运行时）。Python 从平台发布 Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，执行 `pre_process → forward → post_process` 任务（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。迁移前的平台分支入口在收尾前仍以兼容 shim 形式保留在 `platforms/{x5,s}/` 下，其审计记录在迁移文档中，不在本 README 展开。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | b2、b3、b4 | python | supported（2026-09-21 x5-8g + x5-4g 板测通过，见下方说明） |
| s100 | lite0..lite4 | python | supported（2026-09-21 s100 板测通过，见下方说明） |
| s600 | lite0..lite4 | python | supported（2026-09-21 s600 板测通过，见下方说明） |
| s100p | 任意 | python | not-supported（发布 Manifest 无 s100p 资产行；选择时显式报错、无回退） |

源基线：X5 侧 rdk_x5 @ac11571 (x5-v1.1.3)；S 侧 rdk_s @380e1a2 (s-v1.1.2)。统一 sample 的主机测试（28 项）全部通过。板端冒烟（2026-09-21；同板、同制品字节、同输入图，旧 wrapper 对照统一入口）：x5-8g/x5-4g b2/b3/b4 Top-5 ids 全等（最大分差 ≤1.2e-7）；s100/s600 lite0..lite4 全等（≤1.2e-7），逐变体几何 224/240/260/300/380 解析正确；`run.sh` CLI 四板 rc=0。省略变体的默认入口按 target 解析（x5 → b2，s100/s600 → lite0；B2-R1 修复后在两块 S 板复验，显式 `--variant`/`--asset-id` 匹配不变）。s100p 选择显式报错、无回退。raw tensor 等价、数据集精度与延迟不在覆盖范围；已发布基准表仍为源分支记录。证据：[B2 板测](../../../docs/releases/unified-migration/evidence/2026-09-21-b2-board-smoke-evidence.json)。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
python3 -m pip install -r samples/vision/efficientnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:efficientnet:EfficientNet_B2_224x224_nv12.bin）
#    输出：samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin
#    成功判据：下载器退出码 0 并打印实测摘要
bash samples/vision/efficientnet/model/download.sh x5 b2

# 2. 运行分类（输入：上一步制品 + 随附测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientnet:EfficientNet_B2_224x224_nv12.bin \
  --model-path samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin \
  --test-img samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 使用对应的 `s:efficientnet:s…` 引用（见 `--list-models`）与同一
根 `datasets/imagenet/` 标签；S 侧几何随变体（lite0..lite4 = 224/240/260/300/380）。完整命令见 [runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非
指定 `--img-save-path`，不写任何输出文件。使用随附 `Scottish_deerhound.JPEG`
时 Top-5 含鹿猎犬相关 ImageNet 类别；使用 `redshank.JPEG` 时含红脚鹬相关
类别。无法识别的板卡或无匹配制品的目标会显式报错退出——特别是源 S 实现在
一切非 S600 SoC（含 S100P）上静默运行 lite0 S100 模型的行为已移除：S100P
是显式的 no-published-asset 错误。

<a id="performance"></a>
## 性能数据

发布记录，未在本仓库重测。

X5（rdk_x5 @ac11571，x5-v1.1.3）：

| 模型 | 尺寸 | 参数量 (M) | Float Top-1 | Quant Top-1 | 单线程延迟 (ms) | 多线程延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S 系列（rdk_s @380e1a2，s-v1.1.2）：

| 变体 | 单线程延迟 | 单线程 FPS | 多线程延迟 | 多线程 FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

<a id="directory"></a>
## 目录职责

- [model/](model/README_cn.md) — Manifest 驱动的制品下载，不检入二进制
- [runtime/python/](runtime/python/README_cn.md) — 统一 Python 入口与任务模块
- [conversion/](conversion/README_cn.md) — S 侧 lite 全套配方 + X5 侧参考 PTQ 配置
- [evaluator/](evaluator/README_cn.md) — 发布的基准记录与功能检查
- `test_data/` — 随附测试图（[Scottish_deerhound.JPEG](test_data/Scottish_deerhound.JPEG)、[redshank.JPEG](test_data/redshank.JPEG)、[zebra_cls.jpg](test_data/zebra_cls.jpg)）
- `tests/` — 主机 unittest 套件

<a id="entry-points"></a>
## 入口

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行时：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- 转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

样例代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为上游 EfficientNet /
EfficientNet-Lite 发行版；模型/权重许可由上游发行版 govern（见上方论文
链接）。已发布制品遵循平台发布 Manifest；Manifest 不含独立许可字段，
本文件不主张额外许可。
