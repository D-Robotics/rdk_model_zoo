# MobileNetV1 图像分类

MobileNetV1 在 RDK 板卡上的 ImageNet-1k 分类：输入一张 BGR 图像，输出稳定的 Top-K `(类别 ID, 分数, 标签)`。源模型：[tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)，论文 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)。[English](README.md)

<a id="overview"></a>
## 概述

统一实现是一条 Python 流程（全部目标）。Python 从平台发布 Manifest 解析唯一的制品引用，核验板卡身份，懒加载 `hbm_runtime`，执行 `pre_process → forward → post_process` 任务（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)）。迁移前的平台分支入口在收尾前仍以兼容 shim 形式保留在 `platforms/{x5,s}/` 下，其审计记录在迁移文档中，不在本 README 展开。

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | 变体 | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | mobilenetv1 | python | supported-verified（x5 8GB + 4GB 板端冒烟，2026-09-21） |
| s100 | mobilenetv1 | python | supported-verified（S100 板端冒烟，2026-09-21） |
| s600 | mobilenetv1 | python | supported-verified（S600 板端冒烟，2026-09-21） |
| s100p | 任意 | python、cpp | not-supported（发布 Manifest 无 s100p 资产行；2026-09-21 在 S100P 实板验证为拒绝负例——显式报错、无回退） |

源基线：X5 侧 rdk_x5 @ac11571 (x5-v1.1.3)；S 侧 rdk_s @380e1a2 (s-v1.1.2)。统一 sample 的主机测试全部通过。板端冒烟（2026-09-21）在 x5 8GB/4GB 与 S100/S600
全部通过，各板输出一致且与源实现等价；S100P 仅作为拒绝负例验证。证据：[B1 板端冒烟](../../../docs/releases/unified-migration/evidence/2026-09-21-b1-board-smoke-evidence.json)。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：输出 "host dependencies: ok"
python3 -m venv .venv-mobilenetv1
source .venv-mobilenetv1/bin/activate
python3 -m pip install -r samples/vision/mobilenetv1/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 板卡上的一条完整路径（命令在仓库根目录执行）。前置条件：带
`hbm_runtime` 的板卡镜像与可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:mobilenetv1:mobilenetv1_224x224_nv12.bin）
#    输出：samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin
#    成功判据：退出码 0 并打印观察 digest
bash samples/vision/mobilenetv1/model/download.sh x5

# 2. 运行分类（输入：上一步制品与内置测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv1/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv1:mobilenetv1_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv1/model/mobilenetv1_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv1/test_data/bulbul.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 换用对应的 `s:` 引用（见 `--list-models`）与根目录
`datasets/imagenet/` 标签。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非给出
`--img-save-path`，否则不写任何输出文件。X5 上使用内置测试图
`bulbul.JPEG` 时，Top-1 应与图像主体（一只黄鹎（鸟类））一致；
S100/S600 上使用 `zebra_cls.jpg` 时，Top-5 应包含 `zebra`。无法识别的板卡或
无对应制品的目标会以错误退出，不做猜测。

<a id="performance"></a>
## 性能数据

下表为 rdk_x5 @ac11571 (x5-v1.1.3) 发布的 MobileNetV1 在 `RDK X5` 上的公开数据：

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV1 | 224x224 | 1000 | 4.2 | 71.7% | 65.4% | 0.58 | 2800+ |

S 侧源发布（rdk_s @380e1a2 (s-v1.1.2)）未公布该模型的延迟/精度数据，此处不推断、不补造。

<a id="directory"></a>
## 目录职责

- [model/](model/README_cn.md) — Manifest 驱动的制品下载，不提交二进制
- [runtime/python/](runtime/python/README_cn.md) — canonical Python 入口与任务模块
- [conversion/](conversion/README_cn.md) — 转换记录与参考配置
- [evaluator/](evaluator/README_cn.md) — 公开基准与功能检查
- `test_data/` — 内置测试图（[bulbul.JPEG](test_data/bulbul.JPEG)、[zebra_cls.jpg](test_data/zebra_cls.jpg)）
- `tests/` — 主机 unittest 套件

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
Manifest；Manifest 未携带独立许可字段，此处不追加声明。
