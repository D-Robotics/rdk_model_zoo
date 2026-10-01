# ResNet18 图像分类

在 RDK 板卡上运行 ResNet18 ImageNet-1k 分类：输入一张 BGR 图像，输出稳定的
Top-K `(类别 ID, 分数, 标签)`。源模型为 TorchVision ResNet18
（[上游链接](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)）；
本 sample 是仓库的单模型分类参照。[English README](README.md)

<a id="overview"></a>
## 概述

维护实现为一条 Python 流程（全部 target）加一条 S 系列 C++ 流程。Python 从
平台发布 Manifest 解析唯一制品引用，校验板卡身份，懒加载 `hbm_runtime`，
执行 `pre_process → forward → post_process` 任务（见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。C++ 保留审计过的
S 系列 `hbDNNInferV2` 实现（见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)）。
原 X5 与 S18 的 Python 入口仍可用，作为调用同一 canonical 实现的兼容
wrapper；其审计记录在迁移文档中，不在本文重复。

### 算法背景

ResNet 由 Kaiming He、Xiangyu Zhang、Shaoqing Ren 和 Jian Sun 提出。残差
学习通过快捷连接让每个块只学习残差而非完整映射，稳定深层网络优化，避免
普通层堆叠随深度增加出现的退化问题
（[论文](https://arxiv.org/abs/1512.03385)、
[torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)）。

源交付中保留的变体说明（X5 rdk_x5 @ac11571；S rdk_s @380e1a2 的
resnet18/resnet50/resnet152，B1 并入本 sample）：

- **resnet18** — 轻量残差变体；S 交付将其定位为快速分类验证模型。
- **resnet50** — 瓶颈残差块（`1x1 → 3x3 → 1x1`）在受控计算量下构建更深的网络。
- **resnet152** — 152 层设计以更多计算换取更强的特征表达能力。

![ResNet 残差块](./test_data/ResNet_architecture.png)

*恢复自源交付（同一文件在 rdk_x5 @ac11571 中为
`test_data/ResNet_architecture.png`，在 rdk_s @380e1a2 中为
`test_data/resnet_architecture.png`，sha256 `cebea796…`）：ResNet-18/34
的残差基础块（左，两个 3×3 卷积）与 ResNet-50/101/152 的瓶颈构建块
（右，1×1 → 3×3 → 1×1），即 ResNet 论文图 5。*

<a id="support-matrix"></a>
## 支持与实测矩阵

| Target | Variant | 语言 | 状态 |
| --- | --- | --- | --- |
| x5 | resnet18 | python | supported-verified（X5 双板，2026-09-17） |
| s100 | resnet18 | python | supported-verified（S100，2026-09-17） |
| s100 | resnet18 | cpp | supported-verified（S100 构建+运行，Top-5 与源基线一致，2026-09-17） |
| s600 | resnet18 | python | supported-not-run（不在 B1 冒烟集内；板卡已于 2026-09-21 恢复） |
| s600 | resnet18 | cpp | supported-not-run（不在 B1 冒烟集内；板卡已于 2026-09-21 恢复） |
| s100 | resnet50 | python | supported-verified（S100 板端冒烟，2026-09-21） |
| s600 | resnet50 | python | supported-verified（S600 板端冒烟，2026-09-21） |
| s100 | resnet152 | python | supported-verified（S100 板端冒烟，2026-09-21） |
| s600 | resnet152 | python | supported-verified（S600 板端冒烟，2026-09-21） |
| s100p | 任意 | python、cpp | not-supported（发布 Manifest 无 ResNet 资产行） |

验证证据：[2026-09-17 集成评审](../../../docs/releases/unified-migration/2026-09-17-integration-review.md)。
ResNet50/152 在 B1 从 S 分支并入本 sample（未发布 X5 制品）；通过
`s:resnet50`/`s:resnet152` Manifest 行解析，运行同一流程并指定
`--variant resnet50`/`--variant resnet152`。新变体板端冒烟已在 S100 与 S600 通过（2026-09-21）：Top-1 zebra，分数张量与源实现一致（maxdiff ≤1.2e-7）。证据：[B1 板端冒烟](../../../docs/releases/unified-migration/evidence/2026-09-21-b1-board-smoke-evidence.json)。

<a id="prerequisites"></a>
## 环境前提

板端 Python 运行需要镜像中与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；只有真正执行模型时才导入 SDK（`--help`、`--list-models`、
`--dry-run` 与主机测试均不需要 SDK）。读取 Manifest 还需要 PyYAML。开发主机
可从 `requirements-host.txt` 安装用户态依赖：

```bash
# cwd：仓库根目录 — 成功判据：打印 "host dependencies: ok"
python3 -m venv .venv-resnet
source .venv-resnet/bin/activate
python3 -m pip install -r samples/vision/resnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

C++ 构建需要 CMake、C++17 编译器、OpenCV/gflags/fmt 开发包和 Horizon DNN
头文件/库，详见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)。模型转换
在 x86 OpenExplore 环境完成，不在板端，见
[conversion/README_cn.md](conversion/README_cn.md)。

<a id="quickstart"></a>
## 快速体验

在 X5 板上的一条完整路径，命令均在仓库根目录执行。前置条件：含
`hbm_runtime` 的板端镜像和可访问 Manifest 模型服务器的网络。

```bash
# 1. 准备制品（输入：Manifest 行 x5:resnet:resnet18_224x224_nv12.bin）
#    输出：samples/vision/resnet/model/resnet18_224x224_nv12.bin
#    成功判据：下载器退出码 0 并打印观察 digest
bash samples/vision/resnet/model/download.sh x5

# 2. 运行分类（输入：上一步制品与随仓测试图）
#    输出：stdout 上的 Top-5 类别 ID、分数、标签
#    成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 使用对应的 `s:resnet18:<target>/...` 引用与
同一根目录 `datasets/imagenet/` 标签；C++ 流程执行
`bash samples/vision/resnet/runtime/cpp/run.sh`。完整命令见
[runtime/python/README_cn.md](runtime/python/README_cn.md) 与
[runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)。

<a id="expected-results"></a>
## 预期结果

Python 运行打印稳定的 Top-K（默认 5）类别 ID、分数与标签并退出 0；除非
给定 `--img-save-path`，不写任何输出文件。2026-09-17 板端对照记录了
canonical 运行与旧入口在 X5 双板与 S100 上结果一致（同制品、同图、同缩放
模式下类别 ID 与 raw 分数相同），见上方链接的集成评审。C++ 二进制按标签
文件逐行打印 Top-K。无法识别的板卡或无匹配制品的 target 会报错退出，
不进行猜测。

从 S 源交付恢复的历史结果截图（rdk_s @380e1a2，每个变体一份交付；源文档
未注明截图产自哪块板）：均展示旧 S 运行时对随仓
[zebra_cls.jpg](test_data/zebra_cls.jpg) 的结果 — zebra，类别 ID 340 —
记录的 Top-1 置信度分别为 0.9985（resnet18）、0.9956（resnet50）、
0.9649（resnet152）。这些截图记录的是源交付的结果，不是本统一 sample
的运行结果。

![ResNet18 源结果](./test_data/result_resnet18_s.png)
![ResNet50 源结果](./test_data/result_resnet50_s.png)
![ResNet152 源结果](./test_data/result_resnet152_s.png)

<a id="performance"></a>
## 性能数据

X5 源版本（rdk_x5 @ac11571，x5-v1.1.3）发布的 ResNet18 记录，本仓库未
重新实测。源文档未注明延迟/FPS 的线程条件；同一数字保留在
[评估记录](evaluator/README_cn.md)中。S 源版本（rdk_s @380e1a2）未发布
ResNet18/50/152 的延迟或精度数据，此处不做推断。

| 模型 | 尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ |

![推理结果](./test_data/inference.png)

*X5 源版本的历史推理截图（rdk_x5 @ac11571，`test_data/inference.png`，
sha256 `16c9d04e…`）：随仓 [white_wolf.JPEG](test_data/white_wolf.JPEG)
的 Rank-1 为 `white wolf`，其后依次为 Arctic fox、timber wolf、Samoyed、
polar bear。由源版本在其自身运行入口记录 — 不是本仓库的新运行。*

<a id="directory"></a>
## 目录职责

- [model/](model/README_cn.md) — Manifest 驱动的制品下载，不提交二进制
- [runtime/python/](runtime/python/README_cn.md) — canonical Python 入口与任务模块
- [runtime/cpp/](runtime/cpp/README_cn.md) — S 系列 C++ 源码、CMake、启动脚本
- [conversion/](conversion/README_cn.md) — ONNX 导出与 OE 转换记录
- [evaluator/](evaluator/README_cn.md) — 主机检查与功能性板端检查
- `test_data/` — 随仓测试图（[white_wolf.JPEG](test_data/white_wolf.JPEG)、[zebra_cls.jpg](test_data/zebra_cls.jpg)）
- `tests/` — 主机 unittest 套件

<a id="entry-points"></a>
## 入口索引

- 模型准备：[model/README_cn.md](model/README_cn.md)
- Python 运行：[runtime/python/README_cn.md](runtime/python/README_cn.md)
- C++ 运行（S100/S600）：[runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)
- 模型转换：[conversion/README_cn.md](conversion/README_cn.md)
- 评估：[evaluator/README_cn.md](evaluator/README_cn.md)

<a id="license"></a>
## 许可

sample 代码遵循仓库顶层 LICENSE（Apache-2.0）。源模型为 TorchVision
ResNet18，其模型/权重许可由 TorchVision 发行版约定（见上游链接）。已发布
制品以平台发布 Manifest 为准；Manifest 未携带独立许可字段，本文不主张额外
许可。

<a id="readable-example"></a>
## 可读范例与自定义模型

本 sample 是两个可读模型范例之一：完整分类流程（初始化、`preprocess`、
`infer`、`postprocess`、`predict`）在
[`runtime/python/classify.py`](runtime/python/classify.py) 中可见；
`main.py` 保持薄入口（解析参数、构造 `ResNetClassifier`、调用 `predict`、
展示结果）。自训练分类模型经 `model_binding.custom_selection` 接入，无需
Manifest 注册；predict 接受图片路径或 BGR 数组。官方模型运行、自训练权重
接入、修改业务调用三条路径见
[docs/architecture/model-examples.md](../../../docs/architecture/model-examples.md)；
旧新接口映射见
[docs/migration/2026-09-30-model-examples.md](../../../docs/migration/2026-09-30-model-examples.md)。
