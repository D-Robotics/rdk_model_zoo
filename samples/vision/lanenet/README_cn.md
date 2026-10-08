[English](README.md) | 简体中文

# LaneNet：车道二值标签与嵌入特征

<a id="overview"></a>
## 概述

LaneNet 通过二值分割分支预测车道像素，通过嵌入分支学习区分车道实例的特征。本示例提供 RDK S100 的 Python 与 C++ 推理，输出二值标签和原始嵌入。

参考：[Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591), [MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection).

<a id="directory"></a>
## 目录结构

```text
lanenet/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 已发布模型 | Python / C++ | 状态 |
| --- | --- | --- | --- |
| S100 | `s100/lanenet256x512.hbm` | Python 和 C++ | supported |
| X5 / S100P / S600 | 无 LaneNet 资产 | 显式拒绝 | 不静默回退至 S100 |

使用 S100 对应的发布 HBM。`auto` 选择 S100；实际推理检查当前板卡身份。

<a id="prerequisites"></a>
## 前置条件

实际推理需要匹配的 S100 运行环境与已准备好的 HBM。Python 需要 NumPy、OpenCV 和板端 `hbm_runtime`；C++ 需要匹配的 DNN/UCP 开发头文件与库、CMake、C++17 编译器以及 OpenCV 开发库。详见 [Python 环境](runtime/python/README_cn.md#environment)和[原生依赖](runtime/cpp/README_cn.md#dependencies)。

下载与原生构建均需显式执行，运行包装入口不自动安装依赖或拉取资产。使用已发布模型无需自行转换；源导出代码不完整，详见[转换说明](conversion/README_cn.md)。

<a id="quickstart"></a>
## 快速开始

从仓库根目录执行。以下命令检查清单与选择结果，不需要板卡，不导入 SDK，也不下载：

```bash
python3 -m samples.vision.lanenet.runtime.python.main --list-models
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --dry-run
```

网络可用时显式下载模型：

```bash
bash samples/vision/lanenet/model/download.sh --target s100
```

在 S100 运行环境中使用新目录执行 Python 推理：

```bash
bash samples/vision/lanenet/runtime/python/run.sh --target s100 --output outputs/lanenet_python_first
```

原生推理需要先准备开发依赖，再显式构建并运行：

```bash
bash samples/vision/lanenet/runtime/cpp/run.sh --target s100 --build --output outputs/lanenet_cpp_first
```

<a id="expected-results"></a>
## 预期结果

输入图像通过 INTER_AREA 拉伸到 512×256，BGR 转 RGB，除以 255 后按 ImageNet 均值与标准差归一化。模型物理输入为 float32 NCHW `[1,3,256,512]`。输出保持模型网格，不恢复到输入图像分辨率。

两种入口均写出 `embedding.npy`（float32 CHW）、`binary.npy`（uint8，标签 0/1）、`instance_pred.png`、`binary_pred.png` 和 `report.json`。Python 在 `raw_outputs.npz` 中保留全部具名原始输出，并记录名称与归档键的对应关系；C++ 写出 `raw_output_N.npy` 并记录实际元数据和角色索引，其启动器额外记录摘要与完整输出流。比较结果前请阅读相应运行说明。

嵌入 PNG 将特征裁剪至 [0,1]，乘 255 后舍入。它明确替换源 Python 的溢出回绕/截断显示行为，原始嵌入不变。二值 PNG 用 0/255 显示标签。不输出聚类结果、车道 ID、跟踪、曲线拟合、数据集精度或延迟。

以下图片逐字节保留自原 S sample：

| 源记录 Python 嵌入显示图 | 源记录 Python 二值显示图 |
| --- | --- |
| ![源嵌入显示](test_data/instance_pred.png) | ![源二值显示](test_data/binary_pred.png) |

另保留[源原生嵌入显示图](test_data/cpp_instance_pred.png)和[原生二值显示图](test_data/cpp_binary_pred.png)。显示差异本身不能证明原始数值不同，也不能证明已分离出不同车道实例。

车道实例 ID 与拟合曲线需对嵌入执行聚类及曲线拟合。显示颜色本身不代表车道实例身份。

<a id="entry-points"></a>
## 用户与 Agent 的入口

应用集成先构造 `LaneNetSegmenter`，再调用 `predict`。`lanenet.py` 实现 `preprocess`、`infer`、`postprocess` 及模型初始化；初始化时加载 Runtime。CLI 负责下载命令、文件读写与绘图；C++ 接口见原生运行指南。

修改前处理或增加实例聚类前先阅读[阶段 IO 契约](runtime/python/README_cn.md#stage-io)。聚类属于额外的算法能力，需要单独实现并验证；实例掩码应以聚类算法的输出为准。

<a id="license"></a>
## 许可证

本 sample 遵循仓库 [Apache-2.0 许可证](../../../LICENSE)。算法参考、上游 checkpoint 和外部下载资产遵循各自适用条款；授权与来源以其发布方记录为准。
