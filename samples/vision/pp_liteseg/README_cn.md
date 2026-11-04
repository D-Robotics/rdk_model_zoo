[English](README.md) | 简体中文

# PP-LiteSeg-STDC1 语义分割

<a id="overview"></a>
## 算法与来源

PP-LiteSeg 是轻量语义分割网络。本示例使用 STDC1 主干，对道路场景预测 Cityscapes 的 19 类语义，运行于 RDK X5。

参考：[论文](https://arxiv.org/abs/2204.02681), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

<a id="directory"></a>
## 目录结构

```text
pp_liteseg/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

| Target | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | STDC1 / Cityscapes / 1024×512 | supported | not-supported |
| s100 / s100p / s600 | 无已发布制品 | not-supported | not-supported |

<a id="prerequisites"></a>
## 环境前提

RDK X5 OS 3.5.0+、Python 3.10+、板端提供的 hbm_runtime，以及 NumPy、OpenCV、PyYAML。主机检查不加载 SDK。转换在独立容器中使用源 OE 1.2.8 配方。发布记录未提供模型大小与运行峰值内存，请为 BIN 和输出留出空间并在目标板实测；单个校准张量为 6,291,456 字节。

<a id="quickstart"></a>
## 快速体验

```bash
# cwd: repository root; prepare explicitly, then run on RDK X5
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/runtime/python/main.py
# Host-only inspection, no SDK/model/download required:
python3 samples/vision/pp_liteseg/runtime/python/main.py --dry-run --target x5
```

通用 Python 依赖安装命令为 `python3 -m pip install numpy opencv-python PyYAML`。推理不隐式下载，run.sh 仅转发参数。

<a id="expected-results"></a>
## 预期结果

成功时返回 0，输出 `outputs/pp_liteseg/result.jpg`（3078×548，原图／叠加／分割三面板），同目录的 `labels.npy`（512×1024 int32，类别 0..18）和 `result.json`。JSON/stdout 记录实际类别名和运行时元数据。类别列表由输入图片和模型输出决定。错误返回 2。掩码坐标对应拉伸后的模型输入，不是原图尺寸。

编译模型返回解码后的 int32 类别图。`postprocess` 校验类别 ID 并去掉 batch/channel 维，无需 CPU argmax。加载时会检查模型元数据。

<a id="entry-points"></a>
## 入口索引

[模型准备](model/README_cn.md) · [Python CLI 与 API](runtime/python/README_cn.md) · [转换](conversion/README_cn.md) · [验证](evaluator/README_cn.md)。

<a id="license"></a>
## 许可

仓库代码遵循顶层 [LICENSE](../../../LICENSE)，保留的源文件声明继续适用。PaddleSeg、外部预训练权重和数据各有许可；重新分发预训练权重前，请核对权重来源和许可条款。

## S100 / S100P / S600 logits 版本

新增 `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` 使用 PaddleSeg 官方训练的 STDC1 Cityscapes 权重。S 系列输入为归一化 RGB float32 NCHW `(1,3,512,1024)`，处理为 RGB/255、均值 .5、标准差 .5。HBM 输出 19 类浮点 logits，Python 运行时执行一次 CPU argmax。须选择对应转换版本或精确资产 ID。现有 X5 NV12 输入、int32 类别图输出的默认行为保持不变。

```bash
python3 samples/vision/pp_liteseg/model/download.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104
python3 samples/vision/pp_liteseg/runtime/python/main.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104 --test-img /path/to/street.png
```

指定外部 `--model-path` 前，先通过 `--list-models` 获取精确资产 ID。S100P、S600 使用各自 target 和 HBM。板端功能、Cityscapes 精度、Runtime 1/2 并发、C++ 1/2 流仍未实测；编译器估算和源模型主机校验属于转换证据。详见 `conversion/converted_targets_20261103.json` 与 `conversion/source_export_20261104.json`。
