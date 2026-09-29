[English](README.md) | 简体中文

# PP-LiteSeg-STDC1 语义分割

<a id="overview"></a>
## 算法与来源

PP-LiteSeg-STDC1 为每个像素预测 Cityscapes 道路场景的 19 类语义。本示例统一 X5 Python 推理、转换配方与单图验证入口。沿用源分支提供的算法资料：[论文](https://arxiv.org/abs/2204.02681)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)。

源分支的**运行时代码**接收已经解码的 int32 类别图；旧根目录和转换 README 中“logits 再做 CPU argmax”的表述与代码不符。本实现的 post_process 仅校验并去掉 batch/channel 维。实际编译制品的元数据仍需板端验证。

<a id="support-matrix"></a>
## 支持与验证矩阵

| Target | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | STDC1 / Cityscapes / 1024×512 | supported-not-run | not-supported |
| s100 / s100p / s600 | 无已发布制品 | not-supported | not-supported |

主机 fixture 验证源预处理、类别图解码、可视化、选择逻辑和 CLI 行为，不代表 SDK 或已发布 BIN 已经验证。板测、数据集指标及转换执行均为 not-run。[源审计证据](../../../docs/releases/unified-migration/evidence/2026-09-26-b8-ppliteseg-audit.json)。

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

成功时返回 0，输出 `outputs/pp_liteseg/result.jpg`（3078×548，原图／叠加／分割三面板），同目录的 `labels.npy`（512×1024 int32，类别 0..18）和 `result.json`。JSON/stdout 记录实际类别名和运行时元数据。未完成真实推理前，不承诺 street 图片的具体类别列表或精度。错误返回 2。mask 坐标对应拉伸后的模型输入，不是原图尺寸。

<a id="directory"></a>
## 目录职责

- `model/`：显式准备发布制品，仓库不内置权重。
- `runtime/python/`：四阶段任务、绑定、SDK runner、CLI 与可视化。
- `conversion/`：PaddleSeg 导出、原始校准数据、OE YAML 和编译脚本。
- `evaluator/`：单图兼容入口与验证范围。
- `test_data/`：源文件 `street.png` 与 `test.jpg`；旧文档中的 `street.jpg` 不存在。
- `tests/`：主机 fixture 与源行为对照。

<a id="entry-points"></a>
## 入口索引

[模型准备](model/README_cn.md) · [Python CLI 与 API](runtime/python/README_cn.md) · [转换](conversion/README_cn.md) · [验证](evaluator/README_cn.md)。原 X5 快照仍保留在 [platforms/x5](../../../platforms/x5/samples/vision/pp_liteseg/README_cn.md)。

<a id="license"></a>
## 许可

仓库代码遵循顶层 [LICENSE](../../../LICENSE)，保留的源文件声明继续适用。PaddleSeg、外部预训练权重和数据各有许可；manifest 不能证明预训练权重的许可，重新分发前请核对实际 checkpoint 来源。
