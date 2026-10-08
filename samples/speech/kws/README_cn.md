# KWS — MDTC keyword spotting

[English](README.md) | 简体中文

<a id="overview"></a>
## 概述

本样例使用已发布的 MDTC 唤醒词模型处理单声道 16 kHz 音频，返回片段内模型概率的最大值，以及按阈值判断是否出现“hey snips”的结果；它不转写语音或检测任意关键词。只使用前 60000 个采样点，短音频补零；16 kHz 下为 **3.75 秒**。长录音需由调用者明确分窗。

### 算法与流程（MDTC）

已发布的 S100 制品实现 PaddlePaddle + PaddleAudio 语音栈中的 MDTC（Multi-Scale Dynamic Temporal Convolution）。其公开描述采用多尺度时序卷积捕获不同时间尺度的语音特征，并通过动态卷积适配不同说话人与环境。

部署流程：单声道 16 kHz float32 音频截取前 60000 个采样点（3.75 秒，短音频补零），PaddleAudio fbank 前端（25 ms 帧、10 ms 帧移、80 mel bin）生成 `[1, 373, 80]` 特征张量，BPU 上的 MDTC 模型输出关键词概率，后处理取最大值而不叠加 sigmoid。应用层判定为 `score >= threshold`（默认 `0.5`）。应使用带标签的正负音频为实际场景设定阈值。任务实现 preprocess → infer → postprocess 并由 predict 串联；音频文件、SDK 传输、特征提取和数值评分位于独立模块。

<a id="directory"></a>
## 目录结构

```text
kws/
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

| 目标 | 发布制品 | Python | 原生 C++ |
| --- | --- | --- | --- |
| S100 | `s:kws:s100/kws.hbm` | 可用 | 未提供 |
| X5 / S100P / S600 | 无 | 不支持 | 未提供 |

使用 S100 MDTC 制品和匹配系统镜像提供的 SDK；加载时按模型绑定核对实际张量元数据。

<a id="prerequisites"></a>
## 前提

需要 Python 3.10+、NumPy、PyYAML、SoundFile、PaddlePaddle 和 PaddleAudio。推理还需匹配 S100 BSP 的 `hbm_runtime` 及已准备模型；按[运行说明](runtime/python/README_cn.md)安装前端依赖。help/list/dry-run 需要 NumPy/PyYAML，不需要 SDK 或 Paddle；推理不会安装依赖或下载文件。

<a id="quickstart"></a>
## 快速开始

在仓库根目录执行，先在任意主机检查选择：

```bash
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

在 S100 上显式准备依赖后，先下载再运行：

```sh
bash samples/speech/kws/model/download.sh --target s100
bash samples/speech/kws/runtime/python/run.sh --target s100 --output-dir outputs/kws-run1
```

每次使用新结果目录；`PYTHON=/path/to/python` 可指定脚本解释器。已有外部模型需同时传入精确发布身份：

```sh
bash samples/speech/kws/runtime/python/run.sh --target s100 \
  --asset-id s:kws:s100/kws.hbm --model-path /path/to/kws.hbm \
  --audio-file /path/to/mono-16k.wav --output-dir outputs/kws-run2
```

<a id="expected-results"></a>
## 结果判读

成功退出码为 0，写入 `result.json`：分数、阈值、`detected`、实际 SDK metadata、输入/模型摘要及补零/截断数量。判定条件为 `score >= threshold`，默认 0.5。源 S100 对随附“hey snips”片段的记录分数约为 0.985。失败退出码为 2。

<a id="entry-points"></a>
## 导航

- [模型准备](model/README_cn.md)
- [Python 运行、参数及 API](runtime/python/README_cn.md)
- [转换可用性](conversion/README_cn.md)
- [指标与历史性能](evaluator/README_cn.md)
- [测试音频](test_data/README_cn.md)

<a id="license"></a>
## 许可

Sample 代码遵循仓库 [Apache-2.0 许可](../../../LICENSE)；外部依赖保留各自的许可与版权声明。
