# ASR — chunked speech recognition

[English](README.md) | 简体中文

<a id="overview"></a>
## 概述

本样例使用已发布的 S 系列 Wav2Vec2 ASR 模型和固定 3503-token 词表转写音频。WAV/FLAC 按有限大小分块读取，声道平均为单声道，重采样到 16 kHz，再逐块归一化，每次推理输入 30000 点（1.875 秒）。处理完整文件，包括最后补零块；每个窗口独立处理并重置解码状态，运行时不跨窗口保留声学状态或执行重叠拼接。

提供 Python 与 C++ 运行时，均可对完整音频文件转写。Python 运行时支持 CTC 与
legacy 解码；原生入口使用匹配板端 SDK 构建和运行。

<a id="directory"></a>
## 目录结构

```text
asr/
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

| 目标 | 发布制品 | Python 运行时 | C++ 运行时 |
| --- | --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | 可用 | 可用 |
| S600 | `s:asr:s600/asr.hbm` | 可用 | 可用 |
| X5 / S100P | 无 | 不支持 | 不支持 |

为板卡选择相同 target 发布的制品。运行时在推理前核对板卡身份和模型张量
metadata。

<a id="prerequisites"></a>
## 前提

需要 Python 3.10+、NumPy、PyYAML、SoundFile、SciPy，以及匹配 BSP 的板端 `hbm_runtime`。help/list/dry-run 不导入 SDK、SoundFile 或 SciPy。不要安装 PyPI 的同名无关 `hbm_runtime`。模型和依赖均显式准备，详见[运行环境](runtime/python/README_cn.md)。

<a id="quickstart"></a>
## 快速开始

在仓库根目录，可先在任意主机查看目标：

```bash
bash samples/speech/asr/runtime/python/run.sh --list-models
bash samples/speech/asr/runtime/python/run.sh --target s100 --dry-run
bash samples/speech/asr/runtime/python/run.sh --target s600 --dry-run
```

在匹配板卡上准备对应模型并执行：

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/python/run.sh --target s100 --output-dir outputs/asr-run1
```

S600 将两条命令都改为 `s600`，不能将 S100 HBM 改名使用。`PYTHON` 指定解释器，输出目录必须新建。外部模型需精确的目标限定 asset ID，详见[模型准备](model/README_cn.md)。

<a id="expected-results"></a>
## 输出与解码方式

控制台打印完整文件转写及 `result.json` 路径；报告绑定模型/音频/词表摘要、模型
metadata、解码模式，以及逐块位置、有效样本数和文本。错误返回 2；处理开始后失败会
写入带已完成块的 `failed.json`。

默认 `ctc` 先合并相邻重复 ID，再删除 blank ID 0；`--decode-mode legacy` 删除 `<pad>` 并保留重复 ID。词表 `<pad>,a,b` 下，ID `[1,1,0,1,2,2]` 的结果分别为 `aab`（`ctc`）和 `aaabb`（`legacy`）。其他 token、标点及 `|` 按原文保留。每个独立音频块重新开始 CTC 状态。

<a id="entry-points"></a>
## 导航

[Python 使用/API](runtime/python/README_cn.md) · [原生进展](runtime/cpp/README_cn.md) · [转换](conversion/README_cn.md) · [评估](evaluator/README_cn.md) · [输入身份](test_data/README_cn.md)。

Python 音频前端读取 WAV/FLAC、将多声道混合为单声道并重采样到 16 kHz。
[C++ 指南](runtime/cpp/README_cn.md)说明原生音频前处理和张量要求。

<a id="license"></a>
## 许可

遵循仓库 [Apache-2.0 许可](../../../LICENSE)。保留固定 S 源版权、数据和截图；外部依赖遵循各自许可。

## 原生入口

在匹配板卡显式准备模型和依赖后，从仓库根目录执行
`bash samples/speech/asr/runtime/cpp/run.sh --target s100 --build`，S600 将目标
改为 s600。不会自动下载模型。[原生指南](runtime/cpp/README_cn.md)覆盖主机预检、
全部构建/运行参数、结果、API 示例及 Python/C++ 重采样差异。
