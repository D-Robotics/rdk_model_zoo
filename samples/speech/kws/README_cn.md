# KWS — MDTC keyword spotting

[English](README.md) | 简体中文

<a id="overview"></a>
## 概述

本样例使用已发布的 MDTC 唤醒词模型处理单声道 16 kHz 音频，返回片段内模型概率的最大值，以及按阈值判断是否出现“hey snips”的结果。它不执行语音转写，也不是任意关键词模型。只使用前 60000 个采样点，短音频补零；16 kHz 下为 **3.75 秒**，修正源注释的 60 秒。长录音需由调用者明确分窗，本样例不会静默扫描整段。

### 算法与流程（MDTC）

已发布的 S100 制品实现 PaddlePaddle + PaddleAudio 语音栈中的 MDTC（Multi-Scale Dynamic Temporal Convolution）。固定源将 MDTC 描述为多尺度时序卷积模型：不同时间尺度的卷积捕获语音特征，动态卷积自适应调整卷积权重以适配不同说话人和环境，并称其设计边缘友好、检测准确率高。固定源只提供编译制品、没有训练代码，因此这些表述按源描述原样保留，不是本样例重新验证过的行为。

部署流程原地恢复如下：单声道 16 kHz float32 音频截取前 60000 个采样点（3.75 秒，短音频补零），PaddleAudio fbank 前端（25 ms 帧、10 ms 帧移、80 mel bin）把该窗口转成 `[1, 373, 80]` 特征张量，BPU 上的 MDTC 模型把窗口映射为关键词概率，后处理校验这些概率并按最大值归约为一个片段级置信度，不再叠加 sigmoid。应用层判定为 `score >= threshold`（默认 `0.5`），是可配置规则，不是校准过的误唤醒率保证。历史示例和归档实现仍可经 S 快照 (historical `../../../platforms/s/samples/speech/kws/README_cn.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)查阅。统一任务保留 preprocess → forward → postprocess → predict，音频文件、SDK 传输、特征提取和数值评分分别放在独立模块。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 发布制品 | Python | 原生 C++ | 当前验证 |
| --- | --- | --- | --- | --- |
| S100 | `s:kws:s100/kws.hbm` | 已实现 | 源中没有实现 | 仅主机前处理/契约；板测 not-run |
| X5 / S100P / S600 | 无 | 显式拒绝 | 无 | 不支持 |

板卡身份不证明模型精度类型或 SDK 兼容性。虽然归档 wrapper 的 docstring 声称更广支持，这里不提供 S600 回退。加载时核验模型 metadata；主机测试的合成 metadata 不冒充板端实测描述符。

<a id="prerequisites"></a>
## 前提

需要 Python 3.10+、NumPy、PyYAML、SoundFile、PaddlePaddle 和 PaddleAudio。完整推理还需匹配 S100 BSP 的 `hbm_runtime` 及已下载模型；不要安装 PyPI 的同名无关包。[运行说明](runtime/python/README_cn.md)列出实际主机前处理版本和显式依赖安装步骤。主机 help/list/dry-run 需要 NumPy/PyYAML，不需要 SDK 或 Paddle；推理不会安装依赖或下载文件。

<a id="quickstart"></a>
## 快速开始

在仓库根目录执行，先在任意主机检查选择：

```bash
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

在 S100 上显式准备依赖后，先下载再运行（本轮未执行为板测）：

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

成功退出码为 0，写入 `result.json`：分数、阈值、`detected`、实际 SDK metadata、输入/模型摘要及补零/截断数量。判定条件为 `score >= threshold`，默认 0.5；这是可配置应用规则，不是校准过的误唤醒率保证。随附“hey snips”音频在源 S100 实现中的历史分数约为 0.985，本迁移**未重测**该数值。失败退出码为 2，不把旧报告当作本次成功。

<a id="directory"></a>
## 目录

| 位置 | 作用 |
| --- | --- |
| `model/` | 按清单显式下载及制品身份 |
| `runtime/python/` | 纯任务阶段、前处理、音频读取、共享 SDK runner 和 CLI |
| `test_data/` | 原始 2.5 秒单声道音频及摘要/来源说明 |
| `conversion/` | 缺失配方的真实前提，不虚构编译命令 |
| `evaluator/` | 离线有标签分数指标和历史性能 |
| `tests/` | 无真实 SDK 的模型/特征/评分/异常测试 |

<a id="entry-points"></a>
## 导航与验证

- [模型准备](model/README_cn.md)
- [Python 运行、参数及 API](runtime/python/README_cn.md)
- [转换可用性](conversion/README_cn.md)
- [指标与历史性能](evaluator/README_cn.md)
- [测试音频](test_data/README_cn.md)

真实 PaddleAudio 前处理对照覆盖随附片段、静音和截断长音频。SDK 描述符、板端分数、真实延迟和数据集准确率仍为 not-run，主机测试不代表独立迁移验收已关闭。

<a id="license"></a>
## 许可

遵循仓库 [Apache-2.0 许可](../../../LICENSE)；原始版权保留于归档 S 实现，外部依赖遵循各自许可。
