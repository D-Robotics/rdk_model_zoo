# ASR — chunked speech recognition

[English](README.md) | 简体中文

<a id="overview"></a>
## 概述

本样例使用已发布的 S 系列 Wav2Vec2 ASR 模型和固定 3503-token 词表转写音频。WAV/FLAC 按有限大小分块读取，声道平均为单声道，重采样到 16 kHz，再逐块归一化，每次推理输入 30000 点（1.875 秒）。处理完整文件，包括最后补零块；这是独立窗口处理，不是带隐藏流状态或重叠拼接的声学模型。

统一 Python 流程已实现。原生流程也已实现：音频前处理、CTC/legacy、SDK 适配、显式启动器及完整文件结果报告都有主机测试；真实 SDK 构建/ABI 和模型推理尚未验证。原 S 源 (historical `../../../platforms/s/samples/speech/asr/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)保留历史实现背景；本轮没有板测或新的真实模型转写结果。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 发布身份 | 统一 Python | 统一 C++ | 板测 |
| --- | --- | --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | 已实现 | 已实现，主机验证 | not-run |
| S600 | `s:asr:s600/asr.hbm` | 已实现 | 已实现，主机验证 | not-run |
| X5 / S100P | 无 | 拒绝 | 不支持 | not-run |

发布目标以当前清单为准，不沿用源中互相矛盾的注释。S600 有发布制品不等于运行时已验收，仍需验证实际模型/SDK metadata；不提供目标回退。

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

在匹配板卡上准备对应模型并执行（以下推理命令不作为本轮板测证据）：

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/python/run.sh --target s100 --output-dir outputs/asr-run1
```

S600 将两条命令都改为 `s600`，不能将 S100 HBM 改名使用。`PYTHON` 指定解释器，输出目录必须新建。外部模型需精确的目标限定 asset ID，详见[模型准备](model/README_cn.md)。

<a id="expected-results"></a>
## 结果及解码变更

控制台打印完整文件转写及 `result.json` 路径；报告绑定模型/音频/词表摘要、实际 metadata、解码模式，以及逐块位置、有效样本数和文本。错误返回 2，处理开始后失败会写入带已完成块的 `failed.json`，不将部分结果冒充成功转写。

默认 `ctc` 先合并相邻重复 ID，**再**删除 blank ID 0。归档实现没有合并重复，只移除了 `<pad>`；可用 `--decode-mode legacy` 保留其行为做源对照。词表 `<pad>,a,b` 下，ID `[1,1,0,1,2,2]` 的 CTC 结果是 `aab`，legacy 是 `aaabb`。这是明确的解码修正，不是模型 logits 改变的证据。其他 token、标点和 `|` 原样保留；每个独立块重新开始 CTC 状态，不凭空做跨块去重。

<a id="directory"></a>
## 目录

| 位置 | 职责 |
| --- | --- |
| `model/` | 精确清单选择及显式目标下载 |
| `runtime/python/` | 音频读取、纯前处理/解码、共享 raw runner 和 CLI |
| `runtime/cpp/` | 原生音频、任务、SDK 适配、启动器与主机测试，真实 SDK 验证待完成 |
| `test_data/` | 原始 WAV、固定词表及历史图片 |
| `conversion/` | 如实记录源中缺失的导出/编译前提 |
| `evaluator/` | 已保存转写的字符错误指标与历史结果 |
| `tests/` | 主机数值/metadata/身份/报告测试 |

<a id="entry-points"></a>
## 导航

[Python 使用/API](runtime/python/README_cn.md) · [原生进展](runtime/cpp/README_cn.md) · [转换](conversion/README_cn.md) · [评估](evaluator/README_cn.md) · [输入身份](test_data/README_cn.md)。

主机源对照覆盖原始 16 kHz 中文音频、44.1 kHz 立体声和 8 kHz 常量音频，特征一致性与 SDK/板端/数据集验收分开记录；原始延迟、余弦截图均是历史记录。

<a id="license"></a>
## 许可

遵循仓库 [Apache-2.0 许可](../../../LICENSE)。保留固定 S 源版权、数据和截图；外部依赖遵循各自许可。

## 原生入口

在匹配板卡显式准备模型和依赖后，从仓库根目录执行
`bash samples/speech/asr/runtime/cpp/run.sh --target s100 --build`，S600 将目标
改为 s600。不会自动下载模型。[原生指南](runtime/cpp/README_cn.md)覆盖主机预检、
全部构建/运行参数、结果、API 示例及 Python/C++ 重采样差异。
