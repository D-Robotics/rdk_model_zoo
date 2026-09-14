[English](./README.md)

# 模型评估说明

本目录说明 Paraformer 的数据格式、主机精度评测、板端 WAV 验证和已完成验证结果。Runtime 使用 FunASR `WavFrontend` 从 16 kHz 音频生成并补齐到 `[1, 400, 560]` 的 fbank+LFR+CMVN 特征。

## 评测类型

| 场景 | 入口 | 输入 | 结果 |
| --- | --- | --- | --- |
| 主机 FP32 / INT16 CER | `conversion/11_eval_pipeline.py` | 预生成的固定 shape `.npy` 特征 | `results_fp32.json` 或 `results_int16.json` |
| 板端 Python 功能验证 | `runtime/python/run.sh` | WAV + manifest | 终端识别文本与分阶段时延 |
| 板端 C++ CER / 性能验证 | `runtime/cpp/run.sh` | WAV + manifest | `results_board_ucp.json`、CER 与分阶段时延 |

主机量化评测使用转换阶段生成的固定 shape 特征；板端 Runtime 面向最终用户，从原始 WAV 开始。二者共享 FunASR 前端配置、固定 shape、CIF mask 与词表，但输入目录结构不同。

## 板端 WAV 数据格式

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    └── <utt_id>.wav
```

```json
[
  {"utt_id": "BAC009S0002W0122", "text": "参考转写"}
]
```

每个 `audio/<utt_id>.wav` 必须为 16 kHz。`runtime/python/paraformer.py` 从 WAV 自动生成 C 连续 float32 `[1, 400, 560]` 特征，并记录未 padding 的有效帧数（最大 400）；CIF 会在该长度之后屏蔽 Predictor 的 `alphas`，避免 padding 产生伪 token。C++ Runtime 由 `runtime/python/main.py --preprocess-only` 在临时目录生成同一格式特征，用户无需提供 `.npy`。

## 主机 CER 评测

`conversion/11_eval_pipeline.py` 的 `--eval_dir` 使用以下目录：

```text
<AISHELL_EVAL_DIR>/
├── manifest.json
└── feats/
    └── <utt_id>.npy
```

每个 `feats/<utt_id>.npy` 是转换阶段从真实语音生成的 C 连续 float32 `[1, 400, 560]` 特征，`manifest.json` 中需要包含 `utt_id`、`text` 和 `feat_length`。该格式仅用于 ONNX/HMCT 离线评测，不是板端 Runtime 的用户输入。

使用 `conversion/11_eval_pipeline.py` 评测：

```bash
cd conversion
python 11_eval_pipeline.py --pipeline fp32 --eval_dir <AISHELL_EVAL_DIR>
# INT16 评测需在带 HMCT ORTExecutor 的 OpenExplorer Docker 内执行
python 11_eval_pipeline.py --pipeline int16 --eval_dir <AISHELL_EVAL_DIR>
```

脚本在 `<AISHELL_EVAL_DIR>/` 下写出 `results_fp32.json` 或 `results_int16.json`，其中包含每条识别结果、参考文本、字符错误数和最终 CER。

## 板端 WAV 验证

```bash
# Python：验证 WAV 前处理和三段 HBM pipeline
cd runtime/python
N_UTT=300 bash run.sh <AUDIO_DATA_DIR>

# C++：从 WAV 生成临时特征，并计算 manifest 文本的 CER
cd runtime/cpp
N_UTT=300 bash run.sh <AUDIO_DATA_DIR>
```

Python 输出将 WAV 前处理时间与 HBM pipeline 时间分开；C++ Runtime 在 manifest 包含参考文本时计算 CER，并在 `runtime/cpp/build/eval/results_board_ucp.json` 写入结果。对内置两条样例直接执行 `bash run.sh` 即可完成冒烟验证。

## 已完成验证结果

以下为已完成的 Paraformer S100 验证记录：

| 项目 | 配置 |
| --- | --- |
| 评测集 | AISHELL dev，300 条语音、40 名说话人 |
| 输入 | 真实音频经 FunASR 前端提取并补齐为 `[1, 400, 560]` fbank+LFR 特征 |
| 评测链路 | Encoder INT16 HBM → Predictor INT16 HBM → CPU CIF → Decoder INT16 HBM → 贪心 token 解码 |
| 指标 | Character Error Rate（CER） |
| 词表 | 8,404 token 通用词表 |

| 指标 | Python `hbm_runtime` | C++ UCP |
| --- | ---: | ---: |
| CER（300 条 AISHELL dev） | **3.13%** | **3.13%** |
| Encoder | 33.63 ms | 33.15 ms |
| Predictor | 1.44 ms | 1.00 ms |
| CIF（CPU） | 3.41 ms | **0.38 ms** |
| Decoder | 7.12 ms | 6.29 ms |
| 端到端 | 45.61 ms | **40.81 ms** |
| RTF | 0.008 | 0.007 |

表中 `45.61 ms` 和 `40.81 ms` 为历史 300 条验证的 HBM pipeline 指标。当前 Python WAV Runtime 会额外显示 `frontend_ms` 和包含前处理的端到端时间，因此不应将其端到端时间与表中 HBM pipeline 指标直接比较。

结果保持依赖相同的 FunASR 前端、固定 `torch.manual_seed(191009)` 的 fbank dither、固定特征 shape、三段 INT16 HBM、tensor 连线、实际帧数 padding mask、`max_label_len=100`、全零 `bias_embed` 与特殊 token 过滤逻辑。
