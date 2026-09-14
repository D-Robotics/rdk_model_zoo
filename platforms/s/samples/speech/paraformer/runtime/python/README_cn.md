[English](./README.md)

# Python Runtime

Python Runtime 仅包含 `paraformer.py` 与 `main.py` 两个代码文件。`paraformer.py` 使用 `hbm_runtime.HB_HBMRuntime` 在一个常驻 runtime 中加载 Encoder、Predictor 和 Decoder HBM，并完成 FunASR WAV 前处理、CPU CIF 和解码；`main.py` 负责 manifest 调度与统计。用户输入为 16 kHz WAV，推理顺序保持 Encoder → Predictor → CPU CIF → Decoder。

## 运行

```bash
cd runtime/python
bash run.sh [test_data_directory]
```

首次运行时，脚本会在 `runtime/python/.venv` 中自动安装已验证的 CPU `torch/torchaudio 2.6.0`、`funasr` 与 `SoundFile`，不会修改系统 Python 包。默认数据目录为 `../../test_data`，其中含两条 WAV 样例。使用 `N_UTT=10 bash run.sh` 可限制处理 manifest 记录数。

## 自定义数据

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    ├── <utt_id_1>.wav
    └── <utt_id_2>.wav
```

`manifest.json` 的每条记录至少包含 `utt_id`，可选 `text` 作为参考转写；WAV 必须为 16 kHz。程序会打印识别文本，以及 Frontend、Encoder、Predictor、CPU CIF、Decoder 和 HBM pipeline 的平均延迟。`frontend` 时间单列展示，不计入已验证的 HBM pipeline 时延。

`main.py --preprocess-only` 是供 C++ Runtime 调用的内部模式：它把 WAV 临时转换为特征并补充 `feat_length`。普通 Python 用户只需执行 `run.sh`，无需调用该模式。
