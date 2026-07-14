[中文](./README_cn.md)

# Python Runtime

The Python runtime contains only two code files: `paraformer.py` and `main.py`. `paraformer.py` uses `hbm_runtime.HB_HBMRuntime` to load the Encoder, Predictor, and Decoder HBM files in one resident runtime and performs FunASR WAV preprocessing, CPU CIF, and decoding; `main.py` handles manifest scheduling and reporting. Users provide 16 kHz WAV files, and the validated Encoder → Predictor → CPU CIF → Decoder sequence remains unchanged.

## Run

```bash
cd runtime/python
bash run.sh [test_data_directory]
```

On first use, the script creates `runtime/python/.venv` and installs the validated CPU `torch/torchaudio 2.6.0`, FunASR, and SoundFile packages. The system Python environment remains unchanged. The default directory `../../test_data` contains two WAV samples. Set `N_UTT=10` to limit manifest items.

## Custom data

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    ├── <utt_id_1>.wav
    └── <utt_id_2>.wav
```

Each manifest object requires `utt_id`; `text` is optional reference text. WAV input must be 16 kHz. The program reports recognition text and average frontend, Encoder, Predictor, CPU CIF, Decoder, and HBM-pipeline latency. Frontend latency is reported separately from the validated HBM pipeline latency.

`main.py --preprocess-only` is an internal mode used by the C++ runtime. It writes temporary features and valid frame lengths from WAV input; ordinary Python users should run `run.sh` and do not need this mode.
