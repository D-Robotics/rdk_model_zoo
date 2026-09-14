[简体中文](./README_cn.md)

# Model Evaluation

This directory documents Paraformer data layouts, host CER evaluation, board-side WAV validation, and completed validation results. The runtime uses FunASR `WavFrontend` to create fbank+LFR+CMVN features from 16 kHz audio and pad them to `[1, 400, 560]`.

## Evaluation Types

| Scenario | Entry point | Input | Result |
| --- | --- | --- | --- |
| Host FP32 / INT16 CER | `conversion/11_eval_pipeline.py` | Pre-generated fixed-shape `.npy` features | `results_fp32.json` or `results_int16.json` |
| Board Python functional validation | `runtime/python/run.sh` | WAV + manifest | Decoded text and stage latency in the terminal |
| Board C++ CER / performance validation | `runtime/cpp/run.sh` | WAV + manifest | `results_board_ucp.json`, CER, and stage latency |

Host quantization evaluation consumes conversion-stage fixed-shape features, while board runtimes start from raw WAV input. They share the FunASR frontend configuration, fixed shape, CIF mask, and vocabulary, but use different input layouts.

## Board WAV Data Format

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    └── <utt_id>.wav
```

Each manifest object contains `utt_id` and optional reference `text`. Each `audio/<utt_id>.wav` must be 16 kHz. `runtime/python/paraformer.py` creates C-contiguous float32 `[1, 400, 560]` features and records the unpadded frame count; CIF masks Predictor alphas after that count to prevent padding from generating false tokens. The C++ runtime invokes `runtime/python/main.py --preprocess-only` to generate the same temporary bridge features, so users never provide `.npy` input.

## Host CER Evaluation

`conversion/11_eval_pipeline.py` expects this `--eval_dir` layout:

```text
<AISHELL_EVAL_DIR>/
├── manifest.json
└── feats/
    └── <utt_id>.npy
```

Each `feats/<utt_id>.npy` is a C-contiguous float32 `[1, 400, 560]` tensor generated from real audio during conversion. Each manifest object must include `utt_id`, `text`, and `feat_length`. This layout is only for offline ONNX/HMCT evaluation, not board-runtime user input.

Use `conversion/11_eval_pipeline.py` for FP32 or INT16 CER evaluation. Run the INT16 path in OpenExplorer Docker with HMCT ORTExecutor. Python and C++ runtimes process the same manifest and report decoded text and per-stage latency; Python reports WAV frontend time separately from HBM pipeline time, and C++ also computes CER when reference text is available.

The script writes `results_fp32.json` or `results_int16.json` in `<AISHELL_EVAL_DIR>/`, containing per-utterance predictions, reference text, character errors, and final CER.

## Board WAV Validation

```bash
# Python: validate WAV preprocessing and the three-HBM pipeline
cd runtime/python
N_UTT=300 bash run.sh <AUDIO_DATA_DIR>

# C++: create temporary features from WAV and calculate CER from manifest text
cd runtime/cpp
N_UTT=300 bash run.sh <AUDIO_DATA_DIR>
```

Python reports WAV frontend time separately from HBM pipeline latency. C++ calculates CER when reference text is present and writes `runtime/cpp/build/eval/results_board_ucp.json`. Run `bash run.sh` without an argument for the bundled two-sample smoke validation.

## Completed Validation

| Item | Configuration |
| --- | --- |
| Evaluation set | AISHELL dev, 300 utterances, 40 speakers |
| Input | Real audio processed by FunASR and padded to `[1, 400, 560]` fbank+LFR features |
| Pipeline | Encoder INT16 HBM → Predictor INT16 HBM → CPU CIF → Decoder INT16 HBM → greedy decoding |
| Metric | CER |
| Vocabulary | 8,404-token common vocabulary |

| Metric | Python `hbm_runtime` | C++ UCP |
| --- | ---: | ---: |
| CER | **3.13%** | **3.13%** |
| Encoder | 33.63 ms | 33.15 ms |
| Predictor | 1.44 ms | 1.00 ms |
| CPU CIF | 3.41 ms | **0.38 ms** |
| Decoder | 7.12 ms | 6.29 ms |
| End-to-end | 45.61 ms | **40.81 ms** |
| RTF | 0.008 | 0.007 |

The `45.61 ms` and `40.81 ms` figures are historical 300-utterance HBM-pipeline measurements. The current Python WAV runtime additionally reports `frontend_ms` and end-to-end time including preprocessing, so its end-to-end figure should not be compared directly with those HBM-pipeline values.

Preserve the FunASR frontend, fixed input shape, HBM files, tensor wiring, valid-frame mask, `max_label_len=100`, zero `bias_embed`, and special-token filtering to preserve this result.
