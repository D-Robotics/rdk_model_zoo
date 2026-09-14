[简体中文](./README_cn.md)

# Paraformer Speech Recognition

This sample deploys the fixed-shape **Paraformer-large-contextual** ASR pipeline on **RDK S100**. The deployment uses three INT16 HBM models and preserves the validated execution order:

```text
16 kHz WAV -> FunASR WavFrontend (fbank + LFR + CMVN)
  -> [1, 400, 560] feature -> Encoder HBM -> Predictor HBM -> CPU CIF -> Decoder HBM -> greedy token decoding
```

FunASR is used only to reproduce the original WAV preprocessing. Recognition remains the RDK three-HBM, CPU-CIF, and Decoder pipeline. The frontend follows the original fbank, LFR, and CMVN configuration, so graph I/O and inference behavior remain unchanged.

## Model Source

This deployment is derived from the following upstream projects and model package:

- FunASR source repository: `https://github.com/modelscope/FunASR`
- FunASR contextual Paraformer implementation: `https://github.com/modelscope/FunASR/tree/main/funasr/models/contextual_paraformer`
- ModelScope model: `https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`
- Reference paper: *Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition*

The original ModelScope package is a Chinese 16 kHz contextual Paraformer model with a common vocabulary of 8,404 tokens. This sample provides the RDK S100 deployment adaptation only; it does not replace the upstream FunASR training, fine-tuning, audio frontend, VAD, punctuation, or timestamp pipelines.

## Algorithm Overview

Paraformer is a non-autoregressive end-to-end ASR architecture. Unlike autoregressive decoders that emit tokens one at a time, it predicts token-level acoustic representations in parallel, then uses a bidirectional decoder to produce the transcription. The deployed graph is split into four logical stages:

- **Encoder**: Converts fixed fbank+LFR speech features into contextual acoustic representations.
- **Predictor**: Uses Continuous Integrate-and-Fire (CIF) activations to estimate token count and token-aligned acoustic embeddings.
- **CPU CIF**: Preserves the validated CIF calculation and creates the fixed decoder acoustic input.
- **Decoder**: Combines encoder context, CIF embeddings, zero contextual-bias embedding, and token count to generate token logits.

## Capabilities and Use Cases

The upstream model performs Chinese speech-to-text recognition for 16 kHz audio. It is suitable for applications such as:

- Chinese voice input and speech transcription.
- Offline speech navigation and voice-control front ends.
- Meeting transcription and speech content indexing.
- Domain adaptation workflows that use the upstream FunASR/ModelScope training and fine-tuning toolchain.

This RDK sample accepts 16 kHz WAV audio and returns UTF-8 text. It runs the original FunASR preprocessing internally before invoking the Paraformer HBM pipeline.

## Platform

| Platform | Support | Notes |
| --- | --- | --- |
| RDK S100 | Validated | Nash-e INT16 HBM package |

## Layout

The complete file tree is documented in the Chinese README. The sample contains `conversion/`, `evaluator/`, `model/`, `runtime/python/`, `runtime/cpp/`, and `test_data/`.

## Quick Start

1. Download the model package:

```bash
cd model
bash download_model.sh
```

2. The sample includes two validated 16 kHz WAV files under `test_data/`, so no extra data preparation is required:

```text
test_data/
├── manifest.json
└── audio/
    ├── BAC009S0724W0121.wav
    └── BAC009S0724W0168.wav
```

Each manifest item provides `utt_id` and optional reference `text`. The runtime creates fixed `[1, 400, 560]` features and valid frame lengths automatically.

3. Run either runtime with the bundled samples:

```bash
cd runtime/python && bash run.sh
cd runtime/cpp && bash run.sh
```

Pass a custom audio-data directory containing `manifest.json` and `audio/<utt_id>.wav` as the first argument when needed: `bash run.sh <AUDIO_DATA_DIR>`.

See `runtime/python/README.md`, `runtime/cpp/README.md`, and `evaluator/README.md` for CLI and manifest details.

## Accuracy Preservation

The model HBM files, tensor wiring, fixed feature shape, CPU CIF equation, zero `bias_embed`, maximum label length of 100, special-token filtering, and original FunASR audio preprocessing are unchanged from the validated Paraformer deployment.

## Model Conversion

Model Zoo provides compiled S100 HBM files, so normal users can run `model/download_model.sh` and start inference directly. To reproduce fixed-shape FunASR export, ONNX graph surgery, representative-data calibration, and three-stage INT16 compilation from the upstream model, see `conversion/README_cn.md`.

## Model Inference

- **Python Runtime**: intended for prototyping. WAV input is processed, inferred, aligned with CPU CIF, and decoded by `paraformer.py`; `main.py` schedules the manifest and reports latency. See `runtime/python/README.md`.
- **C++ Runtime**: intended for production deployment. It preserves the hbUCP/hbDNN implementation and generates temporary features from WAV before running the C++ pipeline. See `runtime/cpp/README.md`.

## Inference Results

The bundled data contains two AISHELL dev WAV samples. Both runtimes output Chinese text; the second sample has a validated homophone substitution (`搅` / `绞`), which produces a two-sample C++ CER of `3.57%`. Runtime output includes stage latency, and Python reports WAV frontend time separately.

The vocabulary uses `@@` as an English BPE continuation marker. Both runtimes remove this marker and merge subwords before printing the final transcription, so terminal output does not display `@@`.

## Model Evaluation

`evaluator/README.md` documents evaluation data layouts, FP32/INT16 host CER evaluation, board-side WAV evaluation commands, output files, and the completed 300-utterance AISHELL dev validation result.

## Quantization, Evaluation, and Deployment

The complete runnable quantization workflow is documented in `conversion/README_cn.md`: fixed-shape FunASR export, ONNX graph surgery, representative-data calibration, INT16 compilation of all three HBM stages, CER evaluation, and board deployment. The `conversion/` directory contains the scripts and YAML configurations required by that workflow; `evaluator/README.md` contains its validation configuration and results.

## License

This sample is distributed under the repository top-level `LICENSE`.
