# Whisper Model Description

English | [简体中文](./README_cn.md)

This directory provides complete usage instructions for the Whisper sample in the Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation instructions.

## Algorithm Introduction

Whisper is a general-purpose automatic speech recognition (ASR) model developed by OpenAI. It is trained on a large-scale multilingual, multitask dataset and supports transcription, translation, and language identification in a single encoder-decoder Transformer architecture.

- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Official Implementation**: [openai/whisper](https://github.com/openai/whisper)

### Algorithm Functions

Whisper supports the following tasks:

- Multilingual speech-to-text transcription
- Speech translation (to English)
- Language identification

### Algorithm Features

- **Encoder-Decoder Transformer**: Unified architecture for transcription, translation, and language detection.
- **Multilingual Support**: Trained on 680k hours of multilingual speech data covering 99 languages.
- **Log-Mel Spectrogram Input**: Audio is converted to 80-channel log-Mel spectrograms as model input.
- **Token-based Output**: The model outputs text tokens decoded via a multilingual tokenizer.

## Directory Structure

```text
conversion/       # ONNX export, calibration and X5 compilation
evaluator/        # Transcription accuracy evaluation
model/            # Deployment artifacts, vocabulary and provisioning guide
runtime/python/   # Python inference entry point and run script
test_data/        # Example waveform and reference transcript
```

## Quick Start

### Python

Prepare the [model files](./model/README.md) first. On RDK X5, the script selects the local or board-installed model and transcribes the included WAV.

- For detailed Python instructions, please refer to [runtime/python/README.md](./runtime/python/README.md).
- Quick start command:

```bash
cd runtime/python
bash run.sh
```

## Model Conversion

- The board requires the compiled encoder, CPU ONNX decoder and vocabulary. See the [model file list](./model/README.md).
- Conversion instructions can be found in [conversion/README.md](./conversion/README.md).

## Model Inference

Currently, this sample maintains the Python inference path.

- Python inference instructions: [runtime/python/README.md](./runtime/python/README.md)

## Model Evaluation

For evaluation instructions, performance data, and validation results, please refer to [evaluator/README.md](./evaluator/README.md).

## Example output

The X5 toolchain’s quantized simulation transcribed `test_data/test_audio.wav` as **Please open the window.**. Board WER, latency and real-time factor are pending measurement.

## License

Follows the Model Zoo top‑level License.
