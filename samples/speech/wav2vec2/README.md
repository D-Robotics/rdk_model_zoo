# Wav2Vec2 Model Description

English | [简体中文](./README_cn.md)

This directory provides complete usage instructions for the Wav2Vec2 sample in the Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation instructions.

## Algorithm Introduction

Wav2Vec2 is a self-supervised speech representation learning model developed by Meta AI. It learns contextualized speech representations from raw audio via a convolutional feature encoder and a Transformer-based context network, fine-tuned on labeled data with CTC for automatic speech recognition (ASR).

- **Paper**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- **Official Implementation**: [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

### Algorithm Functions

Wav2Vec2 supports the following tasks:

- Automatic speech recognition (ASR) transcription
- Phoneme recognition
- Speech feature extraction

### Algorithm Features

- **Self-supervised Pre-training**: Learns speech representations from unlabeled audio data via contrastive learning.
- **Convolutional Encoder**: Maps raw audio waveforms to latent feature sequences using temporal convolutions.
- **Transformer Context Network**: Builds contextualized representations over the latent features.
- **CTC Decoding**: Fine-tuned with Connectionist Temporal Classification (CTC) loss for sequence-to-sequence ASR.

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

- The [model guide](./model/README.md) lists the deployment artifacts.
- Conversion instructions can be found in [conversion/README.md](./conversion/README.md).

## Model Inference

Currently, this sample maintains the Python inference path.

- Python inference instructions: [runtime/python/README.md](./runtime/python/README.md)

## Model Evaluation

For evaluation instructions, performance data, and validation results, please refer to [evaluator/README.md](./evaluator/README.md).

## Example output

The X5 toolchain’s quantized simulation transcribed `test_data/test_audio.wav` as **THE TRAIN IS COMING**. Board WER, latency and real-time factor are pending measurement.

## License

Follows the Model Zoo top‑level License.
