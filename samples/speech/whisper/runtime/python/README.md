# Whisper ASR Python Example

English | [简体中文](./README_cn.md)

The board loads the X5 encoder with the BSP's `hbm_runtime` and runs the Whisper Tiny decoder with ONNX Runtime's CPU provider. Audio preprocessing uses NumPy and either SciPy or SoundFile for PCM WAV input.

## Directory Structure

```text
.
|-- main.py
|-- whisper.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--encoder-model-path` | Path to the BPU encoder `.bin` model. | `../../model/whisper_encoder_tiny_80x3000_x5.bin` |
| `--decoder-model-path` | CPU decoder ONNX path. An existing decoder `.bin` is also accepted. | `../../model/whisper_decoder_tiny.onnx` |
| `--vocab-path` | Path to the vocabulary text file. | `../../model/multilingual.tiktoken` |
| `--mel-filters-path` | Path to the Mel filterbank data file. | `../../model/mel_80_filters.txt` |
| `--priority` | Model priority, range `0~255`. | `0` |
| `--bpu-cores` | BPU core index used for inference. | `0` |
| `--test-audio` | Path to the test input audio file (WAV). | `../../test_data/test_audio.wav` |
| `--text-save-path` | Path to save the transcribed text. | `../../test_data/result.txt` |
| `--language` | Language code for transcription (e.g., en, zh). | `en` |

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

## Manual Execution

- Using default parameters:

```bash
python3 main.py
```

- Explicitly specifying parameters:

```bash
python3 main.py \
    --encoder-model-path ../../model/whisper_encoder_tiny_80x3000_x5.bin \
    --decoder-model-path ../../model/whisper_decoder_tiny.onnx \
    --test-audio ../../test_data/test_audio.wav \
    --language en
```

## API Description

- **WhisperConfig**: Encapsulates encoder/decoder model paths, vocab, and Mel filterbank paths.
- **Whisper**: Implements audio preprocessing (log-Mel spectrogram), X5 encoder inference and autoregressive CPU decoding.
