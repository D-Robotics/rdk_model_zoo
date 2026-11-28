# Whisper Model Conversion and Compilation Guide

English | [简体中文](./README_cn.md)

This directory provides tools and instructions for converting Whisper models into BPU quantized models (`.bin`) compatible with D-Robotics `RDK X5` hardware.

## Directory Structure

```text
.
|-- onnx_export
|   |-- export_whisper_bpu.py
|-- generate_calibration_data.py
|-- whisper_encoder_tiny.yaml
|-- whisper_decoder_tiny.yaml   (decoder not deployed to BPU — kept for reference)
|-- README.md
`-- README_cn.md
```

## Compilation environment

Compilation uses `x5-asr-compile` from the companion `RDK_x5_asr` project with OpenExplorer X5 CPU 1.2.8. See that project’s README for environment setup.

For PyTorch export, install `torch`, `openai-whisper`, `onnx` and `onnxsim` on the host. If the ONNX and calibration tensors are ready, start at step 3.

## Conversion Process

### 1. PyTorch → ONNX

Export the Whisper encoder and decoder to ONNX format:

```bash
cd samples/speech/whisper/conversion
python3 onnx_export/export_whisper_bpu.py --model_type tiny
```

The export writes the encoder, a variable-length `whisper_decoder_tiny.onnx`, and `multilingual.tiktoken` to `../model/`. For offline export, pass `--checkpoint /path/to/tiny.pt`. The configuration below targets multilingual Tiny.

### 2. Prepare Calibration Data

``hb_mapper`` uses precomputed log-Mel spectrograms for calibration. Start with 20–50 representative utterances.
Use the provided script to generate them from WAV audio files:

```bash
# Create calibration data from a directory of WAV files
python3 generate_calibration_data.py \
    --audio-dir /path/to/wav/files \
    --mel-filters ../model/mel_80_filters.txt \
    --output-dir ../cal_data \
    --num-samples 50
```

Each file in `../cal_data/` contains one headerless float32 log-Mel tensor with shape `(1, 80, 3000, 1)` and size 960000 bytes. The compiler also reads existing float32 NumPy calibration tensors.

**Calibration data requirements:**
- Input: WAV audio (any sample rate; resampling to 16 kHz is handled automatically)
- 20–50 diverse audio samples covering typical speech patterns

### 3. Compile for X5

In the ASR conversion environment, run the following commands from `samples/speech/whisper/conversion/`.

```bash
x5-asr-compile whisper_encoder_tiny.yaml
cp whisper_encoder_tiny_80x3000_x5/whisper_encoder_tiny_80x3000_x5.bin ../model/
```

The resulting X5 encoder uses mixed precision. The board runs `whisper_decoder_tiny.onnx` with the ONNX Runtime CPU provider.

---

## Supported Variants

| Variant | Encoder ONNX | Output `.bin` |
| --- | --- | --- |
| Tiny | `whisper_encoder_tiny.onnx` | `whisper_encoder_tiny_80x3000_x5.bin` |

---

## Troubleshooting

### "file does not start with RIFF id"

The audio bytes in the parquet file are not WAV format (no RIFF header).
This typically means the audio is encoded as **MP3** (starts with ``ID3``) or
another compressed format.

**Solution**: Install ``soundfile`` which acts as a universal codec backend:

```bash
pip install soundfile
```

The calibration script automatically falls back to soundfile for any format
not recognized as WAV or FLAC (supports MP3, OGG, etc.).

### "No module named 'scipy'"

Audio with a sample rate other than 16 kHz needs resampling, which requires
``scipy``.

**Solution**:

```bash
pip install scipy
```

### "Row N: unsupported audio column type dict, keys=['bytes', 'path']"

The parquet ``audio`` column does not contain a ``sampling_rate`` key.
The script now automatically extracts the sample rate from the audio data
via soundfile, so this warning can be ignored as long as soundfile is
installed.

### WARNING: Fewer than 20 calibration samples

INT8 calibration quality may be suboptimal with fewer than 20 samples.
Use ``--num-samples 50`` (or more) to ensure good quantization accuracy.

### Python Dependencies for Calibration (host PC)

The calibration script runs on the **host PC** (not the board) and requires:

```bash
pip install numpy pandas pyarrow soundfile scipy
```

---

## Output Protocol

The runtime sample assumes:

- The encoder receives a float32 log-Mel featuremap of shape `(1, 80, 3000, 1)`.
- The encoder emits hidden states of shape `(1, 1500, 384)` for the decoder.

---

## License
The tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).
