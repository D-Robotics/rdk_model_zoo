# Wav2Vec2 ASR Python Example

English | [简体中文](./README_cn.md)

This example demonstrates how to perform automatic speech recognition (ASR) tasks on the BPU using a quantized Wav2Vec2 model with CTC decoding.

## Directory Structure

```text
.
|-- main.py
|-- compare_accuracy.py
|-- wav2vec2.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the quantized `.bin` model file. | `../../model/wav2vec2_base_960h_1x65535_x5.bin` |
| `--priority` | Model priority, range `0~255`. | `0` |
| `--bpu-cores` | BPU core index used for inference. | `0` |
| `--test-audio` | Path to the test input audio file (WAV). | `../../test_data/test_audio.wav` |
| `--text-save-path` | Path to save the transcribed text. | `../../test_data/result.txt` |
| `--language` | Language code for transcription (e.g., en, zh). | `en` |
| `--blank-id` | CTC blank token ID. | `0` |
| `--vocab-path` | CTC vocabulary JSON. | The model directory or the bundled Base 960h vocabulary |

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
    --model-path ../../model/wav2vec2_base_960h_1x65535_x5.bin \
    --test-audio ../../test_data/test_audio.wav \
    --language en \
    --blank-id 0
```

## API Description

- **Wav2Vec2Config**: Encapsulates the model path, language, and CTC blank token ID.
- **Wav2Vec2**: Implements audio preprocessing, BPU inference, and CTC greedy decoding to text.

## Quantization Accuracy Verification

`compare_accuracy.py` measures the accuracy loss introduced by INT8 quantization by comparing the CTC logits of the float model and the BPU quantized model. Reported metrics:

- **Cosine similarity**: the standard quantization-loss metric of the toolchain; a value `>= 0.95` is usually acceptable.
- **Per-frame argmax match rate**: directly predicts whether greedy CTC decoding changes; the non-blank match rate is the more informative number.
- **Token-level CER**: edit distance between the two decoded token sequences.
- Numerical error statistics (MAE / max absolute error).

### Step 1: Dump the float reference on the conversion PC

After the ONNX export, run once on the PC to dump the float logits to an `.npz` (no BPU needed):

```bash
python3 compare_accuracy.py \
    --float-onnx ../../model/wav2vec2_base.onnx \
    --test-audio ../../test_data/test_audio.wav \
    --save-float-logits ../../test_data/float_logits.npz
```

### Step 2: Compare on the board

Copy `float_logits.npz` together with the compiled `.bin` to the board, then run:

```bash
python3 compare_accuracy.py \
    --float-logits ../../test_data/float_logits.npz \
    --bin-model ../../model/wav2vec2_base_960h_1x65535_x5.bin \
    --test-audio ../../test_data/test_audio.wav
```

If the board CPU is fast enough, `--float-onnx` can also be used to compute the float reference online via onnxruntime.

| Parameter | Description | Default |
| --- | --- | --- |
| `--float-onnx` | Path to the float ONNX model (opset 11). | none |
| `--float-logits` | Path to a previously dumped float logits `.npz`. | none |
| `--bin-model` | Path to the BPU quantized `.bin`; omit to only dump float logits. | none |
| `--test-audio` / `--audio-dir` | Single test audio / directory of audios (averaged). | `../../test_data/test_audio.wav` |
| `--save-float-logits` | Where to save the float logits. | none |
| `--cos-threshold` | Cosine similarity threshold for the PASS verdict. | `0.95` |
| `--blank-id` | CTC blank token ID. | `0` |
| `--vocab-path` | CTC vocabulary JSON. | The model directory or the bundled Base 960h vocabulary |
| `--report-path` | JSON report path. | `../../test_data/wav2vec2_accuracy_report.json` |

The run prints per-sample metrics, the average over all samples, and a `PASS/FAIL` verdict. If the cosine similarity is below the threshold, check whether the calibration data is representative or adjust the compile options.
