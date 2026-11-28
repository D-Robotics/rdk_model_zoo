# Wav2Vec2 Model Conversion and Compilation Guide

English | [简体中文](./README_cn.md)

This directory provides tools and instructions for converting Wav2Vec2 models into BPU quantized models (`.bin`) compatible with D-Robotics `RDK X5` hardware.

## Directory Structure

```text
.
|-- onnx_export
|   |-- export_wav2vec2_bpu.py
|-- wav2vec2_base.yaml
|-- README.md
`-- README_cn.md
```

## Compilation environment

Compilation uses `x5-asr-compile` from the companion `RDK_x5_asr` project with OpenExplorer X5 CPU 1.2.8. See that project’s README for environment setup.

## Conversion Process

### 1. Export static ONNX

Run these commands from the repository root in a separate host Python environment. With an existing ONNX file, start at calibration.

```bash
cd samples/speech/wav2vec2/conversion
python3 -m pip install -r onnx_export/requirements.txt
python3 onnx_export/export_wav2vec2_bpu.py
```

The default checkpoint is `facebook/wav2vec2-base-960h`. It exports to `../model/wav2vec2_base.onnx` with a fixed `[1, 65535]` waveform input at 16 kHz. Use `--samples` to change the length. `--checkpoint` accepts a local pretrained directory or another CTC checkpoint such as `ccoreilly/wav2vec2-large-100k-voxpopuli-catala`.

### 2. Prepare calibration waveforms

Run from `conversion/` with representative speech recordings. The script reads the input length from the ONNX, resamples to 16 kHz mono, and pads or trims each waveform. It writes headerless float32 tensors to `../cal_data/`. Each default tensor occupies 262140 bytes.

```bash
python3 generate_calibration_data.py --audio-dir /path/to/speech --num-samples 50
```

### 3. Compile for X5

In the ASR conversion environment, run the following commands from `samples/speech/wav2vec2/conversion/`.

```bash
x5-asr-compile wav2vec2_base.yaml
cp wav2vec2_base_960h_1x65535_x5/wav2vec2_base_960h_1x65535_x5.bin ../model/
```

The input is a float32 waveform with the length defined by the exported ONNX. If you change that length, update the output name in the YAML and the runtime arguments.

Calibration, the floating-point reference, and the board runtime apply zero-mean, unit-variance normalization to the same fixed-length waveform. The sample includes the Base 960h vocabulary. For a different checkpoint, place its `vocab.json` next to the `.bin` or pass `--vocab-path`.

## Supported Variants

| Variant | Model ID | Output `.bin` |
| --- | --- | --- |
| Base | `facebook/wav2vec2-base-960h` | `wav2vec2_base_960h_1x65535_x5.bin` |

---

## Output Protocol

The runtime sample assumes:

- Input tensor: raw audio waveform resampled to 16 kHz
- Output tensor: CTC emission logits for decoding

---

## Board-Side Accuracy Verification

After compilation, use `runtime/python/compare_accuracy.py` to compare the CTC logits of the float ONNX and the BPU quantized model. The quantization loss is measured by **cosine similarity** (recommended `>= 0.95`), per-frame argmax match rate, and token-level CER:

1. Dump the float logits reference on the conversion PC:
   ```bash
   cd ../runtime/python
   python3 compare_accuracy.py \
       --float-onnx ../../model/wav2vec2_base.onnx \
       --test-audio ../../test_data/test_audio.wav \
       --save-float-logits ../../test_data/float_logits.npz
   ```
2. Copy `float_logits.npz` and the `.bin` to the board, then compare:
   ```bash
   python3 compare_accuracy.py \
       --float-logits ../../test_data/float_logits.npz \
       --bin-model ../../model/wav2vec2_base_960h_1x65535_x5.bin \
       --test-audio ../../test_data/test_audio.wav
   ```

If the cosine similarity is below the threshold, check whether the calibration data is representative (20–50 audio samples close to the deployment scenario) or adjust the compile options and recompile.

---

## License
The tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).
