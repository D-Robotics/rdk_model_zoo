# Model Evaluation

[简体中文](./README_cn.md) | English

This directory provides benchmark instructions and validation references for the Whisper sample.

## Supported Models

| Model | Params (M) | Task |
| --- | --- | --- |
| Whisper Tiny | 39 | ASR |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime` for the encoder and ONNX Runtime CPU for the ONNX decoder.
- Model Format: `.bin`
- CPU: 8xA55@1.8GHz, all cores in Performance mode
- BPU: 1xBayes-e@1GHz, equivalent to 10 TOPS INT8 compute power

## Metrics Description

- **WER**: Word Error Rate, the standard metric for ASR evaluation.
- **Latency**: End-to-end inference latency for a single audio input.
- **RTF**: Real-Time Factor, the ratio of inference time to audio duration.

## Benchmark Results

| Model | WER (%) | Latency (ms) | RTF |
| --- | --- | --- | --- |
| Whisper Tiny | TBD | TBD | TBD |

## Run the evaluation

With the deployment files in `model/`, run the evaluator from this directory on the board.

```bash
python3 eval.py
```

By default it compares `../test_data/test_audio.wav` with `../test_data/reference.json`. Use `--audio-dir` and `--ground-truth` for your own dataset. Use `--json-save-path` to select the output file for the transcripts and mean utterance WER.
