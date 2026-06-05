# Model Evaluation

This directory provides benchmark instructions and validation references for the HGNetv2 sample.

## Supported Models

| Model | Input Size | Number of Classes |
| --- | --- | --- |
| HGNetv2_b0 | 224x224 | 1000 |
| HGNetv2_b1 | 224x224 | 1000 |
| HGNetv2_b2 | 224x224 | 1000 |
| HGNetv2_b3 | 224x224 | 1000 |
| HGNetv2_b4 | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime Backend: `hbm_runtime`
- Model Format: `.bin`
- CPU: 8xA55@1.8GHz, all cores in Performance mode
- BPU: 1xBayes-e@1GHz, equivalent to 10 TOPS INT8 compute power

## Metrics Description

- **Float Top-1**: Classification accuracy of the ONNX model before quantization.
- **Quantized Top-1**: Actual inference accuracy of the deployed quantized model.
- **Single‑thread Latency**: Inference latency for a single frame, single thread, and single BPU core.
- **Multi‑thread Latency**: Measured latency under multi‑threaded task submission.
- **FPS**: Multi‑thread throughput test result on `RDK X5`.

## Benchmark Results

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

## Validation Instructions

This sample is validated through the standard Python inference pipeline:

- `evaluator/eval.py`

The validation dataset is ImageNet-1k val.