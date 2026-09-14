# Model Evaluator

This directory provides benchmark notes and validation references for the RepVGG sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| RepVGG_A0 | 224x224 | 1000 |
| RepVGG_A1 | 224x224 | 1000 |
| RepVGG_A2 | 224x224 | 1000 |
| RepVGG_B0 | 224x224 | 1000 |
| RepVGG_B1g2 | 224x224 | 1000 |
| RepVGG_B1g4 | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- CPU: 8xA55@1.8GHz with full-core Performance scheduling
- BPU: 1xBayes-e@1GHz, 10TOPS equivalent INT8 compute

## Metric Description

- Float Top-1 is measured on the floating-point ONNX model before quantization.
- Quant Top-1 is measured on the quantized deployment model.
- Single-thread latency is the single-frame, single-thread, single-BPU-core inference latency.
- Multi-thread latency is measured under multi-thread task submission.
- FPS is the multi-thread throughput measured on `RDK X5`.

## Benchmark Results

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepVGG_B1g2 | 224x224 | 41.36 | 77.78 | 68.25 | 9.77 | 36.19 | 109.61 |
| RepVGG_B1g4 | 224x224 | 36.12 | 77.58 | 62.75 | 7.58 | 27.47 | 144.39 |
| RepVGG_B0 | 224x224 | 14.33 | 75.14 | 60.36 | 3.07 | 9.65 | 410.55 |
| RepVGG_A2 | 224x224 | 25.49 | 76.48 | 62.97 | 6.07 | 21.31 | 186.04 |
| RepVGG_A1 | 224x224 | 12.78 | 74.46 | 62.78 | 2.67 | 8.21 | 482.20 |
| RepVGG_A0 | 224x224 | 8.30 | 72.41 | 51.75 | 1.85 | 5.21 | 757.73 |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
