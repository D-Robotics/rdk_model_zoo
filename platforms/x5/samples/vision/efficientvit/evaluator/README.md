# Model Evaluator

This directory provides benchmark notes and validation references for the EfficientViT sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 |

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
| EfficientViT_m5 | 224x224 | 12.4 | 73.75% | 72.50% | 6.34 | 22.69 | 174.70 |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
