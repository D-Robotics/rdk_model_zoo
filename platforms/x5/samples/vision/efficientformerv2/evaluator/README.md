# Model Evaluator

This directory records evaluation notes and validation references for the EfficientFormerV2 sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| EfficientFormerV2-S0 | 224x224 | 1000 |
| EfficientFormerV2-S1 | 224x224 | 1000 |
| EfficientFormerV2-S2 | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- CPU: 8xA55@1.8GHz with full-core Performance scheduling
- BPU: 1xBayes-e@1GHz, 10TOPS equivalent INT8 compute

## Metric Description

- Float Top-1 is measured on the floating-point ONNX model before quantization.
- Quant Top-1 is measured on the quantized deployment model.
- Latency is the single-frame, single-thread, single-BPU-core inference latency.
- FPS is measured by multi-threaded task submission to keep BPU utilization high.

## Benchmark Results

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientFormerV2-S2 | 224x224 | 12.6 | 77.50% | 70.75% | 6.99 | 26.01 | 152.40 |
| EfficientFormerV2-S1 | 224x224 | 6.1 | 77.25% | 68.75% | 4.24 | 14.35 | 275.95 |
| EfficientFormerV2-S0 | 224x224 | 3.5 | 74.25% | 68.50% | 5.79 | 19.96 | 198.45 |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
