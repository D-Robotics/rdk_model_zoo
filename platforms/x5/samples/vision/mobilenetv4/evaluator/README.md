# Model Evaluator

This directory records evaluation notes and validation references for the MobileNetV4 sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| MobileNetV4-Conv-Medium | 224x224 | 1000 |
| MobileNetV4-Conv-Small | 224x224 | 1000 |

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
| MobileNetV4-Conv-Medium | 224x224 | 9.68 | 76.75% | 75.14% | 2.42 | 6.91 | 572.36 |
| MobileNetV4-Conv-Small | 224x224 | 3.76 | 70.75% | 68.75% | 1.18 | 2.74 | 1436.22 |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
