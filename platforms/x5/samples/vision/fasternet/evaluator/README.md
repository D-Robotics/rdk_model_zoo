# Model Evaluator

This directory records evaluation notes and validation references for the FasterNet sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| FasterNet-S | 224x224 | 1000 |
| FasterNet-T2 | 224x224 | 1000 |
| FasterNet-T1 | 224x224 | 1000 |
| FasterNet-T0 | 224x224 | 1000 |

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
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 24.34 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 11.56 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 5.58 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 3.48 | 1135.13 |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
