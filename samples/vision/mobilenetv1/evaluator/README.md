# Model Evaluator

This directory records evaluation notes and validation references for the MobileNetV1 sample.

## Supported Model

| Model | Size | Classes |
| --- | --- | --- |
| MobileNetV1 | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`

## Benchmark Results

| Model | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- |
| MobileNetV1 | 71.7% | 65.4% | 0.58 | 2800+ |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
