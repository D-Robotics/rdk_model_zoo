# Model Evaluator

This directory records evaluation notes and validation references for the ResNet sample.

## Supported Model

| Model | Size | Classes |
| --- | --- | --- |
| ResNet18 | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`

## Benchmark Results

| Model | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- |
| ResNet18 | 71.5% | 70.5% | 2.95 | 449+ |

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
