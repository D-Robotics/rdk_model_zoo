# Model Evaluator

This directory provides validation notes for the VargConvNet sample.

## Supported Models

| Model | Size | Classes |
| --- | --- | --- |
| vargconvnet | 224x224 | 1000 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- Input format: packed NV12

## Benchmark Results

This sample does not provide a published benchmark table. The evaluator notes keep the runtime validation flow and model inventory for RDK X5.

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The sample prints Top-K classification results and saves the visualization image.
