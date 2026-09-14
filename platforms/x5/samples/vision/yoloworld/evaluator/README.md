# Model Evaluator

This directory provides validation notes for the YOLOWorld sample.

## Supported Models

| Model | Task | Input Size | Vocabulary Slots |
| --- | --- | --- | --- |
| `yolo_world.bin` | Open-vocabulary detection | 640x640 | 32 |

## Test Environment

- Platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- Default image: `test_data/dog.jpeg`
- Default prompt: `dog`

## Benchmark Results

This sample does not provide a published benchmark table. The evaluator notes keep the runtime validation flow and model protocol for RDK X5.

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The expected default behavior is to detect the dog in the test image and save a visualization image.
