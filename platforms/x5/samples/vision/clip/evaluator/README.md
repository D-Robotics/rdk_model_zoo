# Model Evaluator

This directory provides validation notes for the CLIP image-text matching sample.

## Supported Models

| Model | Role | Runtime |
| --- | --- | --- |
| `img_encoder.bin` | Image encoder | `hbm_runtime` |
| `text_encoder.onnx` | Text encoder | `onnxruntime` |

## Test Environment

- Platform: `RDK X5`
- Image encoder format: `.bin`
- Text encoder format: `.onnx`
- Python dependencies: `onnxruntime`, `ftfy`, `regex`
- Default image: `test_data/dog.jpg`
- Default prompts: `a diagram`, `a dog`

## Benchmark Results

This sample does not provide a published benchmark table. The evaluator notes keep the runtime validation flow and model protocol for RDK X5.

## Validation Summary

This sample is validated through the standardized Python runtime path:

- `runtime/python/run.sh`
- `runtime/python/main.py`

The expected default behavior is that the dog image scores higher for `a dog` than for `a diagram`.
