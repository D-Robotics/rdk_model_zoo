# LPRNet Model Evaluation

This directory documents the benchmark and validation settings for LPRNet on RDK X5.

## Prerequisites

- `RDK OS >= 3.5.0`
- Python 3 on board

## Dataset Preparation

- This sample uses a bundled binary input tensor:
  - `../test_data/test.bin`
- The corresponding visual reference image is:
  - `../test_data/example.jpg`

## Usage

Run the Python sample:

```bash
cd ../runtime/python
bash run.sh
```

## Benchmark Results

| Model | Test Frames | FPS | Average Latency | BPU Usage | ION Memory |
| --- | --- | --- | --- | --- | --- |
| `lpr.bin` | `100` | `266 FPS` | `3.75 ms` | `9%` | `1.11 MB` |

## Validation Summary

The runtime validation for this sample should confirm:

- the model loads successfully on RDK X5
- the input tensor can be read from `test.bin`
- the runtime prints a decoded license plate string
