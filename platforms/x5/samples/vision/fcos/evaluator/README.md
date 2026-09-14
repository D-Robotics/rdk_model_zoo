# FCOS Model Evaluation

This directory documents the benchmark and validation settings for FCOS on RDK X5.

## Prerequisites

- `RDK OS >= 3.5.0`
- Python 3 on board
- `pycocotools` if you want to reproduce COCO mAP

## Dataset Preparation

- Detection benchmark is based on COCO validation data
- Example runtime image is stored in `../test_data/bus.jpg`

## Benchmark Results

### RDK X5 Performance Data

| Model | Size | Classes | BPU Latency / Throughput | Python Post-process |
| --- | --- | --- | --- | --- |
| `fcos_efficientnetb0` | 512x512 | 80 | `3.3 ms / 298.0 FPS (1 thread)`<br>`6.2 ms / 323.0 FPS (2 threads)` | `9 ms` |
| `fcos_efficientnetb2` | 768x768 | 80 | `14.4 ms / 69.5 FPS (1 thread)`<br>`28.1 ms / 70.9 FPS (2 threads)` | `16 ms` |
| `fcos_efficientnetb3` | 896x896 | 80 | `26.1 ms / 38.2 FPS (1 thread)`<br>`51.6 ms / 38.7 FPS (2 threads)` | `20 ms` |

## Performance Test Instructions

- Test platform: RDK X5 or RDK X5 Module
- CPU condition: all A55 cores in performance mode
- BPU condition: Bayes-e at peak operating mode
- Test command example:
  ```bash
  hrt_model_exec perf --thread_num 2 --model_file fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
  ```

## Accuracy Test Instructions

- Accuracy is evaluated on COCO validation data with `pycocotools`
- The published FCOS X5 sample focuses on runtime deployment and benchmark reproduction
- If you need to reproduce mAP exactly, keep the same input size, NV12 preprocessing, and confidence/NMS thresholds as the runtime sample
