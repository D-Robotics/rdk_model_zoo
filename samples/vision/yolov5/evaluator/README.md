English | [简体中文](./README_cn.md)

# YOLOv5 Model Evaluation

This directory records the benchmark data, runtime verification results, and performance notes for YOLOv5 on RDK X5.

## Supported Models

The current X5 benchmark scope covers:

- `YOLOv5s_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5m_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5l_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5x_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5n_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5s_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5m_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5l_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5x_tag_v7.0_detect_640x640_bayese_nv12.bin`

## Test Environment

- Device: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- Input size: `640x640`
- Input format: `NV12`

## Verification Method

The migrated Python sample is verified with:

```bash
cd ../runtime/python
bash run.sh
python3 main.py
python3 main.py --model-path ../../model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

All 9 published X5 YOLOv5 models pass the current Python runtime path and can save output images normally.

## Benchmark Results

### RDK X5 Performance Data

| Model | Size | Params | BPU Throughput | Python Post-process |
| :--- | :--- | ---: | :--- | :--- |
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

## Performance Notes

- `BPU Throughput` is the reference throughput on RDK X5 for the corresponding model.
- `Python Post-process` is the reference CPU-side latency of the Python post-processing path.
- `Params` is the parameter count of the original FP32 model.
- The migrated sample keeps the original YOLOv5 anchor-based decoding protocol.

## Reference Materials

- Runtime usage: `../runtime/python/README.md`
- Model download: `../model/README.md`
- Conversion notes: `../conversion/README.md`
- Benchmark reference assets: `../test_data/`
