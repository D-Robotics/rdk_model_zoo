English | [简体中文](./README_cn.md)

# YOLOE Model Evaluation

This directory records the benchmark data, runtime verification results, and performance notes for YOLOE on RDK X5.

## Supported Models

The current X5 benchmark scope covers:

- `yoloe_11s_seg_pf_bayese_640x640_nv12.bin`

## Test Environment

- Device: `RDK X5`
- Runtime backend: `hbm_runtime`
- Model format: `.bin`
- Input size: `640x640`
- Input format: `NV12`

## Verification Method

The Python sample is verified with:

```bash
cd ../runtime/python
bash run.sh
python3 main.py
```

## Benchmark Results

### RDK X5 Performance Data

| Model | Size | Threads | Latency | FPS |
| :--- | :--- | ---: | :--- | :--- |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 142.9 ms | 7.0 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 2 | 149.5 ms | 13.3 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 3 | 167.4 ms | 17.8 FPS |

## Performance Notes

- `Latency` is the total end-to-end latency per frame on RDK X5.
- `FPS` is the overall throughput for the given number of threads.
- The model supports 4585-class open-vocabulary instance segmentation.

## Reference Materials

- Runtime usage: `../runtime/python/README.md`
- Model download: `../model/README.md`
- Conversion notes: `../conversion/README.md`
- Benchmark reference assets: `../test_data/`
