English | [简体中文](./README_cn.md)

# YOLOE Model Evaluation

This directory records the benchmark data, runtime verification results, and performance notes for YOLOE on RDK X5.

## Supported Models

The current X5 benchmark scope covers:

- `yoloe_11s_seg_pf_bayese_640x640_nv12.bin`
- `yoloe_11m_seg_pf_bayese_640x640_nv12.bin`
- `yoloe_11l_seg_pf_bayese_640x640_nv12.bin`

## Test Environment

- Device: `RDK X5 V1.0`
- OS: `3.4.1-rp1.0.2`
- Runtime backend: C++ `libdnn 1.24.5` / `HBRT 3.15.55`
- Model format: `.bin`
- Input size: `640x640`
- Input format: `NV12`
- BPU: `core_id=1` (BPU core 0), `1000 MHz`

## Verification Method

Timing uses the C++ libdnn API with a fixed NV12 input: three rounds, each with 10 warmup frames and 200 timed frames.

## Benchmark Results

### RDK X5 Performance Data

| Model | Size | Threads | Mean Latency (ms) | P50 (ms) | P95 (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 146.16 | 144.72 | 152.73 | 6.84 |
| YOLOE-11m-Seg-PF | 640x640 | 1 | 177.14 | 176.17 | 182.54 | 5.65 |
| YOLOE-11l-Seg-PF | 640x640 | 1 | 189.97 | 187.99 | 196.30 | 5.26 |

## Performance Notes

- Latency, P50 and P95 cover Runtime inference, including CPU/BPU execution inside the model, excluding image preprocessing and detection/mask postprocessing.
- FPS is the total timed frame count divided by total elapsed time.
- The 11s two-thread test failed with an ION allocation error; only single-thread results are listed.

## Reference Materials

- Runtime usage: `../runtime/python/README.md`
- Model download: `../model/README.md`
- Conversion notes: `../conversion/README.md`
- Benchmark reference assets: `../test_data/`
