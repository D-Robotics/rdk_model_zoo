English | [简体中文](./README_cn.md)

# Model Evaluation

## Runtime Performance

S100 V1P0, RDK OS 4.0.5-Beta, UCP 3.13.6 / HBRT 4.7.5.
HBM: OE 3.7.0, HBDK 4.7.5, INT8 KL, batch 1, 640x640 NV12, 4585 classes.
Measured on 2026-09-08 with `hrt_model_exec perf`, 200 frames, warmup enabled,
thread_num=1 and core_id=0.

Only **model Runtime** latency and throughput are published. Application
preprocessing, output dequantization, postprocessing, visualization and I/O are
excluded. These are not Python/C++ end-to-end FPS.

| Board | Model | Runtime latency (ms) | Runtime FPS |
|---|---|---:|---:|
| S100 | YOLOE-26n Seg PF | 4.943 | 200.74 |
| S100 | YOLOE-26s Seg PF | 9.944 | 100.08 |
| S100 | YOLOE-26m Seg PF | 11.765 | 84.55 |
| S100 | YOLOE-26l Seg PF | 13.417 | 74.18 |
| S100 | YOLOE-26x Seg PF | 22.013 | 45.31 |
| S100P | YOLOE-26n/s/m/l/x Seg PF | Not measured | Not measured |

```bash
hrt_model_exec perf --model_file /path/model.hbm \
  --thread_num 1 --core_id 0 --frame_count 200 --enable_warmup true \
  --profile_path ./profile
```

## Accuracy Status

The original five S100 Python/C++ runs returned matching detection counts and
labels; boxes/scores agreed within 1e-7 and mask areas matched. Area equality
does not establish pixel-wise mask identity. S100P artifacts have passed
compilation and model I/O checks, but no S100P board benchmark is claimed.

Calibration used 100 representative COCO train2017 images. Dataset box/mask mAP
has not been accepted, and the INT8 baseline shows numerical differences from
float inference. Validate on a held-out labeled dataset before production use.
Map the PF vocabulary explicitly to dataset categories: PF IDs are not COCO IDs.
No invented mAP or S100P Runtime numbers are included.
