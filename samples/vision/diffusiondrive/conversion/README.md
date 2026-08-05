English | [简体中文](./README_cn.md)

# DiffusionDrive Model Conversion

This directory records the reproducible PTQ configuration used for the S600 model. Conversion runs on an x86 Linux host with:

```text
registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

## BPU-Friendly ONNX

Start from the official DiffusionDrive NAVSIM checkpoint and export a deterministic four-input graph. The exported interface is:

| Name | Shape | Description |
| --- | --- | --- |
| `camera` | `1x3x256x1024` | Stitched left/front/right RGB tensor |
| `lidar` | `1x1x256x256` | LiDAR histogram |
| `status` | `1x8` | Ego status and driving command |
| `noise` | `1x20x8x2` | Explicit diffusion noise |

The export must replace ScatterND-style in-place writes with concatenation, and fixed adaptive average pools with static depthwise convolutions. Place the resulting model at `build/diffusiondrive_navsim_bpu_clean_float.onnx`.

## Calibration

Use at least 100 real NAVSIM mini samples. Each input has a separate directory under `calibration_data/`, containing matching float32 `.npy` files.

## Compile

From this directory inside the v3.7.0 container:

```bash
hb_compile -c configs/diffusiondrive_r34_256x1024_s600.yaml
```

The accepted configuration uses INT16 activations throughout the graph and max calibration. This is intentional: all-INT8 PTQ reduced the final BEV cosine to `0.370948`, while keeping only the four BEV-head nodes in INT16 still produced `0.371840` cosine and `0.143013` mean IoU. The upstream fused feature at `/_backbone/Add_6` had already degraded, so changing only the last head could not recover the semantic map.

With full INT16 and max calibration, board-versus-float BEV cosine reaches `0.998918`, pixel agreement reaches `0.944061`, and mean IoU reaches `0.868425`. Compiler reports and board profiling confirm that every segment runs on the BPU with `0.0 ms` CPU inference time.

Validate the generated model with:

```bash
hrt_model_exec model_info --model_file build/hbm/diffusiondrive_r34_256x1024_s600.hbm
hrt_model_exec perf --model_file build/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 1 --core_id 0
```
