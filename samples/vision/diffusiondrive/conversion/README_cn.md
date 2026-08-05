[English](./README.md) | 简体中文

# DiffusionDrive 模型转换

本目录记录 S600 模型可复现的 PTQ 配置。转换在 x86 Linux 主机执行，固定使用：

```text
registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

## BPU 友好 ONNX

从 DiffusionDrive 官方 NAVSIM checkpoint 导出确定性的四输入模型：`camera` 为 `1x3x256x1024`，`lidar` 为 `1x1x256x256`，`status` 为 `1x8`，`noise` 为 `1x20x8x2`。导出时需把 ScatterND/in-place 写法改为拼接，并把固定 AdaptiveAvgPool 改为静态深度卷积。将模型放到：

```text
build/diffusiondrive_navsim_bpu_clean_float.onnx
```

## 校准数据

使用不少于 100 个真实 NAVSIM mini 样本。四个输入分别放在 `calibration_data/camera`、`lidar`、`status`、`noise`，文件为对应的 float32 `.npy`。

## 编译

在 v3.7.0 容器、本目录下执行：

```bash
hb_compile -c configs/diffusiondrive_r34_256x1024_s600.yaml
```

最终配置对整张计算图使用 INT16 激活和 max 校准。这样做是因为：全 INT8 PTQ 的最终 BEV cosine 只有 `0.370948`；只把 BEV 末端四个节点改为 INT16 后，cosine 仍只有 `0.371840`，mean IoU 为 `0.143013`。上游融合特征 `/_backbone/Add_6` 已经发生明显失真，因此只调整语义头无法恢复 BEV。

改成全 INT16 + max 后，板端相对浮点的 BEV cosine 达到 `0.998918`，像素一致率达到 `0.944061`，mean IoU 达到 `0.868425`。编译报告与板端 profiler 均确认所有分段在 BPU 上执行，CPU 推理耗时为 `0.0 ms`。

生成后使用以下命令检查：

```bash
hrt_model_exec model_info --model_file build/hbm/diffusiondrive_r34_256x1024_s600.hbm
hrt_model_exec perf --model_file build/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 1 --core_id 0
```
