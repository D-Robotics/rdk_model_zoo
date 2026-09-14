[English](./README.md) | 简体中文

# 模型评测

## Runtime 性能

测试设备为 S100 V1P0，系统 RDK OS 4.0.5-Beta，UCP 3.13.6 / HBRT 4.7.5。
HBM 使用 OE 3.7.0、HBDK 4.7.5、INT8 KL，batch=1、640×640 NV12、4585 类。
测试于 2026-09-08 使用 `hrt_model_exec perf` 完成：200 帧，开启预热，
thread_num=1，core_id=0。

这里只展示**模型 Runtime** 延迟和吞吐量，不包含应用前处理、输出反量化、
后处理、可视化或 I/O，也不代表 Python/C++ 端到端 FPS。

| 板型 | 模型 | Runtime 延迟（ms） | Runtime FPS |
|---|---|---:|---:|
| S100 | YOLOE-26n Seg PF | 4.943 | 200.74 |
| S100 | YOLOE-26s Seg PF | 9.944 | 100.08 |
| S100 | YOLOE-26m Seg PF | 11.765 | 84.55 |
| S100 | YOLOE-26l Seg PF | 13.417 | 74.18 |
| S100 | YOLOE-26x Seg PF | 22.013 | 45.31 |
| S100P | YOLOE-26n/s/m/l/x Seg PF | 未测量 | 未测量 |

```bash
hrt_model_exec perf --model_file /path/model.hbm \
  --thread_num 1 --core_id 0 --frame_count 200 --enable_warmup true \
  --profile_path ./profile
```

## 精度验证状态

此前五个 S100 模型的 Python/C++ 实测返回了相同的检测数量和类别，
框与分数差异小于 1e-7，mask 面积一致。面积相等不代表 mask 逐像素完全相同。
S100P 模型已通过编译与模型 I/O 检查，但尚无 S100P 板端 benchmark 结果。

校准使用了 100 张具有代表性的 COCO train2017 图片。数据集框与 mask 的 mAP
尚未通过验收，INT8 基线与浮点推理存在数值差异。生产使用前应在独立的带标注
评测集上验证，并显式建立 PF 词表到数据集类别的映射：PF 类别 ID 不是 COCO ID。
本说明不提供未经测量的 mAP 或 S100P Runtime 数值。
