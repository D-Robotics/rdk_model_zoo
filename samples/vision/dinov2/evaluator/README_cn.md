[English](./README.md) | 简体中文

# DINOv2 模型评测指南

本目录记录 DINOv2 ViT-S/14 int16 模型在 RDK S100/S100P/S600 上的实测性能
与精度，以及复现命令。

## 性能数据

使用 `hrt_model_exec perf` 测量（200 帧，锁 performance governor）。

| 设备 | 模型 | 输入尺寸 | BPU 任务延迟 / BPU 吞吐 |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS（1 线程）<br> 288.26 FPS（2 线程） |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS（1 线程）<br> 357.63 FPS（2 线程） |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS（1 线程）<br> 1898.42 FPS（12 线程，`--core_id 1,2,3,4`） |

模型参数量：22.06 M。延迟为纯 BPU 前向；板端 CPU 预处理另计。标准 CPU
预处理为：OpenCV BGR 转 RGB，按比例 bicubic 将短边 resize 到 256，中心 crop
为 224 x 224，`/255`、ImageNet mean/std 归一化，并转换为 contiguous float32
NCHW。

## 性能测试方法

先锁频，再运行 perf 工具。以下命令完整对应表格中的线程和核心设置：

```bash
# RDK S100（nash-e）：1 线程和 2 线程
hrt_model_exec perf \
    --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
    --thread_num 2

# RDK S100P（nash-m）：1 线程和 2 线程
hrt_model_exec perf \
    --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm \
    --thread_num 2

# RDK S600（nash-p）：1 线程，以及四个 BPU 核上的 12 线程
hrt_model_exec perf \
    --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm \
    --thread_num 12 \
    --core_id 1,2,3,4
```

锁频命令见仓库顶层 README FAQ。

## 精度数据

### PTQ 逐输出 cosine（工具链报告）

| 输出 | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

Quantized cosine 由工具链对拍浮点 ONNX 模型测得。相同数值已由独立导出
脚本与另一组 50 张校准图片复现。

### 板端执行 cosine（对拍浮点 ONNX）

量化模型经 `hbm_runtime` 在板端执行，与同输入的 ONNXRuntime float32
参考对拍：

| 设备 | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

复现方法：在主机用 ONNXRuntime 跑导出的浮点 ONNX 生成参考输出，将张量
与 `.hbm` 推到板端，用 `hbm_runtime.HB_HBMRuntime(...).run()` 跑相同输入，
计算板端输出与参考的 cosine 相似度。

## 结果检查

运行默认 demo，检查输出统计有限、非零、且重复运行稳定：

```bash
cd ../runtime/python
bash run.sh
```

demo 会打印输出摘要（shape / dtype / mean / std / min / max / l2_norm）
以及两张测试图之间的 cosine 相似度。相同输入的重复运行必须产生一致的
摘要。

## 许可

见 [../../../../LICENSE](../../../../LICENSE)。
