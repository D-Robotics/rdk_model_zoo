[English](./README.md) | 简体中文

# EfficientSAM 模型评估

本目录记录量化 EfficientSAM 编码器与解码器的板端性能评估。该 sample 仅提供延迟/吞吐基准，不包含精度评估脚本（稀疏点提示分割在本 sample 中无标准评估脚本）。

## 环境准备

- **板卡**：已安装 RDK-S 运行时的 RDK S100 / S100P / S600。
- **工具**：`hrt_model_exec`，由 RDK-S 开发套件提供，非 pip 安装。
- **模型**：`../model/nash-e/`、`../model/nash-m/` 或 `../model/nash-p/` 下编译好的 `.hbm` 文件。

## 使用方法

用板端 `hrt_model_exec perf` 分别测量各模型：

```bash
# 编码器(单线程延迟)
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm --thread_num 1

# 编码器(多线程吞吐)
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm --thread_num 2

# 解码器
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_decoder_512_nashe.hbm --thread_num 1
```

按实际情况将 `nash-e` 替换为 `nash-m`（S100P）或 `nash-p`（S600）。下表的基准在 S100/S100P 上使用 1 和 2 线程，S600 上使用 1 和 12 线程（多核，通过 `--core_id 1,2,3,4`）。

## 基准结果

### RDK S100 性能数据

| 设备 | 模型 | 尺寸 <br> (Pixels) | 类别数 | BPU 任务延迟 / <br> BPU 吞吐量 (线程) | CPU 延迟 | 参数量 <br> (M) | 计算量 <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | EfficientSAM Encoder | 512x512 | - | 11.78 ms / 84.75 FPS (1 thread) <br> 22.93 ms / 86.95 FPS (2 threads) | - | 6.16 | 22.19 |
| S100 | EfficientSAM Decoder | 256x32x32 (embedding) | - | 3.25 ms / 306.31 FPS (1 thread) <br> 5.94 ms / 334.44 FPS (2 threads) | - | 4.06 | 0.98 |

### RDK S100P 性能数据

| 设备 | 模型 | 尺寸 <br> (Pixels) | 类别数 | BPU 任务延迟 / <br> BPU 吞吐量 (线程) | CPU 延迟 | 参数量 <br> (M) | 计算量 <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | EfficientSAM Encoder | 512x512 | - | 9.36 ms / 106.69 FPS (1 thread) <br> 18.20 ms / 109.52 FPS (2 threads) | - | 6.16 | 22.19 |
| S100P | EfficientSAM Decoder | 256x32x32 (embedding) | - | 2.49 ms / 399.96 FPS (1 thread) <br> 4.47 ms / 445.74 FPS (2 threads) | - | 4.06 | 0.98 |

### RDK S600 性能数据

| 设备 | 模型 | 尺寸 <br> (Pixels) | 类别数 | BPU 任务延迟 / <br> BPU 吞吐量 (线程) | CPU 延迟 | 参数量 <br> (M) | 计算量 <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | EfficientSAM Encoder | 512x512 | - | 6.72 ms / 148.60 FPS (1 thread) <br> 19.58 ms / 598.44 FPS (12 threads) | - | 6.16 | 22.19 |
| S600 | EfficientSAM Decoder | 256x32x32 (embedding) | - | 1.50 ms / 662.55 FPS (1 thread) <br> 4.08 ms / 2831.10 FPS (12 threads) | - | 4.06 | 0.98 |

## 性能测试说明

- **设备 (Device)**：测试所用的硬件设备（S100、S100P、S600）。
- **模型 (Model)**：`EfficientSAM Encoder` 为 ViT-Tiny 图像编码器（`efficient_sam_vitt_encoder_512x512_*.hbm`）；`EfficientSAM Decoder` 为固定提示掩码解码器（`efficient_sam_vitt_decoder_512_*.hbm`）。
- **尺寸 (Pixels)**：正方形输入分辨率。编码器输入 `512×512` RGB 图像；解码器输入由编码器对同一 `512×512` 图像产出的 `256×32×32` 图像嵌入。
- **类别数 (Classes)**：EfficientSAM 基于提示且类别无关，因此无固定类别数（`-`）。
- **输入类型**：与检测类 sample 不同，SAM 的输入为 float32 NCHW tensor——编码器接收 float32 RGB 图像（缩放到 `1/255`），解码器接收 float32 featuremap——而非 NV12。
- **BPU 任务延迟 / 吞吐量**：
  - **单线程延迟**：单帧图像在单线程、单 BPU 核心下的理想推理延迟，由 `hrt_model_exec perf --thread_num 1 --model_file <model.hbm>` 测得，统计从任务提交到完成的时间，含缓存预热。
  - **多线程吞吐量**：多线程同时向 BPU 提交任务时达到的每秒处理帧数（FPS）。S100/S100P 取 2 线程，S600 取 12 线程（多核）可获得最佳 BPU 利用率；S600 多核测量使用 `--core_id 1,2,3,4`。
  - **测试命令**：`hrt_model_exec perf --model_file <model.hbm> --thread_num N`。
- **CPU 延迟 (单核)**：CPU 侧预处理/后处理（编码器为图像缩放与归一化，解码器为掩码上采样）。可选且依赖于宿主流水线，故表中不列出（见下方流水线说明）。
- **流水线延迟**：runtime 先跑编码器再跑解码器，因此端到端掩码延迟 = 编码器延迟 + 解码器延迟 + CPU 开销（为便于跨板卡比较，两个模型分开测量）。
- **内存管理**：流式推理中输入/输出内存一次性分配并跨帧复用；延迟不含内存分配与释放时间。
- **参数量 (M) & 计算量 (G)**：原始 FP32 模型在 `512×512` 输入下的参数量与计算量。计算量为 MACs 派生值（`2×MACs`），与 Model Zoo 中采用的 Ultralytics 导出日志口径一致。解码器计算量在其 `256×32×32` embedding 输入上测得。

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。