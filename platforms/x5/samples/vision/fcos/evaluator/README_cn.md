# FCOS 模型评测

本目录用于记录 FCOS 在 RDK X5 上的 benchmark 与验证说明。

## 评测前提

- `RDK OS >= 3.5.0`
- 板端 Python 3 环境
- 如需复现 COCO mAP，请安装 `pycocotools`

## 数据集准备

- 检测 benchmark 基于 COCO 验证集
- 运行示例默认测试图位于 `../test_data/bus.jpg`

## Benchmark 结果

### RDK X5 性能数据

| 模型 | 尺寸 | 类别数 | BPU 延迟 / 吞吐 | Python 后处理 |
| --- | --- | --- | --- | --- |
| `fcos_efficientnetb0` | 512x512 | 80 | `3.3 ms / 298.0 FPS (1 thread)`<br>`6.2 ms / 323.0 FPS (2 threads)` | `9 ms` |
| `fcos_efficientnetb2` | 768x768 | 80 | `14.4 ms / 69.5 FPS (1 thread)`<br>`28.1 ms / 70.9 FPS (2 threads)` | `16 ms` |
| `fcos_efficientnetb3` | 896x896 | 80 | `26.1 ms / 38.2 FPS (1 thread)`<br>`51.6 ms / 38.7 FPS (2 threads)` | `20 ms` |

## 性能测试说明

- 测试平台：RDK X5 / RDK X5 Module
- CPU 条件：A55 全核 performance 模式
- BPU 条件：Bayes-e 最佳运行状态
- 测试命令示例：
  ```bash
  hrt_model_exec perf --thread_num 2 --model_file fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
  ```

## 精度测试说明

- 精度以 COCO 验证集和 `pycocotools` 为基准
- 当前发布的 FCOS X5 sample 以部署链路和 benchmark 复现为主
- 如需严格复现 mAP，请保持与 runtime 示例一致的输入尺寸、NV12 预处理和阈值设置
