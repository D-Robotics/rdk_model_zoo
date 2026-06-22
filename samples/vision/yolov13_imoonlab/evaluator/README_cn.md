[English](./README.md) | 简体中文

# YOLOv13 iMoonLab 模型评估

本目录汇总 YOLOv13 Detect 在 RDK S100 与 RDK S100P 上的性能数据、精度数据和评测说明。

## 性能数据

### RDK S100P

| Model | Size(Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|---|---|---:|---|---:|---:|---:|
| YOLOv13n | 640x640 | 80 | 2.8 ms / 353.5 FPS (1 thread)<br>3.9 ms / 509.0 FPS (2 threads) | 2 ms | 2.5 | 6.4 |
| YOLOv13s | 640x640 | 80 | 4.3 ms / 231.7 FPS (1 thread)<br>7.1 ms / 278.5 FPS (2 threads) | 2 ms | 9.0 | 20.8 |
| YOLOv13l | 640x640 | 80 | 12.1 ms / 82.5 FPS (1 thread)<br>22.7 ms / 87.7 FPS (2 threads) | 2 ms | 27.6 | 88.4 |
| YOLOv13x | 640x640 | 80 | 19.7 ms / 50.7 FPS (1 thread)<br>37.8 ms / 52.7 FPS (2 threads) | 2 ms | 64.0 | 199.2 |

### RDK S100

| Model | Size(Pixels) | Classes | BPU Task Latency /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|---|---|---:|---|---:|---:|---:|
| YOLOv13n | 640x640 | 80 | 3.8 ms / 262.0 FPS (1 thread)<br>5.2 ms / 378.3 FPS (2 threads) | 2 ms | 2.5 | 6.4 |
| YOLOv13s | 640x640 | 80 | 5.8 ms / 169.5 FPS (1 thread)<br>9.7 ms / 204.9 FPS (2 threads) | 2 ms | 9.0 | 20.8 |
| YOLOv13l | 640x640 | 80 | 16.6 ms / 59.8 FPS (1 thread)<br>31.1 ms / 63.9 FPS (2 threads) | 2 ms | 27.6 | 88.4 |
| YOLOv13x | 640x640 | 80 | 26.9 ms / 37.1 FPS (1 thread)<br>51.6 ms / 38.6 FPS (2 threads) | 2 ms | 64.0 | 199.2 |

## 性能测试说明

1. 上述数据对应 YUV420SP（NV12）输入模型。NCHW RGB 输入模型的性能通常没有明显差异。
2. 单线程延迟表示单帧、单线程、单 BPU 核心情况下的理想推理延迟；多线程吞吐量表示多个线程同时向 BPU 投递任务后的总体 FPS。
3. S100 / S100P 的 BPU 吞吐能力较强，通常 2 个线程即可接近满载，因此表格只保留吞吐量趋于稳定时的数据。
4. 参考性能测试命令：

```bash
hrt_model_exec perf --thread_num 2 --model_file yolo13n_detect_nashe_640x640_nv12.hbm
python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```

5. 参考测试环境：

- S100P：6 x A78AE @ 2.0GHz，Performance 调度；1 x Nash-m @ 1.5GHz，128 TOPS @ int8
- S100：6 x A78AE @ 1.5GHz，Performance 调度；1 x Nash-e @ 1.0GHz，80 TOPS @ int8

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```

## 精度数据

### RDK S100 / RDK S100P

Object Detection (COCO2017)

| Model | Pytorch | YUV420SP<br>Python | YUV420SP<br>C/C++ | NCHWRGB<br>C/C++ |
|---|---:|---:|---:|---:|
| YOLOv13n | 0.342 | 0.319 (93.27%) | (%) | (%) |
| YOLOv13s | 0.402 | 0.381 (94.78%) | (%) | (%) |
| YOLOv13l | 0.458 | 0.443 (96.73%) | (%) | (%) |
| YOLOv13x | 0.473 | 0.458 (96.83%) | (%) | (%) |

## 精度测试说明

1. 精度数据基于微软官方原版 `pycocotools` 计算，指标取 `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`。
2. 使用 `COCO2017 val` 的 5000 张图片在板端直接推理，保存为 JSON 后再送入 `pycocotools` 评估，阈值设定为 `score=0.25`、`nms=0.7`。
3. `pycocotools` 的 AP 通常会低于 Ultralytics 自带评估结果，这与面积计算方式不同有关，关注点应放在浮点模型与定点模型之间的相对精度损失。
4. 当 BPU 模型从 NCHW RGB888 转为 YUV420SP 输入后，会因为色彩空间变化产生少量精度损失，训练阶段可通过引入对应变换缓解。
5. Python 接口和 C/C++ 接口的精度结果可能有极细微差异，主要来自数据拷贝和浮点处理过程的差别。
6. 评测脚本可参考：<https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools>
7. 本表结果来自 PTQ（训练后量化）配置，使用 50 张图片进行校准和编译，用于反映常规首次编译情况下的精度损失，不代表精度上限。

## 结果检查

使用默认测试图片 `test_data/kite.jpg` 时，应检测到与图片内容匹配的目标，并生成带框的结果图。若检测框明显错位、类别全错、分数异常偏低或结果为空，应重点检查：

- ONNX 导出后的输出顺序
- `remove_node_name` 是否与当前 ONNX 匹配
- 运行时 `score-thres`、`nms-thres` 是否与评测假设一致

## License

本目录内容遵循仓库顶层 `LICENSE`。
