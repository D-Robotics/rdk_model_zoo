English | [简体中文](./README_cn.md)

# YOLOv13 iMoonLab Evaluation

This directory collects the performance data, accuracy data, and evaluation notes for YOLOv13 Detect on RDK S100 and RDK S100P.

## Performance Data

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

## Performance Test Notes

1. The data above corresponds to YUV420SP (NV12) input models. NCHW RGB input models usually show no significant performance gap.
2. Single-thread latency represents the ideal per-frame latency on one thread and one BPU core. Multi-thread throughput represents the total FPS when multiple threads feed the BPU concurrently.
3. The S100 / S100P BPU reaches near-saturation quickly, so the table keeps the thread counts where throughput already becomes stable.
4. Reference commands:

```bash
hrt_model_exec perf --thread_num 2 --model_file yolo13n_detect_nashe_640x640_nv12.hbm
python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```

5. Reference board setup:

- S100P: 6 x A78AE @ 2.0GHz, Performance governor; 1 x Nash-m @ 1.5GHz, 128 TOPS @ int8
- S100: 6 x A78AE @ 1.5GHz, Performance governor; 1 x Nash-e @ 1.0GHz, 80 TOPS @ int8

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```

## Accuracy Data

### RDK S100 / RDK S100P

Object Detection (COCO2017)

| Model | Pytorch | YUV420SP<br>Python | YUV420SP<br>C/C++ | NCHWRGB<br>C/C++ |
|---|---:|---:|---:|---:|
| YOLOv13n | 0.342 | 0.319 (93.27%) | (%) | (%) |
| YOLOv13s | 0.402 | 0.381 (94.78%) | (%) | (%) |
| YOLOv13l | 0.458 | 0.443 (96.73%) | (%) | (%) |
| YOLOv13x | 0.473 | 0.458 (96.83%) | (%) | (%) |

## Accuracy Test Notes

1. The accuracy data is computed with the official unmodified Microsoft `pycocotools`, using `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
2. The evaluation uses all 5000 images from `COCO2017 val`, runs inference on board, dumps JSON results, and evaluates them with `pycocotools` using `score=0.25` and `nms=0.7`.
3. `pycocotools` AP is usually lower than the Ultralytics built-in numbers due to different area calculation rules. The important signal here is the relative gap between floating-point and quantized models.
4. Some accuracy loss appears when converting NCHW RGB888 inputs to YUV420SP inputs for BPU deployment because of color space conversion. This can be reduced if that transformation is considered during training.
5. Python and C/C++ runtime accuracy may differ slightly because of differences in memory copy and floating-point handling.
6. Evaluation scripts can be referenced here: <https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools>
7. The table reflects PTQ results compiled with 50 calibration images. It represents a practical first-pass compilation baseline instead of the upper bound of achievable accuracy.

## Result Check

With the default `test_data/kite.jpg`, the runtime should output boxes that match the visible objects in the image and save a rendered result image. If the boxes are clearly misplaced, all classes are wrong, scores are abnormally low, or the result is empty, check:

- the ONNX export output order
- whether `remove_node_name` matches the current ONNX
- whether `score-thres` and `nms-thres` match the evaluation assumptions

## License

This directory follows the repository top-level `LICENSE`.
