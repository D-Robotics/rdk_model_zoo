# YOLOv5 评估器

<a id="dataset"></a>
## 数据集

源提供 X5 的 `test_data/bus.jpg`、S 的 `test_data/kite.jpg` 以及 `coco_classes.names`，没有带标签的 mAP benchmark harness。评估器在相同图片、target、制品和阈值下比较完整的源/统一运行，是一致性证据工具，不是 mAP 评估器。

<a id="environment"></a>
## 环境

直接在已识别的目标板上运行，需要 Python、NumPy、OpenCV 和目标 `hbm_runtime`，并导入固定源 runtime；它不是一台在外部驱动板卡的主机。主机单测注入 fake runtime，不能证明硬件或板端结果。评估器不会下载模型或图片。

<a id="command"></a>
## 评估命令

在仓库根目录、准备好精确模型且板卡身份已识别后，选择一个全新的空证据目录：

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

工具运行源路径和统一路径，把完整 native 输入/输出/结果数组保存为 `.npy`，在 `comparison.json` 记录 metadata、代码/模型/图片 hash、阈值和板卡身份；所有比较通过才返回 `0`。输出目录必须不存在。本轮没有板端运行。

<a id="metrics"></a>
## 指标

输入要求精确一致；raw 输出检查 shape/dtype，使用 `rtol=0, atol=1e-5`；结果 boxes 使用 `atol=1e-4`，scores `1e-5`，class IDs 精确相等。X5/S 必须各自按源协议比较，host fake fixture 不等于板测。历史性能列于下方，不是本轮测量。

<a id="outputs"></a>
## 输出

每个 run 目录含 `legacy_*`、`unified_*` `.npy` 数组和 `comparison.json`。模型预加载失败或比较不一致会保留 error/failed 记录并返回非零，不会把 mismatch 改成 pass。数组保留全部捕获的输入/输出 tensor 和解码结果字段。

<a id="reference-results"></a>
## 参考结果

下表完整保留源 X5 历史数据，本轮没有复测：

| 模型 | 尺寸 | 参数量 | BPU 吞吐 | Python 后处理 |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

S100/S600 以及当前源/统一板端比较均为 `not-run`。

<a id="boundaries"></a>
## 边界

评估器不下载模型、不构建转换产物，也不会把主机测试写成板端兼容。X5 刻意保留源 OpenCV XYXY-to-NMSBoxes quirk，S 使用按类 XYXY NMS；不能跨 target 要求结果相等。
