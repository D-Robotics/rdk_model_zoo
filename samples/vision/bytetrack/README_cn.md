[English](./README.md) | 简体中文

# ByteTrack 模型说明

ByteTrack 是多目标跟踪算法。本 sample 在 RDK S100/S100P 上使用 YOLOv5x 检测模型生成行人检测框，再通过 BYTETracker 进行轨迹关联，输出带目标 ID 的跟踪视频。

## 算法介绍

多目标跟踪（MOT）用于估计视频中物体的边界框和身份。多数跟踪方法只关联高分检测框，低分检测框通常被丢弃，这会导致被遮挡目标丢失和轨迹碎片化。ByteTrack 提出 BYTE（Tracking By associating Almost Every Detection Box）关联策略，通过关联几乎所有检测框来恢复被遮挡目标并过滤背景检测。

![ByteTrack association](./test_data/readme_img/image1.png)

ByteTrack 的核心流程包括：

- 保留高分和低分检测框；
- 第一次关联：高置信度检测框与已有轨迹匹配；
- 第二次关联：未匹配轨迹与低置信度检测框通过 IoU 匹配；
- 新轨迹只从未匹配的高分检测框初始化。

论文：[ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

## 算法功能

- 行人多目标跟踪；
- 输出目标框和唯一轨迹 ID；
- 生成带跟踪结果的视频文件。

## 算法特点

- 通过低分检测框恢复遮挡目标；
- 两阶段匹配降低轨迹碎片化；
- 跟踪器计算量小，检测性能主要由 YOLO 模型决定。

## 目录结构

```text
bytetrack/
├── conversion/
├── evaluator/
├── model/
├── runtime/
│   └── python/
├── test_data/
│   └── readme_img/
├── README.md
└── README_cn.md
```

## 快速体验

```bash
cd samples/vision/bytetrack/runtime/python
bash run.sh
```

脚本会下载 `../../model/s100/yolov5x_672x672_nv12.hbm` 和 `../../test_data/track_test.mp4`，并生成 `result.mp4`。

## 模型转换

ByteTrack 本身是后处理跟踪算法，转换对象是前级 YOLO 检测模型。检测模型转换说明和 OE 资源入口见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

Python 运行参数、直接 `python3 main.py` 示例和接口说明见 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型评估

MOT 指标、跟踪器性能记录、结果检查和调参建议见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 推理结果

运行成功后，输出视频中的行人应被稳定框出，并且每个行人框旁边有唯一 ID。若检测框很少，可以降低 `--score-thres` 或 `--track-thresh`；若 ID 频繁切换，可以调整 `--match-thresh` 或 `--track-buffer`。

## License

本 sample 遵循 [Apache 2.0 License](../../../LICENSE)。
