English | [简体中文](./README_cn.md)

# ByteTrack Model Description

ByteTrack is a multi-object tracking algorithm. This sample runs a YOLOv5x detector on RDK S100/S100P/S600 to produce pedestrian detections, then uses BYTETracker to associate tracks and output a video with tracking IDs.

## Algorithm Overview

Multi-object tracking (MOT) estimates object boxes and identities in videos. Many tracking methods only associate high-score detections and discard low-score detections, which may lose occluded objects and fragment tracks. ByteTrack introduces BYTE (Tracking By associating Almost Every Detection Box), a strategy that associates almost every detection box to recover occluded objects while filtering background detections.

![ByteTrack association](./test_data/readme_img/image1.png)

The core ByteTrack flow includes:

- Keep both high-score and low-score detections.
- First association: match high-confidence detections with existing tracks.
- Second association: match unmatched tracks with low-confidence detections using IoU.
- Initialize new tracks only from unmatched high-score detections.

Paper: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

## Algorithm Capabilities

- Pedestrian multi-object tracking
- Output target boxes and unique track IDs
- Generate a video with tracking results

## Algorithm Features

- Recovers occluded targets with low-score detections.
- Uses two-stage matching to reduce track fragmentation.
- Tracker overhead is small; detection performance mainly depends on the YOLO model.

## Directory Structure

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

## Quick Start

```bash
cd samples/vision/bytetrack/runtime/python
bash run.sh
```

The script downloads `../../model/s100/yolov5x_672x672_nv12.hbm` and `../../test_data/track_test.mp4`, then generates `result.mp4`.

## Model Conversion

ByteTrack itself is a post-processing tracker. The converted model is the upstream YOLO detector. See [conversion/README.md](./conversion/README.md) for detector conversion notes and OE resource entry points.

## Runtime

See [runtime/python/README.md](./runtime/python/README.md) for Python runtime arguments, direct `python3 main.py` examples, and API notes.

## Model Evaluation

See [evaluator/README.md](./evaluator/README.md) for MOT metrics, tracker performance records, result checks, and tuning notes.

## Inference Result

After successful execution, pedestrians in the output video should be tracked stably, and each pedestrian box should have a unique ID. If too few boxes are detected, lower `--score-thres` or `--track-thresh`; if IDs switch frequently, tune `--match-thresh` or `--track-buffer`.

## License

This sample is licensed under the [Apache 2.0 License](../../../LICENSE).
