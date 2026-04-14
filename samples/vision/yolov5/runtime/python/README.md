English | [简体中文](./README_cn.md)

# YOLOv5 Runtime

This directory provides the Python runtime entry for the `yolov5` sample on `RDK X5`.

## Files

- `main.py`: Unified Python inference entry.
- `yolov5_det.py`: YOLOv5 detection wrapper based on `hbm_runtime`.
- `run.sh`: One-click runtime script.

## Quick Start

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

The script downloads the default `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` model into `../../model/` if needed and saves the result image into `../../test_data/result.jpg`.

## Manual Execution

```bash
python3 main.py
python3 main.py --model-path ../../model/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin
python3 main.py --img-save-path ../../test_data/result_custom.jpg
```

## Command Line Arguments

```bash
python3 main.py -h
```

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the BPU quantized YOLOv5 BIN model. | `../../model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `--label-file` | Path to the class names file used by visualization. | `../../../../../datasets/coco/coco_classes.names` |
| `--priority` | Model priority for runtime scheduling. | `0` |
| `--bpu-cores` | BPU core indexes used for inference. | `0` |
| `--test-img` | Path to the test input image. | `../../test_data/bus.jpg` |
| `--img-save-path` | Path to save the output image. | `../../test_data/result.jpg` |
| `--resize-type` | Resize strategy (`0`: direct resize, `1`: letterbox). | `0` |
| `--classes-num` | Number of detection classes. | `80` |
| `--score-thres` | Score threshold used to filter predictions. | `0.25` |
| `--nms-thres` | IoU threshold used by Non-Maximum Suppression. | `0.45` |
| `--anchors` | Comma-separated YOLOv5 anchor values. | `10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326` |
| `--strides` | Comma-separated detection head strides. | `8,16,32` |
