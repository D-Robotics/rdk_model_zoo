English | [简体中文](./README_cn.md)

# YOLOE Runtime

This directory provides the Python runtime entry for the `yoloe` sample on `RDK X5`.

## Files

- `main.py`: Unified Python inference entry.
- `yoloe_seg.py`: YOLOE segmentation wrapper based on `hbm_runtime`.
- `run.sh`: One-click runtime script.
- `README.md`: This document.
- `README_cn.md`: Chinese version of this document.

## Quick Start

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

The script downloads the default `yoloe_11s_seg_pf_bayese_640x640_nv12.bin` model into `../../model/` if needed and saves the result image into `../../test_data/result_seg.jpg`.

s/m/l share this entry point. An explicit `--model-path` skips the default s download.

## Manual Execution

```bash
python3 main.py
python3 main.py --model-path ../../model/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
python3 main.py --model-path ../../model/yoloe_11m_seg_pf_bayese_640x640_nv12.bin
python3 main.py --model-path ../../model/yoloe_11l_seg_pf_bayese_640x640_nv12.bin
python3 main.py --img-save-path ../../test_data/result_custom.jpg
python3 main.py --score-thres 0.3 --nms-thres 0.65
```

## Command Line Arguments

```bash
python3 main.py -h
```

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the BPU quantized YOLOE BIN model. | `../../model/yoloe_11s_seg_pf_bayese_640x640_nv12.bin` |
| `--test-img` | Path to the test input image. | `../../../../../datasets/coco/assets/bus.jpg` |
| `--label-file` | Path to the class names file used by visualization. | `../../../../../datasets/yoloe/yoloe_seg_pf_classes.names` |
| `--img-save-path` | Path to save the output image. | `../../test_data/result_seg.jpg` |
| `--priority` | Model priority for runtime scheduling. | `0` |
| `--bpu-cores` | BPU core indexes used for inference. | `0` |
| `--classes-num` | Number of segmentation classes. | `4585` |
| `--score-thres` | Score threshold used to filter predictions. | `0.25` |
| `--nms-thres` | IoU threshold used by Non-Maximum Suppression. | `0.70` |
| `--strides` | Comma-separated detection head strides. | `8,16,32` |
| `--reg` | DFL bin count per side. | `16` |
| `--mc` | Mask coefficient dimension. | `32` |

## Interface

### YOLOESegConfig

Configuration dataclass for YOLOE segmentation inference.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | *(required)* | Path to the compiled BIN model file. |
| `classes_num` | `int` | `4585` | Number of segmentation classes. |
| `score_thres` | `float` | `0.25` | Confidence threshold. |
| `nms_thres` | `float` | `0.70` | IoU threshold for NMS. |
| `resize_type` | `int` | `1` | Resize strategy (0: direct, 1: letterbox). |
| `strides` | `np.ndarray` | `[8, 16, 32]` | Detection head strides. |
| `reg` | `int` | `16` | DFL bin count per side. |
| `mc` | `int` | `32` | Mask coefficient dimension. |

### YOLOESeg

YOLOE instance segmentation wrapper based on `hbm_runtime`.

| Method | Description |
| --- | --- |
| `__init__(config)` | Initialize the model and extract runtime metadata. |
| `set_scheduling_params(priority, bpu_cores)` | Set BPU scheduling parameters. |
| `pre_process(img, resize_type, image_format)` | Convert a BGR image to packed NV12 input. |
| `forward(input_tensor)` | Execute inference on BPU. |
| `post_process(outputs, ori_img_w, ori_img_h, score_thres, nms_thres)` | Convert raw outputs to final segmentation results. Returns `(xyxy, score, cls, masks)`. |
| `predict(img, image_format, resize_type, score_thres, nms_thres)` | Run the complete segmentation pipeline. Returns `(xyxy, score, cls, masks)`. |
| `__call__(...)` | Functional-style calling, equivalent to `predict()`. |
