English | [简体中文](./README_cn.md)

# Model Files

This directory contains pre-compiled YOLO model files in HBM format and download scripts for RDK S100/S100P.

## Directory Structure

```text
.
├── download_model.sh      # Download a specific model
└── README.md              # Documentation
```

## Download Models

Download a specific model by specifying the SoC, YOLO variant, task, and
optional model size:

```bash
bash download_model.sh [soc] [yolo_type] [task] [model_size]

# Examples:
bash download_model.sh s100 yolo11 detect
bash download_model.sh s100p yolov8 seg
bash download_model.sh s100 yolov8 pose
bash download_model.sh s100p yolo11 cls
bash download_model.sh s100p yolov5u detect x
```

## Notes

- The RDK S100/S100P inference model format is `.hbm`.
- All models use NV12 input (Y + UV as two separate tensors).
- The model suffix differs by platform: `nashe` for S100, `nashm` for S100P.
- Main download URL base path: `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/{nash-e|nash-m}/`
- `model_size` supports `n/s/m/l/x`; YOLOv10 also supports `b`.

## Published Models

### Detection

- `yolov5{n|s|m|l|x}u_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolov8{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolov10{n|s|m|b|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`
- `yolo12{n|s|m|l|x}_detect_{nashe|nashm}_640x640_nv12.hbm`

### Instance Segmentation

- `yolov8{n|s|m|l|x}_seg_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_seg_{nashe|nashm}_640x640_nv12.hbm`

### Pose Estimation

- `yolov8{n|s|m|l|x}_pose_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_pose_{nashe|nashm}_640x640_nv12.hbm`

### Classification

- `yolov8{n|s|m|l|x}_cls_{nashe|nashm}_640x640_nv12.hbm`
- `yolo11{n|s|m|l|x}_cls_{nashe|nashm}_640x640_nv12.hbm`
