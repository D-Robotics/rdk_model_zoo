# Model Files

This directory contains the pre-converted Ultralytics YOLO model files and
download scripts for RDK X5.

## Directory Structure

```text
.
|-- download_model.sh      # Download default models
|-- fulldownload.sh        # Download all supported models
|-- README.md              # English documentation
`-- README_cn.md           # Chinese documentation
```

## Download Models

Download the default models used by `runtime/python/run.sh`:

```bash
chmod +x download_model.sh
./download_model.sh
```

Download all models supported by this sample:

```bash
chmod +x fulldownload.sh
./fulldownload.sh
```

## Notes

- The RDK X5 inference model format is `.bin`.
- All models use packed `NV12` input.
- The current model download base URL is:
  `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/`

## Default Models

- `detect`: `yolo11n_detect_bayese_640x640_nv12.bin`
- `seg`: `yolo11n_seg_bayese_640x640_nv12.bin`
- `pose`: `yolo11n_pose_bayese_640x640_nv12.bin`
- `cls`: `yolo11n_cls_detect_bayese_640x640_nv12.bin`

## Model List

### Detection

- `yolov5{n/s/m/l/x}u_detect_bayese_640x640_nv12.bin`
- `yolov8{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolov9{t/s/m/c/e}_detect_bayese_640x640_nv12.bin`
- `yolov10{n/s/m/b/l/x}_detect_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolo12{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`
- `yolov13{n/s/l/x}_detect_bayese_640x640_nv12.bin`

### Instance Segmentation

- `yolov8{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`
- `yolov9{c/e}_seg_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`

### Pose Estimation

- `yolov8{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`

### Classification

- `yolov8{n/s/m/l/x}_cls_detect_bayese_640x640_nv12.bin`
- `yolo11{n/s/m/l/x}_cls_detect_bayese_640x640_nv12.bin`
