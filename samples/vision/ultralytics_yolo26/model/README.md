# Model Files

This directory contains the pre-converted YOLO26 model files and download scripts for RDK X5.

## Directory Structure

```text
.
├── download_model.sh      # Download nano models
├── fulldownload.sh        # Download all models
└── README.md              # Documentation
```

## Download Models

To download the pre-compiled YOLO26 models for RDK X5, run:

```bash
chmod +x download_model.sh
./download_model.sh
```

To download all `n / s / m / l / x` models, run:

```bash
chmod +x fulldownload.sh
./fulldownload.sh
```

## Notes

- The RDK X5 inference model format is `.bin`.
- All models use NV12 input.
- The current model download base URL is:
  `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/`

## Model List

### Detection
- `yolo26{n/s/m/l/x}_detect_bayese_640x640_nv12.bin`

### Instance Segmentation
- `yolo26{n/s/m/l/x}_seg_bayese_640x640_nv12.bin`

### Pose Estimation
- `yolo26{n/s/m/l/x}_pose_bayese_640x640_nv12.bin`

### OBB
- `yolo26{n/s/m/l/x}_obb_bayese_640x640_nv12.bin`

### Classification
- `yolo26{n/s/m/l/x}_cls_bayese_224x224_nv12.bin`
