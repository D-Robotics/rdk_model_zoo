English | [简体中文](./README_cn.md)

# Model Files

This directory contains the pre-converted YOLO26 model files and download scripts for RDK S100/S100P.

## Directory Structure

```text
.
├── download_model.sh      # Download pre-compiled models
└── README.md              # Documentation
```

## Download Models

To download the pre-compiled YOLO26 models for RDK S100/S100P, run:

```bash
bash download_model.sh
```

The script auto-detects the board and downloads models to `./nash-e/` or `./nash-m/`.

## Notes

- The RDK S100/S100P inference model format is `.hbm`.
- All models use NV12 input (Y + UV as two separate tensors).
- The model suffix differs by platform: `nashe` for S100, `nashm` for S100P.

## Published Models

### Detection
- `yolo26n_detect_{nashe|nashm}_640x640_nv12.hbm`

### Instance Segmentation
- `yolo26n_seg_{nashe|nashm}_640x640_nv12.hbm`

### Pose Estimation
- `yolo26n_pose_{nashe|nashm}_640x640_nv12.hbm`

### OBB
- `yolo26n_obb_{nashe|nashm}_640x640_nv12.hbm`

### Classification
- `yolo26n_cls_{nashe|nashm}_224x224_nv12.hbm`
