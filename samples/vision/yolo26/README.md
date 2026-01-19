English | [简体中文](./README_cn.md)

# YOLO26 RDK Model Zoo Sample

This directory contains scripts for exporting, converting, and running inference for YOLO26 models (Detection, Segmentation, Pose, Classification) on RDK X5 hardware.

## Project Structure

```
.
├── cls/                # Classification task scripts
│   ├── export_yolo26_cls_bpu.py
│   └── YOLO26_Cls_Inference.py
├── detect/             # Detection task scripts
│   ├── export_yolo26_detect_bpu.py
│   └── YOLO26_Detect_Inference.py
├── pose/               # Pose Estimation task scripts
│   ├── export_yolo26_pose_bpu.py
│   └── YOLO26_Pose_Inference.py
├── seg/                # Segmentation task scripts
│   ├── export_yolo26_seg_bpu.py
│   └── YOLO26_Seg_Inference.py
├── mapper.py           # Common tool for converting ONNX to BPU BIN model
└── README.md
```

## Workflow Overview

The deployment process involves three main steps:
1.  **Export**: Export the PyTorch (`.pt`) model to ONNX format optimized for BPU.
2.  **Convert**: Convert the ONNX model to a BPU-compatible `.bin` model using `mapper.py`.
3.  **Inference**: Run inference on the RDK board using the specific task inference script.

## 1. Detection (Detect)

### Step 1: Export ONNX
```bash
cd detect
python3 export_yolo26_detect_bpu.py
# Generates: yolo26n_detect_bpu.onnx (example)
```

### Step 2: Convert to BIN
Use the `mapper.py` tool in the root of this sample.
```bash
cd ..
python3 mapper.py --onnx detect/yolo26n_detect_bpu.onnx --cal-images /path/to/calibration_data
# Generates: yolo26n_detect_bayese_640x640_nv12.bin
```

### Step 3: Inference on RDK
Copy the `.bin` file to your RDK board.
```bash
python3 detect/YOLO26_Detect_Inference.py --model-path yolo26n_detect_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 2. Segmentation (Seg)

### Step 1: Export ONNX
```bash
cd seg
python3 export_yolo26_seg_bpu.py
```

### Step 2: Convert to BIN
```bash
cd ..
python3 mapper.py --onnx seg/yolo26n_seg_bpu.onnx --cal-images /path/to/calibration_data
```

### Step 3: Inference on RDK
```bash
python3 seg/YOLO26_Seg_Inference.py --model-path yolo26n_seg_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 3. Pose Estimation (Pose)

### Step 1: Export ONNX
```bash
cd pose
python3 export_yolo26_pose_bpu.py
```

### Step 2: Convert to BIN
```bash
cd ..
python3 mapper.py --onnx pose/yolo26n_pose_bpu.onnx --cal-images /path/to/calibration_data
```

### Step 3: Inference on RDK
```bash
python3 pose/YOLO26_Pose_Inference.py --model-path yolo26n_pose_bayese_640x640_nv12.bin --test-img bus.jpg
```

---

## 4. Classification (Cls)

### Step 1: Export ONNX
```bash
cd cls
python3 export_yolo26_cls_bpu.py
```

### Step 2: Convert to BIN
```bash
cd ..
python3 mapper.py --onnx cls/yolo26n_cls_bpu.onnx --cal-images /path/to/calibration_data
```

### Step 3: Inference on RDK
```bash
python3 cls/YOLO26_Cls_Inference.py --model-path yolo26n_cls_bayese_224x224_nv12.bin --test-img bus.jpg
```

## Note on `mapper.py`
The `mapper.py` script is a wrapper around `hb_mapper`. It handles the calibration data preparation and model conversion.
- `--onnx`: Path to the input ONNX model.
- `--cal-images`: Path to the directory containing calibration images (JPG/PNG).
- `--output-dir`: (Optional) Directory to save the output `.bin` model.
- `--quantized`: (Optional) `int8` (default) or `int16`.

Ensure you run `mapper.py` in an environment where `hb_mapper` is installed (e.g., OpenExplore Docker).
