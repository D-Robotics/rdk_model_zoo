# YOLO26 RDK Model Zoo Sample

This directory contains scripts for exporting, converting, and running inference for YOLO26 models on RDK X5 hardware.

## Files
- `export_yolo26_bpu.py`: Exports a YOLO26 `.pt` model to an optimized `.onnx` format for BPU.
- `mapper.py`: Converts the `.onnx` model to a `.bin` model using `hb_mapper`.
- `YOLO26_Inference.py`: Performs inference on the RDK board using the `.bin` model.

## Quick Start

### 1. Export ONNX (PC/Workstation)
Ensure you have `ultralytics` installed.
```bash
python3 export_yolo26_bpu.py
```

### 2. Convert to Bin (OpenExplore Docker)
Use the `hb_mapper` tool to convert the ONNX model.
```bash
python3 mapper.py --onnx yolo26n_bpu.onnx --cal-images /path/to/calibration_data
```

### 3. Inference (RDK Board)
Copy the generated `.bin` model to your RDK board and run:
```bash
python3 YOLO26_Inference.py --model-path yolo26n_bayese_640x640_nv12.bin --test-img bus.jpg
```

## Post-processing Details
The `YOLO26_Inference.py` script includes an optimized post-processing implementation specifically for BPU-exported models, handling the decoded box and class outputs.
