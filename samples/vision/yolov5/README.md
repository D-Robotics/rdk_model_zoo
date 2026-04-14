English | [简体中文](./README_cn.md)

# YOLOv5 Model Description

This directory describes the complete workflow of YOLOv5 in this Model Zoo, including: algorithm introduction, model conversion, runtime inference (Python), reusable pre/post-processing interfaces, and model evaluation steps.

---

## Algorithm Overview

YOLOv5 is a one-stage object detection algorithm in the YOLO series. It predicts bounding boxes and category scores directly on multi-scale feature maps, and is widely used in generic object detection tasks because of its balance between accuracy and runtime efficiency.

### Algorithm Functionality

YOLOv5 can complete the following task:

- Object detection

### Original Resources

- Official Repo: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

## Directory Structure

This directory contains:

```bash
.
├── conversion/                        # Model conversion process
│   ├── yolov5_detect_bayese_640x640_nchw.yaml
│   ├── yolov5_detect_bayese_640x640_nv12.yaml
│   ├── README.md
│   └── README_cn.md
├── evaluator/                         # Accuracy and performance evaluation
│   ├── README.md
│   └── README_cn.md
├── model/                             # Model files and download scripts
│   ├── download.sh
│   ├── README.md
│   └── README_cn.md
├── runtime/                           # Inference examples
│   ├── cpp/
│   └── python/
│       ├── main.py
│       ├── yolov5_det.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data/                         # Sample images and benchmark assets
│   ├── bus.jpg
│   └── *.png / *.jpg
├── README.md
└── README_cn.md
```

---

## QuickStart

For a quick experience, the Python sample provides a `run.sh` script that allows you to run the default model with one command.

### Python

- Go to the `python` directory under `runtime` and run the `run.sh` script:
  ```bash
  cd runtime/python
  chmod +x run.sh
  ./run.sh
  ```
- For detailed usage of the Python code, please refer to `runtime/python/README.md`

---

## Model Conversion

ModelZoo provides pre-adapted BIN model files. Users can directly run the download script in the `model` directory to download and use them.

If you need to reproduce the conversion flow from the YOLOv5 project, refer to `conversion/README.md` for:

- `v2.0` and `v7.0` branch preparation
- ONNX export adjustments for NHWC detection outputs
- `hb_mapper checker` and `hb_mapper makertbin`
- `hb_perf` and `hrt_model_exec` validation
- output tensor protocol used by the Python runtime

---

## Runtime Inference

YOLOv5 runtime inference sample currently provides a Python implementation.

### Python Version

- Provided in script form, suitable for rapid verification of model effects and algorithm flows
- The sample demonstrates the complete process of model loading, inference execution, post-processing, and result visualization
- For detailed usage, parameter descriptions, and interface specifications, please refer to `runtime/python/README.md`

---

## Evaluator

`evaluator/` is used for benchmark data, runtime validation records, and performance descriptions. Please refer to `evaluator/README.md` for details.

---

## Performance Data

The following table shows the reference performance data of YOLOv5 models on the RDK X5 platform.

| Model | Size | Params | BPU Throughput | Python Post-process |
| --- | --- | ---: | --- | --- |
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

---

## License

Follows the Model Zoo top-level License.
