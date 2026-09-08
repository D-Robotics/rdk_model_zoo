English | [简体中文](./README_cn.md)

# YOLOE Model Description

This directory provides the complete usage guide for the YOLOE sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

This sample provides conversion and Python inference for YOLOE-11s/m/l Seg Prompt-Free with 640x640 NV12 input.

---

## Algorithm Overview

YOLOE (You Only Look Once Everything) is a zero-shot, promptable YOLO model for open-vocabulary detection and segmentation. YOLOE-11 Seg Prompt-Free supports 4585 classes without requiring any text prompt, performing instance segmentation directly on input images.

### Algorithm Functionality

YOLOE can complete the following task:

- Instance Segmentation

### Original Resources

- Paper: [YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1)
- Official Repo: [um-assn/yoloe](https://github.com/um-assn/yoloe)

---

## Directory Structure

This directory contains:

```bash
.
├── conversion/                        # Model conversion process
│   ├── onnx_export/
│   │   └── export_yoloe11seg_bpu.py
│   ├── ptq_yamls/
│   │   └── yoloe_11s_seg_pf_bayese_640x640_nv12.yaml
│   ├── README.md
│   └── README_cn.md
├── evaluator/                         # Accuracy and performance evaluation
│   ├── README.md
│   └── README_cn.md
├── model/                             # Model files and download scripts
│   ├── download_model.sh
│   ├── README.md
│   └── README_cn.md
├── runtime/                           # Inference examples
│   └── python/
│       ├── main.py
│       ├── yoloe_seg.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data/                         # Sample images and benchmark assets
│   └── *.jpg / *.png
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
- For detailed usage of the Python code, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Model Conversion

ModelZoo provides YOLOE-11s/m/l Seg-PF BIN downloads. See the download script and instructions in the `model` directory.

If you need to reproduce the conversion flow from the YOLOE project, refer to [conversion/README.md](./conversion/README.md) for:

- YOLOE environment preparation and ONNX export with BPU-compatible patches
- PTQ calibration data preparation
- `hb_mapper makertbin` conversion
- Output tensor protocol used by the Python runtime

---

## Runtime Inference

YOLOE runtime inference sample currently provides a Python implementation.

### Python Version

- Provided in script form, suitable for rapid verification of model effects and algorithm flows
- The sample demonstrates the complete process of model loading, inference execution, DFL box decoding, mask generation, NMS, and result visualization
- For detailed usage, parameter descriptions, and interface specifications, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Evaluator

`evaluator/` is used for benchmark data, runtime validation records, and performance descriptions. Please refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Performance Data

The following table retains historical YOLOE-11s reference measurements on RDK X5. See the [evaluation notes](./evaluator/README.md).

| Model | Size | Threads | Latency | FPS |
| --- | --- | ---: | --- | --- |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 142.9 ms | 7.0 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 2 | 149.5 ms | 13.3 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 3 | 167.4 ms | 17.8 FPS |

---

## License

Follows the Model Zoo top-level License.
