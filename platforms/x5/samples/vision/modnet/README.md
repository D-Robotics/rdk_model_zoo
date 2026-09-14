English | [简体中文](./README_cn.md)

# MODNet Model Description

This directory provides the complete usage guide for the MODNet sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

---

## Algorithm Overview

MODNet (Mobile-friendly One-stage Deep Image Matting Network) is a lightweight deep learning model for portrait matting. It takes a single RGB image and directly predicts an alpha matte without requiring a trimap as auxiliary input.

- **Paper**: [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)
- **Official Implementation**: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

### Algorithm Functionality

MODNet can complete the following tasks:

- Portrait matting from a single RGB image
- Alpha matte prediction (no trimap required)

### Algorithm Features

- **One-stage End-to-End**: Directly predicts the alpha matte from an RGB image, eliminating the need for a trimap
- **Mobile-friendly**: Designed for real-time inference on resource-constrained devices
- **Semantic-Spatial-Detail Fusion**: Uses a tri-module architecture (Semantic Estimation, Detail Refinement, Detail Attention) for high-quality matting

---

## Directory Structure

This directory contains:

```bash
.
├── conversion                          # Model conversion process
│   ├── onnx_export                     # ONNX export scripts
│   ├── ptq_yamls                       # PTQ configuration YAML files
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── evaluator                           # Model evaluation
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── model                               # Model files and download scripts
│   ├── download_model.sh               # BIN model download script
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── runtime                             # Inference samples
│   └── python                          # Python inference sample
│       ├── main.py                     # Python entry script
│       ├── modnet.py                   # MODNet wrapper implementation
│       ├── run.sh                      # One-click execution script
│       ├── README.md                   # Documentation (English)
│       └── README_cn.md                # Documentation (Chinese)
├── test_data                           # Inference results and sample data
│   ├── person.jpg                      # Sample input image
│   └── bg.jpg                          # Sample background image
└── README.md                           # MODNet overview and quickstart
```

---

## QuickStart

For a quick experience, each model provides a `run.sh` script that allows you to run the corresponding model with one click.

### Python

- Go to the `python` directory under `runtime` and run the `run.sh` script:
    ```bash
    cd runtime/python/
    chmod +x run.sh
    ./run.sh
    ```
- For detailed usage of the `python` code, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Model Conversion

- ModelZoo provides pre-adapted BIN model files. Users can directly run the `download_model.sh` script in the `model` directory to download and use them. If you are not concerned about the model conversion process, **you can skip this section**.

- If you need to customize model conversion parameters or understand the complete conversion process, please refer to [conversion/README.md](./conversion/README.md).

---

## Runtime Inference

MODNet model inference sample provides Python implementation.

### Python Version

- Provided in script form, suitable for rapid verification of model effects and algorithm flows
- The sample demonstrates the complete process of model loading, inference execution, post-processing, and result compositing
- For detailed usage, parameter descriptions, and interface specifications, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Evaluator

`evaluator/` is used for model accuracy, performance, and numerical consistency evaluation. Please refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Performance Data

The following table shows the actual test performance data of the MODNet model on the RDK X5 platform.

| Model | Size | Input Format | Latency (ms) | FPS |
| --- | --- | --- | --- | --- |
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet (2 threads) | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

**Notes:**
1. Test platform: RDK X5, CPU 8xA55@1.8G, BPU 1xBayes-e@1G (10TOPS INT8)
2. Single-thread latency is measured under single-frame, single-thread, single-BPU-core conditions
3. Multi-thread FPS is measured with 2 threads for concurrent inference

---

## License

Follows the Model Zoo top-level License.
