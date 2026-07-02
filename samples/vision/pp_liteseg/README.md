English | [简体中文](./README_cn.md)

# PP-LiteSeg-STDC1 Semantic Segmentation

PP-LiteSeg-STDC1 is a lightweight real-time semantic segmentation model from PaddleSeg, quantized and deployed on RDK X5 BPU.

---

## Algorithm Overview

PP-LiteSeg is a lightweight real-time semantic segmentation model from PaddleSeg. The recommended starting point for RDK X5 conversion is `PP-LiteSeg-STDC1` with Cityscapes-style input, because it keeps the model compact while still producing visually meaningful segmentation results.

- Paper: [PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)
- Official Implementation: [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

### Algorithm Functionality

- Semantic segmentation from a single RGB image
- Per-pixel class prediction for road-scene or custom segmentation datasets

### Algorithm Features

- Lightweight segmentation network suitable for edge deployment
- CNN-based structure that is friendlier to PTQ than prompt-based segmentation or large Transformer models
- Simple output protocol: one segmentation logits tensor followed by `argmax` post-processing

---

## Directory Structure

```bash
.
├── conversion
│   ├── onnx_export
│   │   └── export_pp_liteseg_stdc1_onnx.sh
│   ├── ptq_yamls
│   │   └── pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml
│   ├── prepare_calibration.py
│   ├── README.md
│   └── README_cn.md
├── evaluator
│   ├── README.md
│   └── README_cn.md
├── model
│   ├── download.sh
│   ├── README.md
│   └── README_cn.md
├── runtime
│   └── python
│       ├── main.py
│       ├── pp_liteseg.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data
│   └── street.jpg
├── README.md
└── README_cn.md
```

> Note: This sample currently provides only the Python runtime implementation.

---

## Quick Start

Run the following commands on the RDK X5 board:

```bash
# 1. Enter the runtime directory
cd samples/vision/pp_liteseg/runtime/python

# 2. One-click run (downloads model automatically if absent)
chmod +x run.sh
./run.sh

# 3. Or run directly with custom arguments
python3 main.py \
    --model-path ../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
    --test-img ../../test_data/street.jpg \
    --output ../../test_data/result.jpg
```

See [runtime/python/README.md](./runtime/python/README.md) for full parameter reference.

---

## Recommended Flow

This sample covers the complete path from model conversion to on-board inference:

```text
PaddleSeg pretrained model -> exported inference model -> ONNX -> hb_mapper checker -> calibration data -> hb_mapper makertbin -> .bin -> BPU inference
```

- **Model conversion**: [conversion/README.md](./conversion/README.md)
- **Download prebuilt model**: [model/download.sh](./model/download.sh)
- **On-board inference**: [runtime/python/README.md](./runtime/python/README.md)
- **Accuracy & performance**: [evaluator/README.md](./evaluator/README.md)

---

## Runtime Protocol

The generated deployment model is expected to follow this protocol:

- Model: `PP-LiteSeg-STDC1`
- Input resolution: `1024x512` by default
- Runtime input type: `nv12`
- Training input type: `rgb`
- Training layout: `NCHW`
- Normalization: ImageNet-style mean/std through `hb_mapper` YAML
- Output: segmentation logits tensor, decoded by `argmax` along the class dimension

---

## Notes

- Keep the ONNX model with static input shape before running `hb_mapper`.
- Do not include Python post-processing, resizing, or palette rendering in the ONNX graph.
- Use representative road-scene images for calibration. Start with 20 to 50 images.
