[English](./README.md) | [简体中文](./README_cn.md)

# PaddleOCR Python Inference Example

This example demonstrates how to use Python to accelerate the inference of PaddleOCR on the RDK platform (BPU cores).

## Structure
```text
.
├── main.py         # Main inference program
├── run.sh          # One-click execution script
├── README.md       # Usage instructions (English)
└── README_cn.md    # Usage instructions (Chinese)
```

## Usage

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--det_model_path` | Path to detection .bin model | ../../model/en_PP-OCRv3_det_640x640_nv12.bin |
| `--rec_model_path` | Path to recognition .bin model | ../../model/en_PP-OCRv3_rec_48x320_rgb.bin |
| `--image_path` | Path to test input image | ../../test_data/paddleocr_test.jpg |
| `--output_folder` | Path to save the result | ../../test_data/output/predict.jpg |

## Quick Start
```bash
# Using one-click script
bash run.sh

# Manual run
python3 main.py --det_model_path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
                --rec_model_path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
                --image_path ../../test_data/paddleocr_test.jpg
```

---
## API Description
This example shows how to use `hbm_runtime` for model loading, preprocessing (including BGR to NV12), BPU execution, and OCR post-processing (including DB detection post-processing and CTC decoding).
Refer to [docs/Python_API_User_Guide.md](../../../../docs/Python_API_User_Guide.md) for details.
