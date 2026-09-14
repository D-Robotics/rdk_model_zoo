[English](./README.md) | [简体中文](./README_cn.md)

# PaddleOCR Python Inference Example

This example demonstrates how to use Python to accelerate the inference of PaddleOCR on the RDK X5 platform (BPU cores).

## Directory Structure
```text
.
├── main.py         # Inference entry script
├── paddleocr.py    # PaddleOCR model wrapper
├── run.sh          # One-click execution script
├── README.md       # Usage instructions (English)
└── README_cn.md    # Usage instructions (Chinese)
```

## Parameter Description

| Parameter           | Description                              | Default                                          |
|:--------------------|:-----------------------------------------|:-------------------------------------------------|
| `--det-model-path`  | Path to detection .bin model             | ../../model/en_PP-OCRv3_det_640x640_nv12.bin    |
| `--rec-model-path`  | Path to recognition .bin model           | ../../model/en_PP-OCRv3_rec_48x320_rgb.bin       |
| `--test-img`        | Path to test input image                 | ../../test_data/paddleocr_test.jpg               |
| `--det-threshold`   | Binarization threshold for detection     | 0.5                                              |
| `--img-save-path`   | Path to save the result image            | ../../test_data/result.jpg                       |
| `--priority`        | Model priority (0~255)                   | 0                                                |
| `--bpu-cores`       | BPU core indexes to run inference        | [0]                                              |

## Quick Run

- **One-click Execution Script**
    ```bash
    bash run.sh
    ```

- **Manual Execution**
    - Use default parameters
        ```bash
        python3 main.py
        ```
    - Run with specified parameters
        ```bash
        python3 main.py \
            --det-model-path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
            --rec-model-path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
            --test-img ../../test_data/paddleocr_test.jpg
        ```

## Interface Description

- **PaddleOCRConfig**: Encapsulates model paths and inference parameters.
- **PaddleOCR**: Contains the complete two-stage OCR pipeline (detection + recognition) with `pre_process`, `forward`, `post_process`, `predict`.

Refer to the [Source Reference Documentation](../../../../../docs/source_reference/README.md) for more details.
