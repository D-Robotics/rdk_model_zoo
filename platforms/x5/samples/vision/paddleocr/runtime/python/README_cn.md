[English](./README.md) | [简体中文](./README_cn.md)

# PaddleOCR Python 推理示例

本示例演示如何在 RDK X5 平台（BPU 核心）上使用 Python 加速 PaddleOCR 的推理过程。

## 目录结构
```text
.
├── main.py         # 推理入口脚本
├── paddleocr.py    # PaddleOCR 模型封装
├── run.sh          # 一键运行脚本
├── README.md       # 使用说明 (英文)
└── README_cn.md    # 使用说明 (中文)
```

## 参数说明

| 参数                | 描述                           | 默认值                                           |
|:-------------------|:-------------------------------|:-------------------------------------------------|
| `--det-model-path` | 文字检测 .bin 模型路径          | ../../model/en_PP-OCRv3_det_640x640_nv12.bin    |
| `--rec-model-path` | 文字识别 .bin 模型路径          | ../../model/en_PP-OCRv3_rec_48x320_rgb.bin       |
| `--test-img`       | 测试输入图像路径                 | ../../test_data/paddleocr_test.jpg               |
| `--det-threshold`  | 检测二值化阈值                   | 0.5                                              |
| `--img-save-path`  | 结果图片保存路径                 | ../../test_data/result.jpg                       |
| `--priority`       | 模型调度优先级 (0~255)          | 0                                                |
| `--bpu-cores`      | BPU 核心索引列表                 | [0]                                              |

## 快速运行

- **一键运行脚本**
    ```bash
    bash run.sh
    ```

- **手动运行**
    - 使用默认参数
        ```bash
        python3 main.py
        ```
    - 指定参数运行
        ```bash
        python3 main.py \
            --det-model-path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
            --rec-model-path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
            --test-img ../../test_data/paddleocr_test.jpg
        ```

## 接口说明

- **PaddleOCRConfig**: 封装模型路径及推理参数。
- **PaddleOCR**: 包含完整的两阶段 OCR 流水线（检测 + 识别），提供 `pre_process`、`forward`、`post_process`、`predict`。

阅读 [源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。
