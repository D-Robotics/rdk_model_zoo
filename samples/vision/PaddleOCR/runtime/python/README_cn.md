[English](./README.md) | [简体中文](./README_cn.md)

# PaddleOCR Python 推理示例

本示例演示如何在 RDK 平台（BPU 核心）上使用 Python 加速 PaddleOCR 的推理过程。


## 目录结构
```text
.
├── main.py         # 主推理程序
├── run.sh          # 一键运行脚本
├── README.md       # 使用说明 (英文)
└── README_cn.md    # 使用说明 (中文)
```

## 参数说明

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `--det_model_path` | 文字检测 .bin 模型路径 | ../../model/en_PP-OCRv3_det_640x640_nv12.bin |
| `--rec_model_path` | 文字识别 .bin 模型路径 | ../../model/en_PP-OCRv3_rec_48x320_rgb.bin |
| `--image_path` | 测试输入图像路径 | ../../test_data/paddleocr_test.jpg |
| `--output_folder` | 结果保存路径 | ../../test_data/output/predict.jpg |

## 快速运行
```bash
# 使用一键运行脚本
bash run.sh

# 手动运行
python3 main.py --det_model_path ../../model/en_PP-OCRv3_det_640x640_nv12.bin \
                --rec_model_path ../../model/en_PP-OCRv3_rec_48x320_rgb.bin \
                --image_path ../../test_data/paddleocr_test.jpg
```

---
## 接口说明
本示例展示了如何使用 `hbm_runtime` 加载模型、输入数据预处理（包括 BGR 转 NV12）、BPU 推理执行以及 OCR 后处理（包含 DB 检测后处理和 CTC 解码）。
详细 API 文档见 [docs/Python_API_User_Guide.md](../../../../docs/Python_API_User_Guide.md)。
