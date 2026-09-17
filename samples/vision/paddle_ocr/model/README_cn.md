# 模型准备

本试点不把编译模型复制到此目录。使用原生入口的显式 `--prepare`，同时给出
两个限定资产引用和本地 `--model-dir`：

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

准备动作是显式的，并通过已有 manifest 读取器处理资产；普通推理不会下载。
四个已审计模型行没有 publisher SHA-256，因此输出的只是本地观测摘要。详见
[`试点说明`](../README_cn.md)、[X5 原版模型说明](../../../../platforms/x5/samples/vision/paddleocr/model/README.md)
和 [S 原版模型说明](../../../../platforms/s/samples/vision/paddle_ocr/model/README.md)。
