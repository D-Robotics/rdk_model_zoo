# 测试数据来源

本试点只复制 X5 和 S100 Python 对照所需的默认 fixture，文件内容与已有平台
样例保持一致：

| 文件 | 原始来源 | SHA-256 | 许可说明 |
| --- | --- | --- | --- |
| `x5/paddleocr_test.jpg` | [`platforms/x5/samples/vision/paddleocr/test_data/paddleocr_test.jpg`](../../../../platforms/x5/samples/vision/paddleocr/test_data/paddleocr_test.jpg) | `5b4a7fb523c7c459c8d3cec67480c1872cd7b3674b34505467420561ad8c577e` | 见 [X5 Apache-2.0 license](../../../../platforms/x5/LICENSE) |
| `s100/gt_2322.jpg` | [`platforms/s/samples/vision/paddle_ocr/test_data/gt_2322.jpg`](../../../../platforms/s/samples/vision/paddle_ocr/test_data/gt_2322.jpg) | `18a214e1c637fb3a53f71673c6f6a689b5f16d755237ab7e9e58ddc32223580b` | 见 [S Apache-2.0 license](../../../../platforms/s/LICENSE) |
| `s100/ppocrv6_dict.txt` | [`platforms/s/samples/vision/paddle_ocr/test_data/ppocrv6_dict.txt`](../../../../platforms/s/samples/vision/paddle_ocr/test_data/ppocrv6_dict.txt) | `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6` | 见 [S Apache-2.0 license](../../../../platforms/s/LICENSE) |

图像作为 OpenCV BGR 输入；S100 字典按 UTF-8 每行一个 token 读取，再按原版
规则在首位加入 blank、末尾加入空格。没有复制模型、字体或生成结果图。
