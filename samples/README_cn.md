# 按任务选择 Sample

[English](README.md)

以下三个样例使用仓库根目录的统一实现。请先进入相应 README，按所用板卡准备模型；完整检出仓库即可使用，无需 Agent。

| 你要做什么 | 使用说明 | 转换自己的模型 | 验证输出 |
| --- | --- | --- | --- |
| 找出图片中的物体并取得框、类别、分数 | [YOLO 目标检测](vision/ultralytics_yolo/README_cn.md) | [导出与编译](vision/ultralytics_yolo/conversion/README_cn.md) | [检测评估](vision/ultralytics_yolo/evaluator/README_cn.md) |
| 给图片分类，取得 Top-K 类别和分数 | [ResNet 图像分类](vision/resnet/README_cn.md) | [转换流程](vision/resnet/conversion/README_cn.md) | [分类验证](vision/resnet/evaluator/README_cn.md) |
| 找出文字区域并识别文字 | [PaddleOCR](vision/paddle_ocr/README_cn.md) | [检测与识别模型转换](vision/paddle_ocr/conversion/README_cn.md) | [OCR 验证](vision/paddle_ocr/evaluator/README_cn.md) |

## 板卡与制品

| Sample | 代表制品 | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| YOLO | YOLOv8n / YOLO26n 检测 | 8GB / 4GB | 已对照 | 已对照 | 9/16 历史结果；本轮未测 |
| ResNet | ResNet18 | 8GB / 4GB | 已对照 | 无已核定制品 | 9/16 历史结果；本轮未测 |
| PaddleOCR | X5 PP-OCRv3 / S100 PP-OCRv6 | 8GB / 4GB | 已对照 | 无已核定配套模型 | 无已核定配套模型 |

这些是已记录的固定输入运行结果，完整精度和性能要使用对应数据集单独评估。不同板卡的制品不可互换，OCR 不同代际也不可混配检测、识别和词典。

## 代码放在哪里

每个 Sample 内部维护任务流程、模型绑定和调用、转换及评估；跨样例一致的平台识别、制品读取与图像字节转换位于 [`_shared/`](_shared/README.md)。查看对应 README 的“源码阅读”即可定位修改点。

其他模型仍从 [X5](../platforms/x5/README_cn.md) 或 [S 系列](../platforms/s/README_cn.md) 平台目录使用。本轮不扩展分割、姿态等任务；[X3](../platforms/x3/README_cn.md) 保留为历史内容。
