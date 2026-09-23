# 按任务选择 Sample

[English](README.md)

以下样例使用仓库根目录的统一实现。请先进入相应 README，按所用板卡准备模型；完整检出仓库即可使用，无需 Agent。

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

尚未迁移的模型仍从 [X5](../platforms/x5/README_cn.md) 或 [S 系列](../platforms/s/README_cn.md) 平台目录使用。[X3](../platforms/x3/README_cn.md) 保留为历史内容。

## 已迁移分类样例（2026-09-22）

以下是可用统一源码入口；板测状态按各 sample README 与迁移台账读取，不代表所有目标均已验收。B4 八个新增分类样例目前只有主机侧验证，板测 not-run。

| Sample | Guide |
| --- | --- |
| mobilenetv1 | [mobilenetv1](vision/mobilenetv1/README_cn.md) |
| mobilenetv2 | [mobilenetv2](vision/mobilenetv2/README_cn.md) |
| mobilenetv3 | [mobilenetv3](vision/mobilenetv3/README_cn.md) |
| mobilenetv4 | [mobilenetv4](vision/mobilenetv4/README_cn.md) |
| efficientnet | [efficientnet](vision/efficientnet/README_cn.md) |
| efficientformer | [efficientformer](vision/efficientformer/README_cn.md) |
| efficientformerv2 | [efficientformerv2](vision/efficientformerv2/README_cn.md) |
| efficientvit | [efficientvit](vision/efficientvit/README_cn.md) |
| convnext | [convnext](vision/convnext/README_cn.md) |
| edgenext | [edgenext](vision/edgenext/README_cn.md) |
| fasternet | [fasternet](vision/fasternet/README_cn.md) |
| fastvit | [fastvit](vision/fastvit/README_cn.md) |
| repghost | [repghost](vision/repghost/README_cn.md) |
| repvgg | [repvgg](vision/repvgg/README_cn.md) |
| repvit | [repvit](vision/repvit/README_cn.md) |
| mobileone | [mobileone](vision/mobileone/README_cn.md) |
| resnext | [resnext](vision/resnext/README_cn.md) |
| vargconvnet | [vargconvnet](vision/vargconvnet/README_cn.md) |
| googlenet | [googlenet](vision/googlenet/README_cn.md) |
| hgnetv2 | [hgnetv2](vision/hgnetv2/README_cn.md) |
| vit | [ViT CIFAR-10](vision/vit/README_cn.md) |

## 已迁移特征提取样例

以下统一入口输出视觉特征，目前仅通过主机验证，板端复验仍为 not-run。

| Sample | Guide |
| --- | --- |
| SigLIP | [SigLIP](vision/siglip/README_cn.md) |
| DINOv2 | [DINOv2](vision/dinov2/README_cn.md) |

## 图文匹配与视频分类

以下统一入口已完成主机测试，真实板端复验仍为 not-run。

| 你要做什么 | 使用说明 |
| --- | --- |
| 对比图片与多条文字描述的相似度 | [CLIP](vision/clip/README_cn.md) |
| 对预处理的16帧视频张量做动作分类 | [3DResNet](vision/3dresnet/README_cn.md) |

## 提示分割

以下双阶段样例已完成本地主机验证；真实板端验证和转换校准仍为 not-run。两者的提示接口不同，请先阅读相应说明。

| 你要做什么 | 使用说明 | 转换 | 验证 |
| --- | --- | --- | --- |
| 使用导出时固定的双正点提示生成掩码 | [EfficientSAM](vision/efficient_sam/README_cn.md) | [转换](vision/efficient_sam/conversion/README_cn.md) | [对照与性能流程](vision/efficient_sam/evaluator/README_cn.md) |
| 使用运行时单框提示生成掩码 | [MobileSAM](vision/mobile_sam/README_cn.md) | [转换](vision/mobile_sam/conversion/README_cn.md) | [对照与性能流程](vision/mobile_sam/evaluator/README_cn.md) |
