# Choose a sample by task

[简体中文](README_cn.md)

The following samples use the canonical implementations in this repository.
Open a sample guide and prepare artifacts for your board. Keep the full checkout;
an Agent is not required.

| Goal | Usage | Convert your model | Check results |
| --- | --- | --- | --- |
| Locate objects and obtain boxes, classes and scores | [YOLO detection](vision/ultralytics_yolo/README.md) | [Export and compile](vision/ultralytics_yolo/conversion/README.md) | [Detection evaluation](vision/ultralytics_yolo/evaluator/README.md) |
| Obtain Top-K image classes and scores | [ResNet classification](vision/resnet/README.md) | [Conversion](vision/resnet/conversion/README.md) | [Classification validation](vision/resnet/evaluator/README.md) |
| Locate text regions and recognize text | [PaddleOCR](vision/paddle_ocr/README.md) | [Detector and recognizer conversion](vision/paddle_ocr/conversion/README.md) | [OCR validation](vision/paddle_ocr/evaluator/README.md) |

## Boards and artifacts

| Sample | Artifacts checked in this batch | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| YOLO | YOLOv8n / YOLO26n detection | 8GB / 4GB | Compared | Compared | Compared |
| ResNet | ResNet18 | 8GB / 4GB | Compared | No audited artifact | Compared |
| PaddleOCR | X5 PP-OCRv3 / S100 PP-OCRv6 | 8GB / 4GB | Compared | No audited pair | No audited pair |

These are recorded fixed-input checks. Full accuracy and performance require
separate dataset evaluation. Artifacts are not interchangeable between boards;
do not mix OCR generations, components or vocabularies.

## Where code lives

Each sample owns its task flow, model binding/call, conversion and evaluation.
Common target identity, artifact access and image-byte conversion live in
[`_shared/`](_shared/README.md). The sample guides identify the code to read and edit.

Unmigrated models remain under the [X5](../platforms/x5/README.md) and
[S-series](../platforms/s/README.md) directories. [X3](../platforms/x3/README.md) is retained as historical content.

S600 comparisons above are the September 16 snapshot. The September 17 revision could not be retested there because SSH connectivity has not recovered.

## Migrated classification samples (2026-09-22)

These are unified source entrypoints. Read each sample and migration ledger for target-specific validation; source availability is not board acceptance. All eight B4 classification samples have host validation only; board checks are not-run.

| Sample | Guide |
| --- | --- |
| mobilenetv1 | [mobilenetv1](vision/mobilenetv1/README.md) |
| mobilenetv2 | [mobilenetv2](vision/mobilenetv2/README.md) |
| mobilenetv3 | [mobilenetv3](vision/mobilenetv3/README.md) |
| mobilenetv4 | [mobilenetv4](vision/mobilenetv4/README.md) |
| efficientnet | [efficientnet](vision/efficientnet/README.md) |
| efficientformer | [efficientformer](vision/efficientformer/README.md) |
| efficientformerv2 | [efficientformerv2](vision/efficientformerv2/README.md) |
| efficientvit | [efficientvit](vision/efficientvit/README.md) |
| convnext | [convnext](vision/convnext/README.md) |
| edgenext | [edgenext](vision/edgenext/README.md) |
| fasternet | [fasternet](vision/fasternet/README.md) |
| fastvit | [fastvit](vision/fastvit/README.md) |
| repghost | [repghost](vision/repghost/README.md) |
| repvgg | [repvgg](vision/repvgg/README.md) |
| repvit | [repvit](vision/repvit/README.md) |
| mobileone | [mobileone](vision/mobileone/README.md) |
| resnext | [resnext](vision/resnext/README.md) |
| vargconvnet | [vargconvnet](vision/vargconvnet/README.md) |
| googlenet | [googlenet](vision/googlenet/README.md) |
| hgnetv2 | [hgnetv2](vision/hgnetv2/README.md) |
| vit | [ViT CIFAR-10](vision/vit/README.md) |

## Unified feature extraction samples

These entries expose vision features. Current validation is host-only; board verification remains not-run.

| Sample | Guide |
| --- | --- |
| SigLIP | [SigLIP](vision/siglip/README.md) |
| DINOv2 | [DINOv2](vision/dinov2/README.md) |

## Image-text matching and video classification

These unified entries have host tests; real board verification remains not-run.

| Task | Guide |
| --- | --- |
| Match one image to candidate text descriptions | [CLIP](vision/clip/README.md) |
| Classify a prepared 16-frame video tensor | [3DResNet](vision/3dresnet/README.md) |

## Prompted segmentation

These two-stage samples have local host validation. Real board verification and conversion/calibration remain not-run. Their prompt interfaces differ; read the selected guide first.

| Goal | Guide | Conversion | Validation |
| --- | --- | --- | --- |
| Produce a mask using two positive points fixed at export | [EfficientSAM](vision/efficient_sam/README.md) | [Conversion](vision/efficient_sam/conversion/README.md) | [Comparison and performance procedures](vision/efficient_sam/evaluator/README.md) |
| Produce a mask using one runtime box prompt | [MobileSAM](vision/mobile_sam/README.md) | [Conversion](vision/mobile_sam/conversion/README.md) | [Comparison and performance procedures](vision/mobile_sam/evaluator/README.md) |
