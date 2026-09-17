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

Other models remain under the [X5](../platforms/x5/README.md) and
[S-series](../platforms/s/README.md) directories. This batch does not expand
segmentation or pose tasks. [X3](../platforms/x3/README.md) is retained as historical content.

S600 comparisons above are the September 16 snapshot. The September 17 revision could not be retested there because SSH connectivity has not recovered.
