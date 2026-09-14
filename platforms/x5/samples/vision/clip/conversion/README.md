# Model Conversion

This directory provides conversion-side notes for the CLIP sample.

## Overview

The CLIP sample uses two deployed assets:

- `img_encoder.bin`: RDK X5 BPU image encoder.
- `text_encoder.onnx`: ONNX text encoder executed on CPU through `onnxruntime`.

This sample does not provide conversion YAML files or export scripts in the repository. The conversion notes therefore document the model protocol and use the published model files from the Model Zoo archive.

## Model Protocol

### Image Encoder

| Input | Data Type | Shape | Layout |
| --- | --- | --- | --- |
| image | FP32 | `1 x 3 x 224 x 224` | NCHW |

| Output | Data Type | Shape |
| --- | --- | --- |
| image_feature | FP32 | `1 x 512` |

### Text Encoder

| Input | Data Type | Shape |
| --- | --- | --- |
| texts | INT32 | `num_text x 77` |

| Output | Data Type | Shape |
| --- | --- | --- |
| text_features | FP32 | `num_text x 512` |

## Conversion Reference

If you need to regenerate the image encoder `.bin`, use OpenExplorer Docker or the corresponding OE package compilation environment. The text encoder remains an ONNX runtime asset in this sample.

Offline Docker images can also be obtained from the D-Robotics developer forum: [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229).
