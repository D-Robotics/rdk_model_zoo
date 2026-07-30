# Model Conversion

This directory provides conversion-side notes for the YOLOWorld sample.

## Overview

The deployed model is `yolo_world.bin`, compiled for RDK X5. The original demo provides the compiled model and the offline vocabulary embedding JSON, but does not include conversion YAML files or export scripts in the repository.

## Model Protocol

### Inputs

| Input | Data Type | Shape | Layout |
| --- | --- | --- | --- |
| image | FP32 | `1 x 3 x 640 x 640` | NCHW |
| text | FP32 | `1 x 32 x 512 x 1` | NCHW-like embedding tensor |

### Outputs

| Output | Data Type | Shape |
| --- | --- | --- |
| classes_score | FP32 | `1 x 8400 x 32` |
| bboxes | FP32 | `1 x 8400 x 4` |

## Conversion Reference

If you need to regenerate the `.bin` model, use OpenExplorer Docker or the corresponding OE package compilation environment.

Offline Docker images can also be obtained from the D-Robotics developer forum: [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229).
