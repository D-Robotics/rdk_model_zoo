English | [简体中文](./README_cn.md)

# YOLOWorld Model Description

This directory provides the complete usage guide for the YOLOWorld open-vocabulary detection sample in Model Zoo, including algorithm overview, model conversion notes, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

YOLOWorld extends YOLO-style object detection with open-vocabulary prompts. This sample uses offline vocabulary embeddings and a compiled RDK X5 `.bin` model to detect user-selected categories without running a text encoder at runtime.

### Algorithm Functionality

YOLOWorld supports the following task in this sample:

- Open-vocabulary object detection with offline text embeddings

### Algorithm Features

- Two-input model: image tensor and 32-slot text embedding tensor.
- Offline vocabulary embeddings stored in JSON format.
- Standard boxes, scores, and class IDs returned by the Python wrapper.

## Directory Structure

```text
.
|-- conversion
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download.sh
|   |-- README.md
|   `-- README_cn.md
|-- runtime
|   `-- python
|       |-- main.py
|       |-- yoloworld_det.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- dog.jpeg
|   |-- inference.png
|   `-- offline_vocabulary_embeddings.json
|-- README.md
`-- README_cn.md
```

## QuickStart

### Python

- Go to [runtime/python/README.md](./runtime/python/README.md) for detailed Python usage.
- For a quick experience:

```bash
cd runtime/python
bash run.sh
```

## Model Conversion

- Prebuilt `.bin` model files are provided through the [model](./model/README.md) directory.
- Conversion-side notes are provided in [conversion/README.md](./conversion/README.md).

## Runtime Inference

The maintained inference path for this sample is Python.

- Python runtime guide: [runtime/python/README.md](./runtime/python/README.md)

## Evaluator

Evaluation notes and validation summary are provided in [evaluator/README.md](./evaluator/README.md).

## Performance Data

This sample does not provide a published benchmark table. Use [evaluator/README.md](./evaluator/README.md) for validation notes.

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
