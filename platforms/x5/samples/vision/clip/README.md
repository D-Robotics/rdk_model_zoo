English | [简体中文](./README_cn.md)

# CLIP Model Description

This directory provides the complete usage guide for the CLIP image-text matching sample in Model Zoo, including algorithm overview, model conversion notes, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

CLIP maps images and text into a shared feature space and compares them through cosine similarity. This sample keeps the deployed RDK X5 image encoder as a `.bin` model and the text encoder as an ONNX model, matching the original demo asset boundary.

### Algorithm Functionality

CLIP supports the following task in this sample:

- Image-text similarity matching

### Algorithm Features

- BPU image encoder executed through `hbm_runtime`.
- ONNX text encoder executed through `onnxruntime`.
- NumPy-based preprocessing and tokenization path without notebook dependency.

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
|       |-- bpe_simple_vocab_16e6.txt.gz
|       |-- clip_retrieval.py
|       |-- main.py
|       |-- README.md
|       |-- README_cn.md
|       |-- run.sh
|       `-- simple_tokenizer.py
|-- test_data
|   |-- dog.jpg
|   `-- inference.png
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

- Model files are provided through the [model](./model/README.md) directory.
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
