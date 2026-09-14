English | [简体中文](./README_cn.md)

# RepGhost Model Description

This directory provides the complete usage guide for the RepGhost sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

RepGhost is a lightweight CNN family designed to improve hardware efficiency by replacing explicit feature reuse in feature space with re-parameterized reuse in weight space. It avoids costly `Concat` operations while keeping strong classification performance.

- **Paper**: [RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
- **Reference Implementation**: [ChengpengChen/RepGhost](https://github.com/ChengpengChen/RepGhost)

### Algorithm Functionality

RepGhost supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Structural re-parameterization**: Converts training-time complex branches into inference-time efficient structures.
- **Implicit feature reuse**: Moves feature reuse from feature space to weight space to avoid costly `Concat`.
- **Hardware efficiency**: Reduces memory copy overhead and improves deployment efficiency on edge devices.
- **Variant scaling**: Provides multiple published variants from `100` to `200`.

![RepGhost Architecture](./test_data/RepGhost_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- RepGhost_100.yaml
|   |-- RepGhost_111.yaml
|   |-- RepGhost_130.yaml
|   |-- RepGhost_150.yaml
|   |-- RepGhost_200.yaml
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
|       |-- repghost.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- ibex.JPEG
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- RepGhost_architecture.png
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
- Conversion guidance is provided in [conversion/README.md](./conversion/README.md).

## Runtime Inference

The maintained inference path for this sample is Python.

- Python runtime guide: [runtime/python/README.md](./runtime/python/README.md)

## Evaluator

Evaluation notes, performance data, and validation summary are provided in [evaluator/README.md](./evaluator/README.md).

## Performance Data

The following table shows the published RepGhost performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepGhost_200 | 224x224 | 1000 | 9.79 | 76.43 | 75.25 | 2.89 | 451.42 |
| RepGhost_150 | 224x224 | 1000 | 6.57 | 74.75 | 73.50 | 2.20 | 626.60 |
| RepGhost_130 | 224x224 | 1000 | 5.48 | 75.00 | 73.57 | 1.87 | 743.56 |
| RepGhost_111 | 224x224 | 1000 | 4.54 | 72.75 | 71.25 | 1.71 | 881.19 |
| RepGhost_100 | 224x224 | 1000 | 4.07 | 72.50 | 72.25 | 1.55 | 964.69 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
