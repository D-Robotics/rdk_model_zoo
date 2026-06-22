English | [简体中文](./README_cn.md)

# Python Runtime

This directory contains the Python runtime for 3D ResNet-18 video action classification.

## Files

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet3d.py
`-- run.sh
```

## Quick Start

```bash
bash run.sh
```

## Direct Run

```bash
python3 main.py \
  --model-path ../../model/s100/r3d_18.hbm \
  --test-clip ../../test_data/video0.npy \
  --label-file ../../test_data/kinetics_classnames.json \
  --top-k 5
```

## Arguments

| Argument | Description | Default |
| -------- | ----------- | ------- |
| `--model-path` | HBM model path | `../../model/s100/r3d_18.hbm` |
| `--test-clip` | Preprocessed video clip in `.npy` format | `../../test_data/video0.npy` |
| `--label-file` | Kinetics-400 class mapping JSON | `../../test_data/kinetics_classnames.json` |
| `--top-k` | Number of predictions to print | `5` |
| `--priority` | hbm_runtime scheduling priority | `0` |
| `--bpu-cores` | hbm_runtime BPU core indexes | `0` |

## Input and Output

The model input is a float32 tensor with shape `(1, 3, 16, 112, 112)`.

The model output is a Kinetics-400 logits tensor. `resnet3d.py` converts the logits to Top-K probabilities and `main.py` prints class names with scores.
