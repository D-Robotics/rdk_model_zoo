English | [简体中文](./README_cn.md)

# ViT Python Runtime

The Python runtime contains a command line entry and a reusable wrapper:

- `main.py`: argument parsing, image and label loading, configuration construction, `predict()` call, and result printing.
- `vit.py`: `ViTConfig` and `ViT` wrapper based on `hbm_runtime`.
- `run.sh`: default runnable command that downloads the sample-local model before inference.

## Directory Structure

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- main.py
|-- run.sh
`-- vit.py
```

## Environment

Run this sample in the RDK S100 Python environment where `hbm_runtime`, `numpy`, and OpenCV are available. The script reuses shared helpers from `utils/py_utils`.

## Run

```bash
bash run.sh int8
```

Use the int16 model:

```bash
bash run.sh int16
```

## Direct Command

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model-variant` | `int8` | Model variant, either `int8` or `int16`. |
| `--model-path` | `../../model/s100/vit_cifar10_batch1_int8.hbm` | Path to the compiled HBM model. |
| `--test-img` | `../../test_data/airplane_0000.png` | Input image path. |
| `--label-file` | `../../test_data/cifar10_classes.names` | CIFAR-10 label file. |
| `--top-k` | `5` | Number of classification results to print. |
| `--priority` | `0` | Runtime scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Wrapper Interface

`ViT` exposes:

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

The preprocessing stage converts a BGR image to NV12 and provides two fixed runtime inputs: Y plane and UV plane.
