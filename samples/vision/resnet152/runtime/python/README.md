English | [简体中文](./README_cn.md)

# ResNet152 Python Runtime

The Python runtime contains a thin command line entry and a reusable wrapper:

- `main.py`: argument parsing, image and label loading, configuration construction, `predict()` call, and result printing.
- `resnet152.py`: `Resnet152Config` and `Resnet152` wrapper based on `hbm_runtime`.
- `run.sh`: default runnable command that downloads the sample-local model before inference.

## Directory Structure

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet152.py
`-- run.sh
```

## Environment

Run this sample in the RDK S100 Python environment where `hbm_runtime`, `numpy`, and OpenCV are available. The script reuses shared helpers from `utils/py_utils`.

## Run

```bash
bash run.sh
```

## Direct Command

```bash
python3 main.py \
  --model-path ../../model/s100/resnet152_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path` | `../../model/s100/resnet152_224x224_nv12.hbm` | Path to the compiled HBM model. |
| `--test-img` | `../../test_data/zebra_cls.jpg` | Input image path. |
| `--label-file` | `../../../../../datasets/imagenet/imagenet_classes.names` | ImageNet label file. |
| `--top-k` | `5` | Number of classification results to print. |
| `--priority` | `0` | Runtime scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Wrapper Interface

`Resnet152` exposes:

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

The preprocessing stage converts a BGR image to NV12 and provides two fixed runtime inputs: Y plane and UV plane.
