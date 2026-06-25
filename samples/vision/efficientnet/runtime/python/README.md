English | [简体中文](./README_cn.md)

# EfficientNet-Lite Python Runtime

The Python runtime contains a command line entry and a reusable wrapper:

- `main.py`: argument parsing, image and label loading, configuration construction, `predict()` call, and result printing.
- `efficientnet.py`: `EfficientNetConfig` and `EfficientNet` wrapper based on `hbm_runtime`.
- `run.sh`: default runnable command that auto-detects the target SoC, downloads the matching model, and runs inference.

## Directory Structure

```text
runtime/python/
|-- README.md
|-- README_cn.md
|-- efficientnet.py
|-- main.py
`-- run.sh
```

## Environment

Run this sample in the RDK Python environment where `hbm_runtime`, `numpy`, and OpenCV are available. The script reuses shared helpers from `utils/py_utils`.

## Run

```bash
bash run.sh
```

## Direct Command

```bash
python3 main.py \
  --model-path ../../model/s100/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

For S600:

```bash
python3 main.py \
  --model-path ../../model/s600/efficientnet_lite0_224x224_nv12.hbm \
  --test-img ../../test_data/Scottish_deerhound.JPEG \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path` | Auto-detected per SoC (defaults to Lite0) | Path to the compiled HBM model. |
| `--test-img` | `../../test_data/Scottish_deerhound.JPEG` | Input image path. |
| `--label-file` | `../../test_data/imagenet_classes.names` | ImageNet label file. |
| `--top-k` | `5` | Number of classification results to print. |
| `--resize-type` | `1` | Resize mode: `0` stretch, `1` keep aspect ratio with padding. |
| `--priority` | `0` | Runtime scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Wrapper Interface

`EfficientNet` exposes:

- `set_scheduling_params(...)`
- `pre_process(...)`
- `forward(...)`
- `post_process(...)`
- `predict(...)`
- `__call__(...)`

The preprocessing stage converts a BGR image to NV12 and provides two fixed runtime inputs: Y plane and UV plane.