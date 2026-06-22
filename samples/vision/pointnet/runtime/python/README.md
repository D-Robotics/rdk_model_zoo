English | [简体中文](./README_cn.md)

# Python Runtime

This directory contains the Python runtime for PointNet chair part segmentation.

## Files

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- pointnet.py
`-- run.sh
```

## Dependencies

The runtime uses `numpy`, `matplotlib`, and `hbm_runtime`.

## Quick Start

```bash
bash run.sh
```

## Direct Run

```bash
python3 main.py \
  --model-path ../../model/s100/pointnet.hbm \
  --test-pts ../../test_data/chair.pts \
  --img-save-path result.png
```

## Arguments

| Argument | Description | Default |
| -------- | ----------- | ------- |
| `--model-path` | HBM model path | `../../model/s100/pointnet.hbm` |
| `--test-pts` | Input point cloud in `.pts` format | `../../test_data/chair.pts` |
| `--img-save-path` | Segmentation visualization output path | `result.png` |
| `--priority` | hbm_runtime scheduling priority | `0` |
| `--bpu-cores` | hbm_runtime BPU core indexes | `0` |

## Input and Output

`pointnet.py` loads `chair.pts`, centers and scales the coordinates, and transposes the input to `(1, 3, N)`.

The model output is per-point logits shaped `(1, N, 4)`. The runtime prints point counts for `back`, `seat`, `leg`, and `arm`, then saves the original and segmented point cloud views.
