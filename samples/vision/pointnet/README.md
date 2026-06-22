English | [简体中文](./README_cn.md)

# PointNet Model Description

PointNet is a neural network for point cloud processing. It directly consumes unordered 3D points, applies shared MLP layers to extract point features, and aggregates global features with a symmetric max operation. This sample uses a PointNet chair part segmentation model to assign each input point to one of four parts: back, seat, leg, and arm.

The test input is the `chair.pts` point cloud. The runtime saves the original point cloud view and the predicted segmentation view.

## Algorithm Overview

PointNet is a deep learning model that directly processes unordered point sets for 3D classification and segmentation.

- **Paper**: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- **Reference Implementation**: [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

## Algorithm Capabilities

- Chair point cloud part segmentation
- Per-point part label prediction

## Algorithm Features

- **Unordered point input**: directly consumes `N x 3` point cloud coordinates.
- **Shared MLP**: extracts local features for each point.
- **Symmetric aggregation**: uses max pooling to produce order-invariant global features.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime
|   `-- python
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- pointnet.py
|       `-- run.sh
`-- test_data
    |-- chair.pts
    `-- readme_img
```

## Quick Start

```bash
cd samples/vision/pointnet/runtime/python
bash run.sh
```

The script checks `../../model/s100/pointnet.hbm`, runs segmentation on `../../test_data/chair.pts`, and saves `result_orig.png` and `result.png`.

## Model Conversion

- The HBM model file is provided in the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Model | Task | Input | Output |
| ----- | ---- | ----- | ------ |
| PointNet | Chair part segmentation | Normalized point cloud `(1, 3, N)` | Per-point logits `(1, N, 4)` |

## Model Evaluation

Evaluation notes, performance records, and result checks are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

For `chair.pts`, the output should include point counts for `back`, `seat`, `leg`, and `arm`, and save both original and segmented point cloud views.

## License

This sample is licensed under the [Apache 2.0 License](../../../LICENSE).
