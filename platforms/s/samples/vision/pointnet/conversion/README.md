English | [简体中文](./README_cn.md)

# PointNet Model Conversion Guide

This directory provides conversion information, network structure notes, ONNX operator notes, and quantization accuracy records for the PointNet chair part segmentation model.

## Source Model

PointNet implementation reference:

```text
https://gitee.com/chenguanzhong/rdk_-s100_-point-net_-official
```

PointNet directly consumes unordered point cloud coordinates. A point cloud with `N` points can be represented as an `N x 3` array, where each point contains 3D coordinates `(x, y, z)`. Some point cloud datasets also include normal vectors `(nx, ny, nz)`. Since point order does not change the point cloud semantics, the network uses order-invariant symmetric operations such as max or sum.

The deployed chair part segmentation model takes normalized points and outputs per-point predictions for four chair parts: `back`, `seat`, `leg`, and `arm`.

## Network Structure

PointNet applies shared MLP layers to extract point features, then aggregates global features with a max operation. For segmentation, it combines local point features with global features and outputs a part label for each point.

![PointNet overview](../test_data/readme_img/image-1.png)
![PointNet segmentation](../test_data/readme_img/image.png)

## ONNX Notes

The PointNet ONNX graph mainly contains common operators such as `Conv`, `BatchNorm`, and `ReLU`, which are supported on RDK S100.

![PointNet ONNX graph](../test_data/readme_img/char_static.png)

## Quantization Notes

This model uses int16 quantization. The quantization accuracy record reports `trans` accuracy greater than 0.9999 and `pred` accuracy greater than 0.98.

![Quantization accuracy](../test_data/readme_img/pixpin_2025-07-07_20-44-37.jpg)

## Artifact Used by Runtime

The runtime sample loads the HBM model from:

```text
samples/vision/pointnet/model/s100/pointnet.hbm
```

The HBM model file is provided in the [model](../model/README.md) directory.

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
