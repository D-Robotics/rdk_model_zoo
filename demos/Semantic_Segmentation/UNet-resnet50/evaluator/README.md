# UNet-resnet50 Model Evaluation

[English]| [简体中文](./README_cn.md)

## Overview

This directory is reserved for model evaluation tools and scripts for UNet-resnet50 semantic segmentation.

## Evaluation Metrics

Typical semantic segmentation evaluation metrics include:

- **Pixel Accuracy (PA)**: Ratio of correctly classified pixels
- **Mean Pixel Accuracy (mPA)**: Average per-class pixel accuracy
- **Mean Intersection over Union (mIoU)**: Primary metric for segmentation quality
- **Frequency Weighted IoU (FWIoU)**: IoU weighted by class frequency

## Usage

Evaluation scripts and detailed tutorials will be provided here. For now, users can evaluate the model using standard semantic segmentation evaluation libraries such as:

- `mmsegmentation`
- `torchmetrics`
- Custom evaluation scripts based on VOC or Cityscapes protocols

## Notes

- This module is optional.
- Please refer to the original UNet repository for training and evaluation best practices.
