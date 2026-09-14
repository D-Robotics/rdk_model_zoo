English | [简体中文](./README_cn.md)

# Model Files

This directory provides the model download script for the 3D ResNet-18 video action classification sample.

## Download

```bash
bash download_model.sh s100
```

The model is downloaded to:

```text
model/s100/r3d_18.hbm
```

## Model Artifact

| File | Description |
| ---- | ----------- |
| `s100/r3d_18.hbm` | R3D-18 HBM model for a `(1, 3, 16, 112, 112)` float32 video clip input and Kinetics-400 logits output. |
