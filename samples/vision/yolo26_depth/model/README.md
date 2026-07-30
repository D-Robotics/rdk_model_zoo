[English](./README.md) | [简体中文](./README_cn.md)

# Model Files

The quantized RDK X5 models are downloaded into `model/bayes-e/`. Model binaries are not stored in the repository.

## Model List

| Variant | File | SHA256 |
| --- | --- | --- |
| N | `yolo26n_depth_bayese_768x768_nv12.bin` | `e55091eb594e20e37e6c36a36cce42a94ad80ec651ae893a2143cd2273ed9b0b` |
| S | `yolo26s_depth_bayese_768x768_nv12.bin` | `0e43958195f504d7a8ac48b1c99f4802cd9a4c3580321bfb251d0e0f892ccf4c` |
| M | `yolo26m_depth_bayese_768x768_nv12.bin` | `f4f2f1958dc16324932b4492490209c817cf7565c3c29240bcf4f0012f9c0be0` |
| L | `yolo26l_depth_bayese_768x768_nv12.bin` | `6a5fa40bda20ee56208ca6e594ecfd9781329385d0baf1b15c9eaa9625286d14` |
| X | `yolo26x_depth_bayese_768x768_nv12.bin` | `61798227fb7e0772a739b483ae5b5acd58a8e785dd7fd9aec5dcac7db0903d91` |

All variants accept a `1x768x768x3` NV12 pyramid input and return `1x192x192x1` float32 calibrated log-depth.

## Download

The script downloads models from the official archive URL. Override `MODEL_BASE_URL` only when using an internal mirror, then run:

```bash
bash download_model.sh
```

Pass one or more variant names to download a subset:

```bash
bash download_model.sh n s
```

The script verifies every downloaded file against the SHA256 values above.
