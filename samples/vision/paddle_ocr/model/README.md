English | [简体中文](./README_cn.md)

# Model Download

This directory stores the PP-OCRv6 detection/recognition HBM models. The `download_model.sh` script reads `/sys/class/boardinfo/soc_name` and automatically selects the correct prebuilt variant for the current board:

| SOC | Model source |
|---|---|
| `s100` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/` |
| `s100p` | Falls back to `rdk_s100/paddle_ocr/` |
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/paddle_ocr/` |
| Other / read failure | Falls back to `rdk_s100/paddle_ocr/` |

Run `download_model.sh` to download the HBM models to this directory:

```bash
./download_model.sh
```

## Model Files

| File | Purpose | Input format | Input size |
|---|---|---|---|
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | Text detection model | NV12 (Y+UV) | 640×640 |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | Text recognition model | Float32 RGB | 48×320 |

## Download Sources

The download URL is assembled by `download_model.sh` based on the detected SOC. For example on RDK S100:

- Detection: `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`
- Recognition: `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`

On RDK S600 the `rdk_s100/` segment is automatically replaced with `rdk_s600/`.