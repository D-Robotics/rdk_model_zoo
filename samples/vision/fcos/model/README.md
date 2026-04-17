# FCOS Model Files

This directory contains model download instructions for FCOS detection on RDK X5.

## Default Model

- `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin`

## Supported Models

| Model | Input Size | Platform | Format |
| --- | --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | 512x512 | RDK X5 | `.bin` |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | 768x768 | RDK X5 | `.bin` |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | 896x896 | RDK X5 | `.bin` |

## Download

- Download the default model:
  ```bash
  chmod +x download_model.sh
  ./download_model.sh
  ```

- Download all supported FCOS models:
  ```bash
  chmod +x fulldownload.sh
  ./fulldownload.sh
  ```
