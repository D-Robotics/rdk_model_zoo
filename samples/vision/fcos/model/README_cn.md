# FCOS 模型文件

本目录提供 FCOS 在 RDK X5 上使用的模型下载说明。

## 默认模型

- `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin`

## 支持模型

| 模型 | 输入尺寸 | 平台 | 格式 |
| --- | --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | 512x512 | RDK X5 | `.bin` |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | 768x768 | RDK X5 | `.bin` |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | 896x896 | RDK X5 | `.bin` |

## 下载方式

- 下载默认模型：
  ```bash
  chmod +x download_model.sh
  ./download_model.sh
  ```

- 下载全部支持模型：
  ```bash
  chmod +x fulldownload.sh
  ./fulldownload.sh
  ```
