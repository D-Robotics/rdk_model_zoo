# 模型下载方式

本目录用于存放 PP-OCRv6 检测/识别 HBM 模型。`download_model.sh` 会读取 `/sys/class/boardinfo/soc_name`，根据当前板卡自动选择对应的预编译版本：

| SOC          | 模型来源                                                                                           |
|--------------|----------------------------------------------------------------------------------------------------|
| `s100`       | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/`                       |
| `s100p`      | 默认走 `rdk_s100/paddle_ocr/`                                                                       |
| `s600`       | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/paddle_ocr/`                       |
| 其他 / 读取失败 | 默认走 `rdk_s100/paddle_ocr/`                                                                       |

直接运行 `download_model.sh` 即可将转换好的 hbm 模型下载到此目录：

```bash
./download_model.sh
```

## 模型文件说明

| 文件名                                              | 用途           | 输入格式        | 输入尺寸    |
|-----------------------------------------------------|---------------|----------------|------------|
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`        | 文本检测模型   | NV12（Y+UV）   | 640×640    |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`          | 文本识别模型   | Float32 RGB    | 48×320     |

## 下载来源

模型下载地址由 `download_model.sh` 自动按 SOC 拼接，例如在 RDK S100 上：

- 检测模型：`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`
- 识别模型：`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`

在 RDK S600 上将自动替换为 `rdk_s600/paddle_ocr/` 路径下的同名文件。
