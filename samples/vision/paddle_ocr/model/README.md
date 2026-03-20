# 模型下载方式

> ⚠️ **平台说明**：PaddleOCR 模型仅支持 **RDK S100** 平台，不支持 RDK S600。

直接运行 `download_model.sh` 即可将转换好的 hbm 模型下载到此目录：

```bash
./download_model.sh
```

## 模型文件说明

| 文件名                                                | 用途           | 输入格式        | 输入尺寸    |
|------------------------------------------------------|---------------|----------------|------------|
| `cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm`     | 文本检测模型   | NV12（Y+UV）   | 640×640    |
| `cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm`       | 文本识别模型   | Float32 RGB    | 48×320     |

## 下载来源

- 检测模型：`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm`
- 识别模型：`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm`
