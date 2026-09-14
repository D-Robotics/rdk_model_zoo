English | [简体中文](./README_cn.md)

# MobileNetV4 Model Download

Run the download script from this directory:

```bash
bash download_model.sh           # auto-detect SoC from /sys/class/boardinfo/soc_name
bash download_model.sh s100      # force the S100 build
bash download_model.sh s600      # force the S600 build
```

The script reads `/sys/class/boardinfo/soc_name` and routes the download
based on the detected platform. `s600` pulls the S600 build; everything else
(`s100`, `s100p`, `(null)`, unknown) falls back to the S100 build.

Files are downloaded to `./<soc>/` (e.g. `./s100/`, `./s600/`):

| Model | S100 URL | S600 URL |
| --- | --- | --- |
| `mobilenetv4_small_224x224_nv12.hbm` | `rdk_s100/MobileNet/...` | `rdk_s600/MobileNet/...` |
| `mobilenetv4_medium_256x256_nv12.hbm` | `rdk_s100/MobileNet/...` | `rdk_s600/MobileNet/...` |

Full URLs:

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_small_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_medium_256x256_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv4_small_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv4_medium_256x256_nv12.hbm
```
