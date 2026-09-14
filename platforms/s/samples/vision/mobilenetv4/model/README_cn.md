[English](./README.md) | 简体中文

# MobileNetV4 模型下载

在本目录运行下载脚本：

```bash
bash download_model.sh           # 按 /sys/class/boardinfo/soc_name 自动识别
bash download_model.sh s100      # 强制下载 S100 版
bash download_model.sh s600      # 强制下载 S600 版
```

脚本根据当前 SOC 路由：`s600` 拉取 S600 版；其它（`s100` / `s100p` / `(null)` / 未知）回落到 S100 版。

文件下载到 `./<soc>/`（例如 `./s100/`、`./s600/`）：

| 模型 | S100 URL | S600 URL |
| --- | --- | --- |
| `mobilenetv4_small_224x224_nv12.hbm` | `rdk_s100/MobileNet/...` | `rdk_s600/MobileNet/...` |
| `mobilenetv4_medium_256x256_nv12.hbm` | `rdk_s100/MobileNet/...` | `rdk_s600/MobileNet/...` |

完整下载地址：

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_small_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv4_medium_256x256_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv4_small_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv4_medium_256x256_nv12.hbm
```
