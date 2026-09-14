[English](./README.md) | 简体中文

# 模型下载

在本目录执行下载脚本：

```bash
bash download_model.sh           # 按 /sys/class/boardinfo/soc_name 自动识别
bash download_model.sh s100      # 强制下载 S100 版
bash download_model.sh s600      # 强制下载 S600 版
```

脚本根据当前 SOC 路由：`s600` 拉取 S600 版；其它（`s100` / `s100p` / `(null)` / 未知）回落到 S100 版。

模型会下载到：

```text
model/<soc>/mobilenetv3_224x224_nv12.hbm   # <soc> ∈ {s100, s600}
```

## 已发布模型

| 文件 | 平台 | 输入 |
| --- | --- | --- |
| `mobilenetv3_224x224_nv12.hbm` | S100 / S600 | NV12 (Y + UV) |

下载地址：

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv3_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv3_224x224_nv12.hbm
```
