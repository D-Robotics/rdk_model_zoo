# 模型文件

本目录提供 `RDK X5` 上 MobileNetV4 sample 使用的预编译部署模型。

当前运行链路使用 `.bin` 模型文件和 `hbm_runtime`。

## 模型列表

- `MobileNetV4_conv_small_224x224_nv12.bin`
- `MobileNetV4_conv_medium_224x224_nv12.bin`

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用：

- `MobileNetV4_conv_small_224x224_nv12.bin`

## 下载模型

执行以下脚本下载两个模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型下载到当前目录。
