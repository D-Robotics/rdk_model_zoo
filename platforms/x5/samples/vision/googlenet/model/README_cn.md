[English](./README.md) | 简体中文

# 模型文件

本目录提供 `RDK X5` 上 GoogLeNet sample 使用的预编译部署模型。

当前运行链路使用 `.bin` 模型文件和 `hbm_runtime`。

## 模型列表

- `googlenet_224x224_nv12.bin`

## 默认模型

`runtime/python/main.py` 默认使用板端预置模型：

- `/opt/hobot/model/x5/basic/googlenet_224x224_nv12.bin`

`model/download.sh` 下载的本地模型为：

- `googlenet_224x224_nv12.bin`

## 下载模型

执行以下脚本下载模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型下载到当前目录。
