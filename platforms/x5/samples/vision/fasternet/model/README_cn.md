[English](./README.md) | 简体中文

# 模型文件

本目录提供 FasterNet sample 在 `RDK X5` 上使用的预编译部署模型。

当前运行链路使用 `hbm_runtime` 加载 `.bin` 模型。

## 模型列表

- `FasterNet_S_224x224_nv12.bin`
- `FasterNet_T0_224x224_nv12.bin`
- `FasterNet_T1_224x224_nv12.bin`
- `FasterNet_T2_224x224_nv12.bin`

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用：

- `FasterNet_S_224x224_nv12.bin`

## 下载模型

执行以下脚本下载全部 FasterNet 模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型文件下载到当前目录。
