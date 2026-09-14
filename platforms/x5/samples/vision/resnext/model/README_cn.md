# 模型文件

本目录提供 ResNeXt sample 在 `RDK X5` 上使用的预编译部署模型。

当前维护的运行链路使用 `.bin` 模型和 `hbm_runtime`。

## 模型列表

- `ResNeXt50_32x4d_224x224_nv12.bin`

## 默认模型

`runtime/python/run.sh` 与 `runtime/python/main.py` 默认使用：

- `ResNeXt50_32x4d_224x224_nv12.bin`

## 下载模型

执行以下脚本下载 ResNeXt 模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型下载到当前目录。
