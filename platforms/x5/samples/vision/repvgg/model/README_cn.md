[English](./README.md) | 简体中文

# 模型文件

本目录提供 RepVGG sample 在 `RDK X5` 上使用的预编译部署模型。

当前运行路径使用 `.bin` 模型和 `hbm_runtime`。

## 模型列表

- `RepVGG_A0_224x224_nv12.bin`
- `RepVGG_A1_224x224_nv12.bin`
- `RepVGG_A2_224x224_nv12.bin`
- `RepVGG_B0_224x224_nv12.bin`
- `RepVGG_B1g2_224x224_nv12.bin`
- `RepVGG_B1g4_224x224_nv12.bin`

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用：

- `RepVGG_A0_224x224_nv12.bin`

## 下载模型

执行以下脚本下载 RepVGG 模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型文件下载到本目录。
