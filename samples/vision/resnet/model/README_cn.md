# 模型文件

本目录提供 `RDK X5` 上 ResNet sample 使用的预编译部署模型。

当前运行链路使用 `.bin` 模型文件和 `hbm_runtime`。

## 目录结构

```text
.
├── download.sh
├── README.md
└── README_cn.md
```

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用的模型为：

- `resnet18_224x224_nv12.bin`

## 下载模型

执行以下脚本即可下载默认模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型下载到当前目录。
