# 模型文件

本目录提供 `MODNet` 样例在 `RDK X5` 平台上的预编译部署模型。

当前运行链路使用 `.bin` 模型，并通过 `hbm_runtime` 执行推理。

## 目录结构

```text
.
├── download_model.sh
├── README.md
└── README_cn.md
```

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用以下模型：

- `modnet_512x512_rgb.bin`

## 下载模型

运行以下脚本即可下载默认模型：

```bash
chmod +x download_model.sh
./download_model.sh
```

脚本会将 `.bin` 模型文件下载到本目录。

**注意**：模型下载链接暂未开放。若下载脚本执行失败，请手动将 `modnet_512x512_rgb.bin` 文件放置到本目录中。
