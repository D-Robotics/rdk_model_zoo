# 模型文件

本目录提供 PP-LiteSeg-STDC1 样例在 RDK X5 平台上的预编译部署模型。

当前运行链路使用 `.bin` 模型，并通过 `hbm_runtime` 执行推理。

## 目录结构

```text
.
├── download.sh
├── README.md
└── README_cn.md
```

## 默认模型

`runtime/python/run.sh` 和 `runtime/python/main.py` 默认使用以下模型：

- `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`

## 下载模型

运行以下脚本即可下载默认模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 模型文件下载到本目录。

> 如需自行转换，参考 [../conversion/README_cn.md](../conversion/README_cn.md)。
