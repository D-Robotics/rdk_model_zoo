# 模型文件

[English](./README.md) | 简体中文

本目录提供 `RDK X5` 上 HGNetV2 sample 使用的预编译模型。当前运行链路为 `hbm_runtime` 加载 `.bin` 模型。

## 目录结构

```text
.
├── download.sh
├── README.md
└── README_cn.md
```

## 可用模型

| 变种 | 文件名 | 体积 |
| --- | --- | --- |
| HGNetV2 b0 | `hgnetv2_b0_224x224_nv12.bin` | ~5.9 MB |
| HGNetV2 b1 | `hgnetv2_b1_224x224_nv12.bin` | ~6.2 MB |
| HGNetV2 b2 | `hgnetv2_b2_224x224_nv12.bin` | ~11 MB |
| HGNetV2 b3 | `hgnetv2_b3_224x224_nv12.bin` | ~16 MB |
| HGNetV2 b4 | `hgnetv2_b4_224x224_nv12.bin` | ~19 MB |

`runtime/python/run.sh` 与 `runtime/python/main.py` 默认使用 `hgnetv2_b0_224x224_nv12.bin`。

## 下载模型

默认仅下载 `runtime/python/run.sh` 需要的 b0 变种：

```bash
chmod +x download.sh
./download.sh                 # 仅 b0(约 5.9 MB)
```

如需下载其他变种,传入变种名(或 `all`)：

```bash
./download.sh b3 b4           # 下载 b3 与 b4
./download.sh all             # 下载全部 5 个变种(约 57 MB)
```

脚本会把 `.bin` 文件下载到本目录,已存在的文件会自动跳过。
