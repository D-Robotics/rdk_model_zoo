# 模型文件

本目录包含编译好的 BPU 模型文件以及下载它们的脚本。

## 目录结构

```text
.
├── download.sh            # HBM 模型下载脚本
├── README.md              # 使用说明 (英文)
└── README_cn.md           # 使用说明 (中文)
```

## 下载模型

运行以下脚本以下载适用于 RDK X5 的预编译 ConvNeXt 模型：

```bash
chmod +x download.sh
./download.sh
```

脚本会将 `.bin` 文件下载到此目录中。
