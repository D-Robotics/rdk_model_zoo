[English](./README.md) | 简体中文

# ViT 模型

使用 `download_model.sh` 下载公开 S100 HBM 模型到当前 sample 目录。

```bash
bash download_model.sh s100 int8
```

下载两个变体：

```bash
bash download_model.sh s100 all
```

支持的变体为 `int8`、`int16` 和 `all`。

| 变体 | 文件 |
| --- | --- |
| `int8` | `s100/vit_cifar10_batch1_int8.hbm` |
| `int16` | `s100/vit_cifar10_batch1_int16.hbm` |
