English | [简体中文](./README_cn.md)

# ViT Models

Use `download_model.sh` to download public S100 HBM models into this sample directory.

```bash
bash download_model.sh s100 int8
```

Download both variants:

```bash
bash download_model.sh s100 all
```

Supported variants are `int8`, `int16`, and `all`.

| Variant | File |
| --- | --- |
| `int8` | `s100/vit_cifar10_batch1_int8.hbm` |
| `int16` | `s100/vit_cifar10_batch1_int16.hbm` |
