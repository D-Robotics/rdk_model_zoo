[English](./README.md) | 简体中文

# ResNet152 模型

使用 `download_model.sh` 下载公开 RDK HBM 模型到当前 sample 目录，通过参数指定目标 SoC（`s100` 或 `s600`）：

```bash
# RDK S100
bash download_model.sh s100

# RDK S600
bash download_model.sh s600
```

下载后的文件存放在 `model/<soc>/` 目录：

```text
model/s100/resnet152_224x224_nv12.hbm   # RDK S100
model/s600/resnet152_224x224_nv12.hbm   # RDK S600
```

Python runtime 默认使用 S100 路径，在 RDK S600 上请用 `--model-path` 覆盖。
