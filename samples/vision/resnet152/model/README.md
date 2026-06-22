English | [简体中文](./README_cn.md)

# ResNet152 Model

Use `download_model.sh` to download the public RDK HBM model into this sample directory. Pass the target SoC (`s100` or `s600`):

```bash
# RDK S100
bash download_model.sh s100

# RDK S600
bash download_model.sh s600
```

The downloaded file is placed under `model/<soc>/`:

```text
model/s100/resnet152_224x224_nv12.hbm   # RDK S100
model/s600/resnet152_224x224_nv12.hbm   # RDK S600
```

The Python runtime uses the S100 path by default; override `--model-path` when running on RDK S600.
