English | [简体中文](./README_cn.md)

# ResNet50 Model Download

Run the download script from this directory, passing the target SoC (`s100` or `s600`):

```bash
# RDK S100
bash download_model.sh s100

# RDK S600
bash download_model.sh s600
```

The script downloads the HBM file to `./<soc>/`:

| SoC | Local path | URL |
| --- | --- | --- |
| `s100` | `./s100/resnet50_224x224_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet50_224x224_nv12.hbm` |
| `s600` | `./s600/resnet50_224x224_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet50_224x224_nv12.hbm` |

This sample uses the public RDK ResNet50 HBM model. The file name is the same across SoCs; only the archive sub-directory differs.
