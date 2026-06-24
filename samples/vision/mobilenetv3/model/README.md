English | [简体中文](./README_cn.md)

# Model Download

Run the download script from this directory:

```bash
bash download_model.sh           # auto-detect SoC from /sys/class/boardinfo/soc_name
bash download_model.sh s100      # force the S100 build
bash download_model.sh s600      # force the S600 build
```

The script reads `/sys/class/boardinfo/soc_name` and routes the download
based on the detected platform. `s600` pulls the S600 build; everything else
(`s100`, `s100p`, `(null)`, unknown) falls back to the S100 build.

The model is downloaded to:

```text
model/<soc>/mobilenetv3_224x224_nv12.hbm   # <soc> ∈ {s100, s600}
```

## Published Model

| File | Platform | Input |
| --- | --- | --- |
| `mobilenetv3_224x224_nv12.hbm` | S100 / S600 | NV12 (Y + UV) |

Download sources:

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv3_224x224_nv12.hbm
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv3_224x224_nv12.hbm
```
