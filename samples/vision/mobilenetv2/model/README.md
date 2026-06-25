English | [简体中文](./README_cn.md)

# Model Download

Run `download_model.sh` to download the pre-built HBM model to this directory. The script reads `/sys/class/boardinfo/soc_name` and automatically selects the correct prebuilt variant for the current board:

| SOC resolved | Model source |
|---|---|
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv2_224x224_nv12.hbm` |
| Other (`s100` / `s100p` / `(null)` / unknown) | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm` |

```bash
./download_model.sh           # auto-detect SOC
./download_model.sh s100      # force S100 build
./download_model.sh s600      # force S600 build
```