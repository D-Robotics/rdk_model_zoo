English | [简体中文](./README_cn.md)

# Model Download

Run `download_model.sh` to download the pre-built HBM model. The script reads
`/sys/class/boardinfo/soc_name` (or accepts the SoC as the first argument) and
downloads the matching variant to `/opt/hobot/model/<soc>/basic/`, which is
the same path used by the runtime samples and the default `--model-path` in
`main.py` / `main.cpp`.

| SOC resolved | Model source |
|---|---|
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv2_224x224_nv12.hbm` |
| Other (`s100` / `s100p` / `(null)` / unknown) | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm` |

```bash
./download_model.sh           # auto-detect SOC
./download_model.sh s100      # force S100 build
./download_model.sh s600      # force S600 build
```

The downloaded file lands at:

```text
/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm
```

If the file already exists, the script exits without re-downloading.