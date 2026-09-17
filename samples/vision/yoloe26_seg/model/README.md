English | [简体中文](./README_cn.md)

# Released Models

The public model directory is:
`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/yoloe26_seg/`

S100 uses `nash-e/`; S100P uses `nash-m/`. Each contains five HBM files and
matching `.json` metadata, `.names` vocabulary and `manifest.json`.

```bash
# On the target board: automatically select its march.
bash download_model.sh auto n
bash download_model.sh auto all

# Explicit selection also works on a development host.
bash download_model.sh nash-e all
bash download_model.sh nash-m all
```

Optional third argument: destination directory. The default is `model/<march>/`.
HBM names follow `yoloe_26<SIZE>_seg_pf_nashe_640x640_nv12.hbm` or
`yoloe_26<SIZE>_seg_pf_nashm_640x640_nv12.hbm`.

This script downloads deployable HBM files, not upstream PyTorch weights.
It validates file size and SHA256 over HTTPS, preserves verified files, and
refuses to replace a different existing file. No credentials are needed.

The Python runner validates the HBM against its matching JSON. C++ uses the
matching names file and the canonical HBM filename. Keep each model's companion
files together. There are no S600 artifacts in this release.
