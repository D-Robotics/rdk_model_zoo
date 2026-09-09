English | [简体中文](./README_cn.md)

# YOLOE-11 PF Model Artifacts

Target: RDK X5, 640x640 NV12. The following BINs are available for download; their contents match the locally compiled artifacts.

| Model | Download | Size (MB) |
| --- | --- | ---: |
| YOLOE-11s-Seg-PF | [yoloe_11s_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11s_seg_pf_bayese_640x640_nv12.bin) | 13.17 |
| YOLOE-11m-Seg-PF | [yoloe_11m_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11m_seg_pf_bayese_640x640_nv12.bin) | 26.73 |
| YOLOE-11l-Seg-PF | [yoloe_11l_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11l_seg_pf_bayese_640x640_nv12.bin) | 32.83 |

## Download

Run from this directory. With no arguments, the script downloads s, preserving the default runtime startup behavior. Requires Bash, curl and sha256sum.

```bash
bash download_model.sh
bash download_model.sh m
bash download_model.sh l
bash download_model.sh all
```

Files are saved beside the script, regardless of the current working directory. Existing files with matching SHA256 are reused. Mismatched files are not overwritten; move them aside before retrying. Downloads use temporary files and are published locally only after checksum verification.

## SHA256

| Model | SHA256 |
| --- | --- |
| 11s-PF | `8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa` |
| 11m-PF | `ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31` |
| 11l-PF | `a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2` |

See the [conversion guide](../conversion/README.md) and [runtime guide](../runtime/python/README.md). Use `--model-path` to select m/l; all three models use the repository's 4585-class label file.
