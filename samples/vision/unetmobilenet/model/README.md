English | [简体中文](README_cn.md)

# UNetMobileNet model preparation

<a id="artifacts"></a>
## Artifacts

| Target | File under model/ | Exact asset-id |
| --- | --- | --- |
| s100 | s100/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm |
| s600 | s600/unet_mobilenet_1024x2048_nv12.hbm | s:unetmobilenet:s600/unet_mobilenet_1024x2048_nv12.hbm |

Both are HBM deployment artifacts. [Manifest](../../../../platforms/s/docs/release/models.yaml). The identical basenames do not mean interchangeable model bytes. No S100P/X5 artifact is published.

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root; explicit network preparation
bash samples/vision/unetmobilenet/model/download.sh --target s100
bash samples/vision/unetmobilenet/model/download.sh --target s600
```

`download_model.sh` forwards the same arguments for compatibility. Failed downloads can be retried or fetched manually from [S100](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm) / [S600](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm) into the corresponding subdirectory. Preparation never proves local hardware identity.

<a id="accompanying-files"></a>
## Accompanying files

No external class file is needed; the fixed 19-class IDs use source rdk_colors for display. segmentation.png is an example input; result.jpg is historical output, neither is a labeled validation set.

<a id="local-paths"></a>
## Local paths

Runtime defaults resolve from the sample directory, not cwd: model/<target>/unet_mobilenet_1024x2048_nv12.hbm. This replaces the old /opt/hobot/model/<soc>/basic default. Downloader --output-dir changes its root while preserving target subdirectories. For an existing /opt copy, specify both --model-path /opt/hobot/model/s100/basic/unet_mobilenet_1024x2048_nv12.hbm and --asset-id s:unetmobilenet:s100/unet_mobilenet_1024x2048_nv12.hbm. Paths and target must agree.

<a id="formats-checksums"></a>
## Formats and checksums

Both HBM rows have `sha256: null (unknown)`. The downloader prints observed SHA-256 for tracking copies, not independent authentication. Runtime metadata guards prevent wrong shapes/dtypes, but cannot prove a custom file is the publisher artifact. Preserve custom-file provenance and never rename an S100 artifact to imply S600 support.
