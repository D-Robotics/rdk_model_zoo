# YOLOv5 model artifacts

<a id="artifacts"></a>
## Artifacts

The active manifest publishes nine X5 artifacts and two S artifacts. The `{size}` notation below is a documentation shorthand for each of `s`, `m`, `l`, and `x`; it is not a filename to pass literally.

| target | variant | filename | format | URL | SHA-256 |
|---|---|---|---|---|---|
| X5 | n-v7.0 | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v2.0 | `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v2.0 | `yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v2.0 | `yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v2.0 | `yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v7.0 | `yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v7.0 | `yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v7.0 | `yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v7.0 | `yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| S100 | x-672 | `s100/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | x-672 | `s600/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

S100P has no YOLOv5 asset. Every manifest publisher checksum is `sha256: null (unknown)`.

<a id="preparation"></a>
## Preparation

Run from the repository root in an environment where explicit network access is allowed. This migration did not run it:

```bash
python3 -m samples.vision.yolov5.model.download \
  --target x5 --variant n-v7.0 \
  --output-dir samples/vision/yolov5/model
```

For S100/S600, use `--target s100 --variant x-672` or `--target s600 --variant x-672`. The script creates the nested target path and prints the observed digest plus the publisher value (`unknown`). It downloads only the selected manifest row. `run.sh` and runtime never call this helper.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/coco_classes.names` is the 80-line COCO label file used only for visualization.
- `../test_data/bus.jpg` is the X5 default image; `../test_data/kite.jpg` is the S default image.
- The `conversion/` YAMLs are X5 source configuration references, not model binaries.

<a id="local-paths"></a>
## Local paths

With the commands above, X5 files are under `samples/vision/yolov5/model/`; S files are under `samples/vision/yolov5/model/s100/` or `model/s600/`. A custom `--model-path` must be paired with the exact matching `--asset-id`; filenames alone do not establish target identity.

<a id="formats-checksums"></a>
## Formats and checksums

X5 artifacts are flat `.bin` Bayes-e/NV12 deployments at 640x640. S artifacts are target-relative `.hbm` Nash-e deployments at 672x672 with split NV12 inputs. The manifest does not record publisher SHA-256 values; local observed hashes printed by the downloader are evidence of downloaded bytes only.

<a id="license"></a>
## License

The repository helper follows Apache-2.0. Model provenance and any upstream weight license remain tied to the manifest/source release.
