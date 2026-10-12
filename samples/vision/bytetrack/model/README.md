English | [简体中文](README_cn.md)

# ByteTrack model artifacts

<a id="artifacts"></a>
## Artifacts

ByteTrack uses a YOLOv5 detector and runs association on the CPU. X5 reuses published YOLOv5 640×640 BIN models, with `n-v7.0` as the default:

| Variant | Asset ID | Filename |
|---|---|---|
| `n-v7.0` | `x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `s-v2.0` | `x5:yolov5:yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `m-v2.0` | `x5:yolov5:yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `l-v2.0` | `x5:yolov5:yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `x-v2.0` | `x5:yolov5:yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | `yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` |
| `s-v7.0` | `x5:yolov5:yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `m-v7.0` | `x5:yolov5:yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `l-v7.0` | `x5:yolov5:yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` |
| `x-v7.0` | `x5:yolov5:yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | `yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` |

See the [YOLOv5 model catalog](../../yolov5/model/README.md) for download URLs and checksums. S targets use these HBM models:

| target | asset ID | filename | URL | SHA-256 |
|---|---|---|---|---|
| S100 | `s:bytetrack:s100/yolov5x_672x672_nv12.hbm` | `s100/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S100P | `s:bytetrack:s100p/yolov5x_672x672_nv12.hbm` | `s100p/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100p/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | `s:bytetrack:s600/yolov5x_672x672_nv12.hbm` | `s600/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

Use publisher checksums from the model catalog when available.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── download_model.sh  # Shell command
```

<a id="preparation"></a>
## Preparation

From the repository root, choose one target and run the explicit downloader in an environment with network access:

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
```

S100:

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

Success prints the exact asset ID, nested `Saved:` path, and observed digest, and creates `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`. Use `s100p` or `s600` for the other rows. The downloader does not fetch the input video.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/coco_classes.names` supplies COCO labels for the detector/visualization.
- `../test_data/bus.jpg` is an example still image.
- `../test_data/readme_img/` contains source reference GIF/PNG images only.
- `../requirements-host.txt` records CPU tracker packages; board SDK is separate.

<a id="local-paths"></a>
## Local paths

X5 models are stored at `samples/vision/bytetrack/model/<BIN filename>`. S-target selection resolves to `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`. External paths must carry the exact qualified asset ID for the same target. `track_test.mp4` must be separately placed under `test_data/` or passed with `--input`.

<a id="formats-checksums"></a>
## Formats and checksums

X5 artifacts use 640x640 packed-NV12 `.bin` format. The three S artifacts are target-relative `.hbm` split-NV12 YOLOv5x deployments at 672x672. Manifest URL and `sha256: null (unknown)` are authoritative; an observed local digest is not publisher authentication.

<a id="license"></a>
## License

Preparation helpers follow Apache-2.0. Model and upstream tracker licenses remain governed by their sources.
