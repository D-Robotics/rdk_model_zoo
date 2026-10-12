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
| S100P | `s:bytetrack:s100p/yolov5x_672x672_nv12.hbm` | `s100p/yolov5x_672x672_nv12.hbm` | not published (no archive file) | — |
| S600 | `s:bytetrack:s600/yolov5x_672x672_nv12.hbm` | `s600/yolov5x_672x672_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

`null` in the table means the publisher has not provided a SHA-256 checksum. No S100P detector file is published: convert a 672×672 YOLOv5x HBM for the nash-m march with the [conversion guide](../conversion/README.md) and run it with `--model-path` plus the S100P `--asset-id` above.

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

Run the download command from the repository root with network access:

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
```

S100:

```bash
python3 -m samples.vision.bytetrack.model.download \
  --target s100 --output-dir samples/vision/bytetrack/model
```

After downloading, the script prints the model identifier, saved path, and file SHA-256. The S100 command above saves `samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm`. Use `--target s600` to download the S600 file. S100P has no published detector: convert the nash-m HBM as described in the [conversion guide](../conversion/README.md), place it at `samples/vision/bytetrack/model/s100p/yolov5x_672x672_nv12.hbm`, and pass `--model-path` with the S100P asset ID from the table.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/coco_classes.names` supplies COCO labels for the detector/visualization.
- `../test_data/bus.jpg` is an example still image.
- `../test_data/readme_img/` contains tracking examples as GIF/PNG images.
- `../requirements-host.txt` lists Python tracking dependencies.

<a id="local-paths"></a>
## Local paths

X5 models are saved under `samples/vision/bytetrack/model/<BIN filename>`. S models are saved under `samples/vision/bytetrack/model/<target>/yolov5x_672x672_nv12.hbm`. To use another path, pass `--model-path` with the matching full `--asset-id` from the table. Select your video with `--input`.

<a id="formats-checksums"></a>
## Formats and checksums

X5 uses 640×640 packed-NV12 `.bin` models; S boards use 672×672 split-NV12 `.hbm` models. The downloader prints the local file’s SHA-256. Compare it with the publisher’s checksum when one is available.

<a id="license"></a>
## License

Download scripts use Apache-2.0. Model weights and ByteTrack source retain their upstream licenses.
