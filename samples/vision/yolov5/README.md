English | [简体中文](./README_cn.md)

# YOLOv5

<a id="overview"></a>
## Algorithm and source

YOLOv5 is a one-stage, anchor-based object detector. It predicts three feature-map scales and returns COCO-style boxes, scores, and class IDs. The source resource is [ultralytics/yolov5](https://github.com/ultralytics/yolov5); this sample keeps the source X5 and S tensor protocols as separate target contracts.

<a id="support-matrix"></a>
## Support and verification matrix

| target | variant(s) | Python | C++ | status |
|---|---|---|---|---|
| X5 | n-v7.0, s/m/l/x-v2.0, s/m/l/x-v7.0 | supported-not-run | supported-not-run | host contract fixtures; board not-run |
| S100 | x-672 | supported-not-run | supported-not-run | host contract fixtures; board not-run |
| S100P | — | not-supported | not-supported | no YOLOv5 manifest asset |
| S600 | x-672 | supported-not-run | supported-not-run | host contract fixtures; board not-run |

The Python X5 path uses one packed NV12 input and defaults to direct stretch; S Python uses split Y/UV inputs and defaults to letterbox. The C++ runtime has its own source defaults and remains documented in `runtime/cpp/`; this README does not claim either language was run on a board.

<a id="prerequisites"></a>
## Prerequisites

Host checks use Python 3, NumPy, OpenCV, and PyYAML for manifest tools. Board inference additionally requires the matching target image and `hbm_runtime`; S tracking/detection CPU post-processing may need SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`. Conversion needs the matching RDK OpenExplorer environment. Published SHA-256 values are unknown in the active manifests.

<a id="quickstart"></a>
## Quick start

From the repository root, prepare one published artifact explicitly, then run the Python detector. This example uses X5 `n-v7.0`; it was not downloaded or run in this migration:

```bash
python3 -m samples.vision.yolov5.model.download \
  --target x5 --variant n-v7.0 \
  --output-dir samples/vision/yolov5/model
python3 -m samples.vision.yolov5.runtime.python.main \
  --target x5 --variant n-v7.0
```

The first command writes `samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` and prints an observed digest; publisher SHA-256 is unknown. The second reads `test_data/bus.jpg`, writes `test_data/result_unified.jpg`, prints detection arrays, and exits `0` on success. For S100, replace both target/variant with `--target s100 --variant x-672`; its file is `model/s100/yolov5x_672x672_nv12.hbm` and the default image is `test_data/kite.jpg`.

`--list-models` is manifest-only. `--dry-run` requires an explicit target and reports the target protocol without loading SDK or model data. Runtime and `run.sh` never download or install anything.

<a id="expected-results"></a>
## Expected results

A successful Python run returns JSON arrays `boxes`, `scores`, and `class_ids`, then a saved annotated image. Boxes are original-image XYXY coordinates, scores are `[0,1]`, and class IDs are COCO indices. X5 boxes are integer-truncated after inverse resize; S boxes retain subpixel float coordinates. Empty detections use shapes `(0,4)`, `(0,)`, and `(0,)`. Exact object counts are model/input dependent and are not invented here.

<a id="directory"></a>
## Directory

```text
.
├── model/                 # explicit manifest artifact preparation
├── runtime/python/        # binding, NV12 I/O, task, runner, CLI, visualization
├── runtime/cpp/           # target-specific C++ implementation and README pair
├── conversion/            # source YAMLs and export/PTQ limits
├── evaluator/             # same-board source/unified evidence comparator
├── test_data/             # bus/kite images, labels, historical result assets
└── tests/                 # host numerical, metadata, CLI, and evaluator fixtures
```

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): all X5/S assets, URLs, paths, and checksum facts.
- [`runtime/python/README.md`](./runtime/python/README.md): parser, stages, target-specific tensor contracts, and API example.
- [`runtime/cpp/README.md`](./runtime/cpp/README.md): existing C++ build and run contract; it is maintained separately.
- [`conversion/README.md`](./conversion/README.md): verbatim X5 YAMLs and source export/compile commands.
- [`evaluator/README.md`](./evaluator/README.md): complete raw-array source/unified comparison.

<a id="historical-performance"></a>
## Historical source performance

The source X5 reference table is preserved here; these measurements are historical and were not re-run:

| Model | Size | Params | BPU throughput | Python post-process |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

<a id="license"></a>
## License

Repository wrapper code and source sample documentation follow Apache-2.0. The upstream YOLOv5 project and any downloaded weights retain their own license and provenance; a manifest `sha256: null (unknown)` is not authentication.
