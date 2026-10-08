English | [简体中文](README_cn.md)

# YOLOv5

<a id="overview"></a>
## Algorithm and source

YOLOv5 is a one-stage, anchor-based object detector: a CSPDarknet backbone with FPN+PAN feature fusion feeds detection heads on three feature-map scales (strides 8/16/32), which return COCO-style boxes, scores, and class IDs. The `n/s/m/l/x` scale variants trade speed against accuracy. The source resource is [ultralytics/yolov5](https://github.com/ultralytics/yolov5). X5 and S models use different input tensors and output quantization metadata.

<a id="directory"></a>
## Directory structure

```text
yolov5/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Supported models and boards

| target | variant(s) | Python | C++ | note |
|---|---|---|---|---|
| X5 | n-v7.0, s/m/l/x-v2.0, s/m/l/x-v7.0 | supported | supported | all nine Python variants; C++ build/run per `runtime/cpp/` |
| S100 | x-672 | supported | supported | Python uses `kite.jpg`; C++ per `runtime/cpp/` |
| S100P | — | not-supported | not-supported | no YOLOv5 manifest asset; explicitly rejected |
| S600 | x-672 | supported | supported | Python uses `kite.jpg`; C++ per `runtime/cpp/` |

Python and C++ entries use the target-specific artifacts listed above. Prepare the selected variant and its matching board SDK before running; choose `bus.jpg` for X5 and `kite.jpg` for the S100/S600 672×672 model. Dataset scoring follows the [evaluator](evaluator/README.md).

The Python X5 path uses one packed NV12 input and defaults to direct stretch; S Python uses split Y/UV inputs and defaults to letterbox. The C++ runtime has its own source defaults and remains documented in `runtime/cpp/`.

<a id="prerequisites"></a>
## Prerequisites

Use Python 3, NumPy, OpenCV and PyYAML. Board inference additionally requires the matching target image and `hbm_runtime`; S tracking/detection CPU post-processing may need SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`. Conversion needs the matching RDK OpenExplorer environment. Published SHA-256 values are unknown in the active manifests.

<a id="quickstart"></a>
## Quick start

From the repository root, prepare one published artifact explicitly, then run the Python detector. This example uses X5 `n-v7.0` with its default quick-check image:

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

A successful Python run returns JSON arrays `boxes`, `scores`, and `class_ids`, then a saved annotated image. Boxes are original-image XYXY coordinates, scores are `[0,1]`, and class IDs are COCO indices. X5 boxes are integer-truncated after inverse resize; S boxes retain subpixel float coordinates. Empty detections use shapes `(0,4)`, `(0,)`, and `(0,)`. The number of detections depends on the selected model and input image.

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): all X5/S assets, URLs, paths, and checksum facts.
- [`runtime/python/README.md`](./runtime/python/README.md): parser, stages, target-specific tensor contracts, and API example.
- [`runtime/cpp/README.md`](./runtime/cpp/README.md): existing C++ build and run contract; it is maintained separately.
- [`conversion/README.md`](./conversion/README.md): X5 YAML configurations and export/compile commands.
- [`evaluator/README.md`](./evaluator/README.md): input, raw-output and detection-result comparisons.

<a id="historical-performance"></a>
## Reference performance

The source X5 reference table:

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

Repository helper code and source sample documentation follow Apache-2.0. The upstream YOLOv5 project and any downloaded weights retain their own license and provenance; a manifest `sha256: null (unknown)` is not authentication.
