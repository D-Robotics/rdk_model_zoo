English | [简体中文](README_cn.md)

# MODNet

<a id="overview"></a>
## Algorithm and source

MODNet is a one-stage portrait matting network. It predicts an alpha matte from a single image without requiring a trimap; this sample also supports compositing the portrait over a chosen background.

References: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet), [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961).

<a id="directory"></a>
## Directory structure

```text
modnet/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Supported models

| target | variant | Python | C++ | status |
|---|---|---|---|---|
| X5 | `modnet_512x512_rgb.bin` | supported | not-supported | the artifact is prepared manually; see [model preparation](model/README.md) |
| S100/S100P/S600 | `photographic-portrait-512-s-o1-20261103` | native binding verified | not-supported | HBM converted; board execution/accuracy/performance `not-run` |

Runtime language: Python. Download an exact photographic artifact or prepare the historical manual model before inference.

The public photographic model has a separate [conversion recipe](conversion/README.md). Its new filename `modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin` uses exact asset ID `x5:modnet:modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`; the historical manual model remains the default. Board performance and accuracy for the new model are pending; the historical table below does not describe it.

<a id="prerequisites"></a>
## Prerequisites

Board execution requires the selected RDK X5/S100/S100P/S600 target with its matching `hbm_runtime`, NumPy, and OpenCV. The historical default is the external model identified exactly as `x5:modnet:modnet_512x512_rgb.bin`; the active manifest provides no URL and `sha256: null (unknown)`. The model contract is float32 RGB NCHW `(1,3,512,512)` to float32 matte `(1,1,512,512)`.

<a id="quickstart"></a>
## Quick start

Place the external model at `samples/vision/modnet/model/modnet_512x512_rgb.bin`, then run from the repository root:

```bash
python3 -m samples.vision.modnet.runtime.python.main --target x5 \
  --asset-id x5:modnet:modnet_512x512_rgb.bin
```

The command reads `test_data/person.jpg`, writes `test_data/matte.png`, and, when `test_data/bg.jpg` exists, writes `test_data/result.png`. Success is exit code `0` and JSON naming the output paths. No downloader can obtain this manual asset; `model/download.py` reports this legacy manual requirement and exits `2`; an explicitly selected published photographic artifact uses the manifest downloader.

<a id="expected-results"></a>
## Expected results

The matte is an 8-bit grayscale PNG with the original input height and width. The optional composite is a BGR PNG with the original input geometry. Exact alpha values and quality depend on the external model; the source performance record is listed below.

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): manual model identity, path, and unknown checksum.
- [`runtime/python/README.md`](./runtime/python/README.md): CLI and `MODNetTask` API.
- [`conversion/README.md`](./conversion/README.md): source conversion capability and real gaps.
- [`evaluator/README.md`](./evaluator/README.md): complete saved-matte comparison.

<a id="historical-performance"></a>
## Source-recorded performance

The complete source table and test conditions:

| Model | Size | Input format | Latency (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet (2 threads) | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

Conditions: RDK X5, CPU 8xA55@1.8G, BPU 1xBayes-e@1G (10TOPS INT8). Single-thread latency used one frame, one thread, and one BPU core; multi-thread FPS used two concurrent threads.

<a id="license"></a>
## License

Sample code follows the repository Apache-2.0 license. Preserve MODNet copyright notices and the upstream license for operator-supplied model artifacts.

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

Independent converted profiles and explicit commands are listed in [model preparation](model/README.md). Board accuracy and performance are not measured yet.
