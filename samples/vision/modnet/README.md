English | [简体中文](./README_cn.md)

# MODNet

<a id="overview"></a>
## Algorithm and source

MODNet is a one-stage portrait matting network: one RGB image produces an alpha matte without a trimap. The source is [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) and the paper is [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961). This sample preserves source letterbox-style geometry, RGB normalization, uint8 matte output, and optional background compositing.

<a id="directory"></a>
## Directory structure

```text
modnet/
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
## Support and verification matrix

| target | variant | Python | C++ | status |
|---|---|---|---|---|
| X5 | `modnet_512x512_rgb.bin` | supported | not-supported | the artifact is prepared manually; see [model preparation](model/README.md) |
| S100/S100P/S600 | — | not-supported | not-supported | no source asset |

Runtime language: Python. Prepare the external model manually before inference.

<a id="prerequisites"></a>
## Prerequisites

Board execution requires RDK X5 with `hbm_runtime`, NumPy, and OpenCV. Obtain the external model identified exactly as `x5:modnet:modnet_512x512_rgb.bin`; the active manifest provides no URL and `sha256: null (unknown)`. The model contract is float32 RGB NCHW `(1,3,512,512)` to float32 matte `(1,1,512,512)`.

<a id="quickstart"></a>
## Quick start

Place the external model at `samples/vision/modnet/model/modnet_512x512_rgb.bin`, then run from the repository root:

```bash
python3 -m samples.vision.modnet.runtime.python.main --target x5 \
  --asset-id x5:modnet:modnet_512x512_rgb.bin
```

The command reads `test_data/person.jpg`, writes `test_data/matte.png`, and, when `test_data/bg.jpg` exists, writes `test_data/result.png`. Success is exit code `0` and JSON naming the output paths. No downloader can obtain this manual asset; `model/download.py` only prints the exact preparation requirement and exits `2`.

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
