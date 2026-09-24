English | [简体中文](./README_cn.md)

# LPRNet

<a id="overview"></a>
## Algorithm and source

LPRNet recognizes a cropped license-plate tensor as a character sequence without a separate character detector. This migration preserves the source X5 protocol: a pre-packed `float32` file is reshaped to `1x3x24x94`; it does not invent image decoding, resize, or normalization. The source paper is [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447).

<a id="support-matrix"></a>
## Support and verification matrix

| target | variant | Python | C++ | status |
|---|---|---|---|---|
| X5 | `lpr.bin` | supported-not-run | not-supported | host fixtures pass; board binding remediated 2026-09-24 (native output `(1,68,18,1)`), board re-run pending |
| S100/S100P/S600 | — | not-supported | not-supported | no source asset |

This sample has no C++ implementation. Host tests do not certify board execution.

<a id="prerequisites"></a>
## Prerequisites

Board execution requires RDK OS `>=3.5.0`, the matching board `hbm_runtime`, the published X5 `lpr.bin`, and Python with NumPy. The bundled `test_input.dat` is already a float32 tensor; no image package is needed for the LPR task. The model hash is `sha256: null (unknown)` in the active manifest.

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the published asset explicitly, then run on a recognized X5 board:

```bash
# cwd: repository root
python3 -m samples.vision.lprnet.model.download \
  --target x5 --output-dir samples/vision/lprnet/model
python3 -m samples.vision.lprnet.runtime.python.main --target x5
```

The first command writes `model/lpr.bin` and reports the observed hash; the manifest publisher hash is unknown. The second command reads `test_data/test_input.dat`, runs the model, and prints JSON containing `plate`. The downloader is never called by runtime or by `run.sh`. For a host-only selection check, use `python3 -m samples.vision.lprnet.runtime.python.main --dry-run --target x5`.

<a id="expected-results"></a>
## Expected results

Successful inference exits with code `0` and prints a JSON object containing `target`, the qualified `asset_id`, and a decoded `plate` string. The exact plate is model/input dependent and is not fabricated here; `test_data/example.jpg` is only the visual reference shipped by the source, while `test_input.dat` is the actual runtime input.

<a id="directory"></a>
## Directory

```text
.
├── model/                 # explicit model preparation and model notes
├── runtime/python/        # binding, lazy runner, task, CLI, and run.sh
├── conversion/            # source conversion facts and unavailable recipe
├── evaluator/             # self-contained raw/text comparison utility
├── test_data/              # source test_input.dat and example.jpg
└── tests/                 # host CTC, metadata, task, and CLI fixtures
```

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): manifest asset, download, path, and checksum facts.
- [`runtime/python/README.md`](./runtime/python/README.md): CLI and `LPRNetTask` API.
- [`conversion/README.md`](./conversion/README.md): source OE commands and missing reproducibility inputs.
- [`evaluator/README.md`](./evaluator/README.md): exact raw/text comparison procedure.

<a id="historical-performance"></a>
## Historical source performance

The complete source benchmark row is retained below. It is historical source data and was not re-run in this migration.

| Model | Test frames | FPS | Average latency | BPU usage | ION memory |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="license"></a>
## License

The source sample and repository code follow the repository Apache-2.0 license. The LPRNet paper and upstream project remain their authors' references; model provenance and the unknown publisher checksum are recorded in the model README.
