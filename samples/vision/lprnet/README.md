English | [简体中文](README_cn.md)

# LPRNet

<a id="overview"></a>
## Algorithm and source

LPRNet recognizes license-plate text without a separate character detector. The Python entry runs the selected model on a prepared plate tensor and decodes its character sequence with CTC.

References: [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447).

<a id="directory"></a>
## Directory structure

```text
lprnet/
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

| target | variant | Python | C++ | note |
|---|---|---|---|---|
| X5 | `lpr.bin` | supported | not-supported | verify with the bundled `test_input.dat` |
| S100/S100P/S600 | `sirius-68class` | integrated; board not-run | not-supported | independent HBM conversions, 2026-11-04 |

Python is the available runtime. The evaluator compares raw `(1,68,18,1)` logits and decoded text for the bundled input; dataset recognition accuracy requires labeled plates.

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

Successful inference exits with code `0` and prints a JSON object containing `target`, the qualified `asset_id`, and a decoded `plate` string. The decoded plate depends on the model and input; `test_data/example.jpg` is only the visual reference shipped by the source, while `test_input.dat` is the actual runtime input.

The CLI consumes pre-packed float32 `test_input.dat`, reshaped to `(1,3,24,94)`. Decode, resize and normalize plate images before generating this tensor.

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): manifest asset, download, path, and checksum facts.
- [`runtime/python/README.md`](./runtime/python/README.md): CLI and `LPRNetRecognizer` API.
- [`conversion/README.md`](./conversion/README.md): source OE commands and missing reproducibility inputs.
- [`evaluator/README.md`](./evaluator/README.md): exact raw/text comparison procedure.


## Independently converted S models (2026-11-04)

S100, S100P and S600 are independently compiled for nash-e, nash-m and nash-p. The trained source is [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch/tree/7c976664b3f3879efabeaff59c7a117e49d5f29e), Apache-2.0. Exact hashes, configuration and provenance are in the [conversion registry](conversion/converted_targets_20261104.json).

List exact IDs with `python3 -m samples.vision.lprnet.runtime.python.main --target s100 --list-models`. The new model consumes a cropped plate: BGR, linear resize to 94×24, `(pixel-127.5)/128`, float32 NCHW. Historical X5 fixtures do not establish preprocessing for this independent checkpoint.

```bash
python3 -m samples.vision.lprnet.model.download --target s100
python3 -m samples.vision.lprnet.model.prepare_input --image /path/to/plate.jpg --output /path/to/plate.dat
python3 -m samples.vision.lprnet.runtime.python.main --target s100 --test-bin /path/to/plate.dat
```

The downloader preserves `model/s100/`; an external `--model-path` also requires the target's exact `--asset-id`. Substitute s100p/s600 as appropriate. Full-output float gates, compilation, actual HBM ABI and complete OE reports passed. Board smoke, recognition accuracy, Runtime 1/2 concurrency and C++ 1/2 streams remain `not-run`. Historical performance below belongs only to the old X5 file.

<a id="historical-performance"></a>
## Reference performance

The complete source benchmark row:

| Model | Test frames | FPS | Average latency | BPU usage | ION memory |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="license"></a>
## License

The source sample and repository code follow the repository Apache-2.0 license. The LPRNet paper and upstream project remain their authors' references; model provenance and the unknown publisher checksum are recorded in the model README.
