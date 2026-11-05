English | [简体中文](README_cn.md)

# PaddleOCR model conversion

## Independently rebuilt English PP-OCRv3 S profiles / 新 S 规格

The 2026-11-04 R02 profiles use the official `en_PP-OCRv3_det_infer.tar` and `en_PP-OCRv3_rec_infer.tar` source inference programs and parameters, pinned by their separate SHA-256 values in `export_v3.py` and `converted_targets_20261104.json`. Preprocessing and dictionary are independently checked against PaddleOCR commit `c833dfc9ff04a4bce5749b808c200688b4f7bd3a`. These are distinct from the historical PP-OCRv6 S100 pair described below.

```bash
# dev-server only; trained official source inference directory already present.
paddle2onnx --model_dir /path/to/en_PP-OCRv3_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file /path/to/raw-det.onnx --opset_version 11
python3 -m samples.vision.paddle_ocr.conversion.export_v3 --role det --model-dir /path/to/en_PP-OCRv3_det_infer --raw-onnx /path/to/raw-det.onnx --image /path/to/real1.jpg --image /path/to/real2.jpg --image /path/to/real3.jpg --output /path/to/fresh/det-export
python3 -m samples.vision.paddle_ocr.conversion.prepare_v3_calibration --role det --images /path/to/real-det-images --output /path/to/fresh/det-calibration --count 100
# Use rec for the recognizer, its official source directory and real word crops.
# Each accepted S100/nash-e, S100P/nash-m or S600/nash-p target has its own fresh recipe:
hb_compile -c /path/to/target-specific-config.yaml
```

Source Paddle Inference CPU versus all ONNX outputs passed on three real CTW1500 training images/word crops plus deterministic random verification at atol1e-4/rtol1e-3. SDK CPU full-output gates passed independently. Calibration uses100 distinct real CTW1500 train images (seed42) and actual annotated word crops. Both calibration and Native runtime use the same pinned BGR producer: detection fixed linear640x640, `/255` and ImageNet mean/std; recognition aspect resize to height48, maxwidth320, `(pixel/255-.5)/.5`, right zero padding. This recognizer has97 classes (blank plus the exact96-character alphabet, including its two space entries). Do not use the18710-class PP-OCRv6 dictionary.

Cached S SDK3.7.0 CPU image: hb_compile3.5.3/HMCT2.6.5/HBDK4.7.5. Each march has an independent INT16 max-percentile0.9999 PTQ/O1 build, one core, two compiler jobs. Runtime inputs are **single normalized BGR F32 NCHW FEATUREMAP** tensors; do not supply NV12, RGB or unnormalized0..255 to these new profiles. Detector output is full F32`[1,1,640,640]`; recognizer output is full F32`[1,40,97]`. Exact model names, asset pairs, IO, checksums, original NodeInfo/OE and compiler receipts are recorded after acceptance. Full Native detection→DB polygons/crops→recognition→CTC is host checked with the actual source graphs. Board smoke, accuracy, Runtime1/2 and C++1/2 remain `not-run`.

The early R01 RGB/preprocessing-mismatch attempts are retained in Workbench and never uploaded. The portable producer here is a subsequent repository entry; the actual original export/calibration producer identities remain in the run evidence. The following recipes preserve historical source deliveries.


Conversion runs on an x86 Linux host inside the matching RDK OpenExplorer
(OE) environment, never on the board. Two complete, target-separated
recipe families are provided — choose the one matching the target board:

| Target | Family | Toolchain / march | Committed recipes | Runtime contract they must reproduce |
| --- | --- | --- | --- | --- |
| X5 | PP-OCRv3 English | `hb_mapper`, `bayes-e` (OE X5 v1.2.8) | `x5/ptq_yamls/*.yaml` | packed NV12 detector input; `[1,40,97,1]` recognizer output |
| S100 | PP-OCRv6 | `hb_compile`, `nash-e` (S OpenExplore) | `s100/*_configs.yaml` | split NV12 detector input; `[1,40,18710]` F32 recognizer output; trailing `Dequantize` on the detector |

Graph files, output names, dictionaries, and input protocols are
target-specific. Do not combine the X5 detector with the S100 recognizer or
substitute one dictionary for the other. The S100 YAMLs can also be
recompiled with `march: nash-m` (S100P) or `nash-p` (S600); the published
manifest rows cover X5 (PP-OCRv3) and S100 (PP-OCRv6), so S100P/S600
artifacts must be prepared through this compile route before use.

<a id="source-model"></a>
## Source model

- X5 pair: upstream PaddleOCR PP-OCRv3 English inference models
  (`en_PP-OCRv3_det_infer`, `en_PP-OCRv3_rec_infer`). This repository
  ships **no** PP-OCRv3 exporter; obtain the exported files from the exact
  upstream release and verify their graph inputs before compiling.
- S100 pair: upstream PaddleOCR PP-OCRv6 inference models, exported with
  Paddle2ONNX opset 19 by the command under [export](#export).
- Dictionaries are part of the model contract: the X5 recognizer decodes
  against the fixed 96-character alphabet; the S100 recognizer against
  [`test_data/s100/ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt)
  (plus blank and trailing space).

<a id="directory"></a>
## Directory structure

```text
conversion/
├── s100/  # Files for s100
├── scripts/  # Files for scripts
├── x5/  # Files for x5
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain matching the target and record the image tag,
OE version, and host date.

X5 (OE X5 v1.2.8) — download, load, and start the official image:

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Or download the offline Docker image from the [D-Robotics Developer Community](https://forum.d-robotics.cc/t/topic/35229).

S (OpenExplore) — start the matching S-series OE container, then work from
`/workspace`:

```bash
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <docker-image-name> /bin/bash
```

OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>.
OE toolchain online manual: <https://toolchain.d-robotics.cc/>.

`hb_mapper`, `hb_compile`, and `hrt_model_exec` are supplied by those
environments. The committed S100 recipes use `nash-e`; switch the `march`
field to `nash-m` (S100P) or `nash-p` (S600) to compile for the other
S-series SoCs — the output prefixes stay unchanged.

<a id="export"></a>
## Export

S100 PP-OCRv6 ONNX export (upstream PaddlePaddle; network access to clone
and install happens here, on the host, before any OE step):

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python3 -m pip install -e .
python3 -m pip install paddle2onnx onnxruntime

# outputs: det_onnx/model_detv6.onnx and rec_onnx/model_recv6.onnx
paddle2onnx --model_dir ./inference/PP-OCRv6_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model_detv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

paddle2onnx --model_dir ./inference/PP-OCRv6_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model_recv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

Copy the results to the paths the S100 YAMLs expect (or edit only the
documented `onnx_model` field): `conversion/onnx/model_detv6.onnx` and
`conversion/onnx/model_recv6.onnx`.

X5 PP-OCRv3: there is no repository-owned exporter. The recipes expect
files named `en_PP-OCRv3_det_infer.onnx` and `en_PP-OCRv3_rec_infer.onnx`
from the exact PP-OCRv3 release; this sample does not present a guessed
command as a supported X5 export recipe.

<a id="calibration"></a>
## Calibration

A suitable calibration corpus for the recognizers is the
[ICDAR2019-LSVT dataset](https://ai.baidu.com/broad/introduction?dataset=lsvt)
(450,000 Chinese street-view images: 50,000 fully annotated with bbox +
text, 400,000 weakly annotated with text only;
[download](https://ai.baidu.com/broad/download?dataset=lsvt)). Any
representative image set from the deployment domain works.

The checked-in helper turns a directory of BGR images into the input
tensors the committed recipes expect (no conversion, no network). Run from
`samples/vision/paddle_ocr/conversion`:

```bash
# X5 hb_mapper: two separate empty directories so detector and recognizer
# tensors cannot mix; point the det/rec YAML cal_data_dir at them first.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./x5/calibration_data_detector \
  --target x5 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops --output ./x5/calibration_data_recognizer \
  --target x5 --stage recognizer

# S100 hb_compile: these outputs are the ../calibration_data and
# ../calibration_data_rec_new/cropped_images_npy values in the committed YAMLs.
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./calibration_data \
  --target s100 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops \
  --output ./calibration_data_rec_new/cropped_images_npy \
  --target s100 --stage recognizer
```

Detector tensors are resized to 640×640 RGB float32; recognizer tensors to
320×48 RGB float32 in `[0,1]`, matching the runtime's `no_preprocess`
input. The X5 output extension is `.rgbchw`, the S100 extension `.npy`.
Use a separate empty output directory for each stage. Use representative images/crops from the model's
training domain; this helper does not determine an accuracy or a required
sample count.

The S recipes keep the source convention of two dedicated calibration
producers (`get_det_calibration_data.py` / `get_rec_calbration_data.py` in
the source delivery); the helper above reproduces their tensor formats
with explicit paths.

<a id="compile"></a>
## Compile

X5 PP-OCRv3, inside the OE X5 v1.2.8 environment from `conversion/x5`
after placing the two ONNX files and calibration directories there
(outputs: `model_output/*.bin`; success: `hb_mapper` exits 0):

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_rec_infer.onnx

hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_rec_config.yaml
```

Emitted filenames:

```text
model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

The detector YAML retains the observed RGB mean/scales, `nv12` runtime
input, `bayes-e`, and O3 compiler level; the recognizer YAML retains
`featuremap` NCHW/no-preprocess and the three observed `p2o.Softmax.*`
int16 node mappings.

S100 PP-OCRv6, inside the S OpenExplore environment from
`conversion/s100` (outputs: `model_output/*.hbm`; success: `hb_compile`
exits 0):

```bash
hb_compile -c paddleocr_det_configs.yaml
hb_compile -c paddleocr_rec_configs.yaml
```

Emitted filenames:

```text
model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

The detector recipe keeps the trailing `Dequantize` node — do **not** set
`remove_node_type: "Dequantize"`: the runtime reads `fetch_name_0` as an
F32 probability map and thresholds it directly, so removing the node
changes the contract. The recognizer keeps the three `p2o.Softmax.*`
int16 node mappings and `set_all_nodes_int16`; its class count must stay
aligned with the checked-in dictionary plus blank and trailing space.

To compile the S recipes for the other S-series SoCs, switch `march` in
the YAMLs:

| Target platform | `march` | Output prefix unchanged |
| --- | --- | --- |
| RDK S100 | `nash-e` | `PP-OCRv6_*-deploy_*` |
| RDK S100P | `nash-m` | `PP-OCRv6_*-deploy_*` |
| RDK S600 | `nash-p` | `PP-OCRv6_*-deploy_*` |

Per-platform latency/FPS is obtained with:

```bash
hrt_model_exec perf --model_file model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec perf --model_file model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

<a id="validation"></a>
## Validation

Inspect each generated artifact before board use (record the complete
output):

```bash
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
```

Compare tensor names, shapes, dtypes, and output names with the selected
pair's contract (see [stage I/O](../runtime/python/README.md#stage-io));
register only an artifact whose exact qualified reference and metadata
match. Then confirm on the matching board with the runtime and
the bundled fixture.

<a id="artifacts"></a>
## Artifacts

| Stage | Artifact to retain |
| --- | --- |
| export | the two `.onnx` files, exporter command/version (S100), upstream release identity (X5) |
| calibration | the calibration directories, their image source, and the helper invocation |
| checker | `hb_mapper checker` command and log |
| compile | the YAML used, OE version/image tag, `model_output/` listing |
| deployment | the four manifest filenames listed under [compile](#compile) |
| validation | `hrt_model_exec model_info` output, board run command, predictions |

<a id="known-gaps"></a>
## Additional preparation

- No repository-owned PP-OCRv3 (X5) exporter; obtain the ONNX files from
  the exact upstream release and record the release identity used.
- The original calibration corpora are not checked in; the helper prepares
  tensors but does not fix a corpus or sample count, so regenerated
  artifacts carry their own identity, distinct from the published ones.
- `march` values other than the committed ones are unclaimed experiments.
