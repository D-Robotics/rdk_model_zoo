# MobileNetV2 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. The shipped
`mobilenetv2_config.yaml` targets **S100** (`march: "nash-e"`); the S600
build is produced from the same source ONNX with the same quantization
configuration — only `march` changes to `nash-p`.

<a id="source-model"></a>
## Source model

MobileNetV2 ([paper](https://arxiv.org/abs/1801.04381),
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)).
The upstream flow exports the ONNX model with `timm`. Before running the
commands below, obtain the operator-supplied export workspace containing
`runtime/python/get_mobilenetv2_onnx.py`,
`runtime/python/timm2onnx_local.py`,
`runtime/python/get_calibration_data.py`,
`runtime/python/x86_inference.py` and
`runtime/python/s100_inference.py`. Run those commands from that workspace.
The repository provides the quantization YAML
(`mobilenetv2_config.yaml`) and bundled test image; use the YAML paths as
configured.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── mobilenetv2_config.yaml  # Configuration
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain release matching the target board. Authoritative
references:
[RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).
Targets: X5 compiles with `hb_mapper` using march `bayes-e`; S100 uses
`hb_compile` with `nash-e`, S600 with `nash-p`. Mount the repository at
`/workspace` in the container with enough shared memory
(`--shm-size=15g`).

<a id="export"></a>
## Export

The upstream flow uses `timm` (PyTorch Image Models). Install the
dependencies, log in to Hugging Face (the script pulls
`timm/mobilenetv2_100.ra_in1k`), and export:

```bash
# cwd: the export workspace (provides runtime/python/get_mobilenetv2_onnx.py)
pip install timm onnx
huggingface-cli login
python runtime/python/get_mobilenetv2_onnx.py
```

If you cannot configure a proxy, download the model manually from
[timm/mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k)
and convert locally:

```bash
# cwd: the export workspace (provides runtime/python/timm2onnx_local.py)
python runtime/python/timm2onnx_local.py
```

After exporting, the script prints the model metadata:

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv2_100.onnx
Total number of parameters in the model: 3487818
```

Both exporters run from the export workspace described under
[Source model](#source-model) and write `mobilenetv2_100.onnx`; place the
resulting ONNX next to this directory's YAML before compiling.

<a id="calibration"></a>
## Calibration

The model uses the [ImageNet](https://image-net.org/) ILSVRC2012 dataset
(training ~1.2 million / validation 50,000 / test 100,000 images across
1000 classes). The original flow generated the calibration set from 100
validation images (`ILSVRC2012_val_*.JPEG`) into a float32 directory:

```text
imagenet/
├── calibration_data/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...  (100 images)
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── val.txt
```

```bash
# cwd: the export workspace (provides runtime/python/get_calibration_data.py)
python runtime/python/get_calibration_data.py
```

The shipped YAML consumes float32 data at `../calibration_data_bgr`
(BGR order, `mean_value: 103.53 116.28 123.675`, `scale_value: 0.017429
0.017507 0.017124`). When regenerating, record the exact image list and
preprocessing so the output can be compared with the published artifact.

<a id="compile"></a>
## Compile

Quick-verify the ONNX model before full compilation:

```bash
hb_compile --model mobilenetv2_100.onnx --march nash-e
```

Run quantization compilation with the calibration dataset using the
reference YAML:

```bash
hb_compile --config conversion/mobilenetv2_config.yaml
```

After compilation the deployment file is written to
`model_output/mobilenetv2_224x224_nv12.hbm` (per the shipped YAML's
`working_dir`/prefix). S600 changes only the march (`nash-p`). Before
rebuilding, compare the target, input metadata, output shape/dtype, and
numerical results.

| Config | Target | Command (inside the OE container) |
| --- | --- | --- |
| `mobilenetv2_config.yaml` | s100 | `hb_compile --config mobilenetv2_config.yaml` |

<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the sample runtime that the contract holds: S100/S600 expose Y
`[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
output semantics are post-softmax probabilities. The original recipe also
documents two workspace scripts: `runtime/python/x86_inference.py` for
ONNX/HBIR/HBM inference on x86 with optional val-dataset accuracy
validation, and `runtime/python/s100_inference.py` for on-board HBM
inference. Run them from the export workspace:

```bash
# cwd: the export workspace
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  -i test_data/zebra_cls.jpg
```

```bash
# cwd: the export workspace
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  --validate \
  -d ../../../imagenet/val \
  -l ../../../imagenet/val.txt
```

Published quantization record of the original S build (cosine similarity
after quantization):

```text
+------------+-------------------+------------------+
| TensorName | Calibrated Cosine | Quantized Cosine |
+------------+-------------------+------------------+
| output     | 0.993383          | 0.988877         |
+------------+-------------------+------------------+
```

Toolchain performance reference of the original S build:

```text
FPS (1 core): 4968.89
Latency: 0.2 ms (201.3 us)
BPU conv original OPs per run: 601,548,544
```

<a id="artifacts"></a>
## Kept material

- `mobilenetv2_config.yaml`

<a id="known-gaps"></a>
## Additional preparation

Run the export, calibration and x86-inference commands from the operator-supplied export workspace containing the helper scripts listed under [Source model](#source-model). For X5, use the X5 OE flow with `hb_mapper`, march `bayes-e`, and a matching X5 quantization configuration. The S-series YAML and recipe remain the commands shown above.
