English | [简体中文](README_cn.md)

# Conversion — FCOS

<a id="source-model"></a>

## Source Model

- Framework: the fixed X5 source records FCOS EfficientNet-B0/B2/B3 deployment artifacts; it does not include training checkpoints or an export script.
- Weights: source snapshot `platforms/x5` at `ac115717197920355fc390bb04299b20e6436864`.
- Correspondence: the three released artifacts are 512, 768, and 896 FCOS variants. The source README and screenshots do not identify a checkpoint release or training commit.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain & Targets

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| x5 / B0 512 | bayes-e | not recorded in source | no YAML in source |
| x5 / B2 768 | bayes-e | not recorded in source | no YAML in source |
| x5 / B3 896 | bayes-e | not recorded in source | no YAML in source |

The three `hb_perf` graphs below show the deployed dataflow for the published variants. Each graph records the deployed dataflow of one published variant: the packed NV12 input is converted by a BPU `NV12TOYUV444` node into `YUV444, NHWC, INT8`, executes as `torch-jit-export_subgraph_0` on BPU, and emits fifteen INT32 outputs — five stride levels × three branches (80-channel classification scores, 4-channel box regression, 1-channel center-ness). This is the source output protocol: 5 classification, 5 box-regression, and 5 center-ness outputs, which the Python runtime reorders by their fixed tensor shapes before decoding; decode semantics are documented in the [runtime guide](../runtime/python/README.md).

Reference deployment graphs:

![FCOS EfficientNet-B0 hb_perf](./fcos_efficientnetb0_512x512_nv12.png)

FCOS EfficientNet-B0, 512×512 input: score/box/center-ness heads on 64×64, 32×32, 16×16, 8×8 and 4×4 levels (strides 8–128).

![FCOS EfficientNet-B2 hb_perf](./fcos_efficientnetb2_768x768_nv12.png)

FCOS EfficientNet-B2, 768×768 input: the same three branches on 96×96 down to 6×6 levels.

![FCOS EfficientNet-B3 hb_perf](./fcos_efficientnetb3_896x896_nv12.png)

FCOS EfficientNet-B3, 896×896 input: the same three branches on 112×112 down to 7×7 levels.

<a id="export"></a>
## ONNX export

Prepare the checkpoint and exporter for the selected B0/B2/B3 variant. Record its input preprocessing and the names, shapes and types of all fifteen output heads. Checkpoints and export scripts are external prerequisites.

<a id="calibration"></a>
## Calibration

- Dataset: not recorded; no calibration files or count are present.
- Config: no FCOS YAML is present.
- Command: no reproducible calibration command exists.

<a id="compile"></a>
## Compile

Prepare the ONNX and quantization YAML in the matching X5 OpenExplorer environment, then run `hb_mapper makertbin --model-type onnx --config your_fcos_config.yaml`. Replace `your_fcos_config.yaml` with the actual configuration path; that file is not supplied here.

<a id="validation"></a>
## Post-Conversion Validation

On a matching X5, inspect a supplied artifact with `hrt_model_exec model_info --model_file <exact-file>` and run the runtime README command. Save the command output, raw fifteen tensor metadata, and result JSON under one UTC evidence directory. The [evaluator](../evaluator/README.md#reference-results) additionally provides recorded board parity runs of the published artifacts.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | x5 | `model/` |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | x5 | `model/` |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | x5 | `model/` |

<a id="known-gaps"></a>
## Additional preparation

Custom conversion requires FCOS EfficientNet-B0/B2/B3 checkpoints, an ONNX exporter, representative calibration images, X5 quantization YAML and a matching OpenExplorer toolchain. A complete rebuild recipe is not included. To run the sample, download the published BIN; publisher SHA-256 values are unavailable.
