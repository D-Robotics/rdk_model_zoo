English | [简体中文](README_cn.md)

# GoogLeNet conversion

<a id="source-model"></a>
## Source model

For inference, prepare the published X5 deployment binary
(`googlenet_224x224_nv12.bin`) through the model guide. To build a
replacement, first prepare a matching ONNX graph, checkpoint revision,
and PTQ YAML.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OpenExplorer Docker or corresponding OE package environment for
the X5 target. Prepare the graph and PTQ configuration, then use the OE
package tools `hb_mapper checker`, `hb_mapper makertbin`, `hb_perf`, and
`hrt_model_exec`. Offline
Docker images are available from the D-Robotics developer forum
([topic 35229](https://forum.d-robotics.cc/t/topic/35229)).

<a id="export"></a>
## ONNX export

Export a GoogLeNet ONNX graph with the helper's runtime contract:
RGB/NCHW input at 224×224 before NV12 packing and 1000 classification
scores as output.

<a id="calibration"></a>
## Calibration

Prepare calibration images and preprocessing values for the selected graph
and its quantization recipe before running PTQ.

<a id="compile"></a>
## Compile

A replacement build requires a matching ONNX graph and PTQ configuration. For inference, use the
published artifact route in [model/README.md](../model/README.md).

<a id="validation"></a>
## Post-conversion validation

The runtime protocol for any rebuilt model is an input tensor of
`1x3x224x224` before NV12 packing and an ImageNet-1k classification logits
output (one F32 vector squeezing to 1000 scores). A rebuilt model can be
selected with the exact contract reference and an external `--model-path`;
keep its hash/provenance separate from the published artifact. The
functional check with the published artifact:

```bash
# cwd: repository root on X5
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py --target x5 --variant googlenet
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts).

<a id="known-gaps"></a>
## Additional preparation

For inference, prepare the published X5 model with [model/download.sh](../model/README.md). To build a replacement, prepare an ONNX graph that meets the runtime contract in [Source model](#source-model), prepare matching calibration data and normalization, and configure its PTQ YAML for the X5 OE toolchain. Run the sample runtime on the resulting artifact after compiling it.
