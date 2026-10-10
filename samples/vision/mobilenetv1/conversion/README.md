English | [简体中文](README_cn.md)

# MobileNetV1 model conversion

For the pinned timm checkpoint workflow, use [`export.py`](export.py) and the [shared host workflow](../../../../utils/tools/mobilenet/README.md). It records the exact checkpoint, center-crop preprocessing, batch-one logits contract, and full-dataset evaluation inputs. The existing artifact commands below retain their own contracts.

Run conversion on an x86 Linux host in the RDK OpenExplore (OE)
environment. The steps below identify the graph, calibration data, and
target-specific PTQ configuration to prepare before rebuilding.

<a id="source-model"></a>
## Source model

MobileNetV1 ([paper](https://arxiv.org/abs/1704.04861),
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)),
fixed NCHW input `[1,3,224,224]`, ImageNet-1k class count. For X5,
prepare an ONNX graph from the selected MobileNetV1 checkpoint. The S
conversion notes identify the MobileNet-Caffe source model converted with
the S100 OE toolchain.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Use the OE Docker/toolchain release matching the target board and record
its image tag, OE version, and host date. Authoritative references:
[RDK S toolchain overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[D-Robotics toolchain download](https://toolchain.d-robotics.cc/).
Targets: X5 compiles with `hb_mapper` using march `bayes-e`; S100 uses
`hb_compile` with `nash-e`, S600 with `nash-p`. Mount the repository at
`/workspace` in the container with enough shared memory
(`--shm-size=15g`).

<a id="export"></a>
## Export

Export an ONNX graph from the selected upstream MobileNetV1 checkpoint
with input `[1,3,224,224]` and 1,000-class output. Record the framework,
exporter, and checkpoint revision with the graph.

<a id="calibration"></a>
## Calibration

Prepare a calibration set using the selected graph's image preprocessing,
and record the image selection, normalization, and PTQ configuration.

<a id="compile"></a>
## Compile

Create an OE configuration for the selected target and graph. Match its
input protocol to the runtime contract (packed NV12 on X5, split Y/UV on
S100/S600), and bind the calibration data prepared above.

<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the runtime that the contract holds: X5 exposes one packed
NV12 input and an F32 `[1,1000,1,1]` output; S100/S600 expose Y
`[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
output semantics are post-softmax probabilities.

<a id="artifacts"></a>
## Artifacts

For inference, use the manifest-backed artifacts in [model/README.md](../model/README.md).

<a id="known-gaps"></a>
## Additional preparation

For inference, use the manifest-backed artifacts in [model/README.md](../model/README.md). A rebuild requires a MobileNetV1 ONNX graph with the 224x224 RGB/NCHW input and 1,000-class output, a matching weight revision, an OE PTQ configuration, and calibration data using the graph's preprocessing. Use the release artifact route when those inputs are not part of your conversion workspace.
