# MobileNetV1 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source branches shipped and records the gaps
honestly; it does not invent a configuration that could produce a
different artifact.

<a id="source-model"></a>
## Source model

MobileNetV1 ([paper](https://arxiv.org/abs/1704.04861),
[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)),
fixed NCHW input `[1,3,224,224]`, ImageNet-1k class count. The X5
source shipped no export script (its `conversion/` was README-only);
the S conversion notes identify the MobileNet-Caffe source model
converted with the S100 OE toolchain.

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

Not reproducible from this repository (known gap): neither source branch
shipped an ONNX exporter or weight provenance for the published MobileNetV1
artifacts. Regenerating the `.onnx` means authoring the export against the
upstream model first; until that exists, this directory documents the
deployment artifacts without reproducing them.
<a id="calibration"></a>
## Calibration

Not reproducible from this repository (known gap): no calibration set,
preprocessing record, or PTQ configuration shipped for MobileNetV1 on
either branch.
<a id="compile"></a>
## Compile

No OE configuration shipped for MobileNetV1 on either source branch (known
gap). The published `.bin`/`.hbm` artifacts were built outside this
repository's material; regenerating them means authoring an OE
configuration whose input protocol matches the runtime contract (packed
NV12 on X5, split Y/UV on S100/S600) and recording the calibration data.
<a id="validation"></a>
## Validation

For a regenerated artifact, run `hb_perf` and `hrt_model_exec` per the OE
manual and keep the complete output; then confirm on the matching board
with the canonical runtime that the contract holds: X5 exposes one packed
NV12 input and an F32 `[1,1000,1,1]` output; S100/S600 expose Y
`[1,224,224,1]`, UV `[1,112,112,2]`, and an F32 `[1,1000]` output; the
output semantics are post-softmax probabilities. Validation status for regenerated
artifacts in this repository: **not-run**.

<a id="artifacts"></a>
## Kept material

- (none — README-only record)

<a id="known-gaps"></a>
## Known gaps

- No exporter, OE configuration, or calibration set shipped on either
branch; the conversion is documented, not reproduced.
