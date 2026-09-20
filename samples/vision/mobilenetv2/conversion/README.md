# MobileNetV2 model conversion

Conversion runs on an x86 Linux host in the RDK OpenExplore (OE)
environment; it is not a board operation. This directory keeps the
conversion material the source branches shipped and records the gaps
honestly; it does not invent a configuration that could produce a
different artifact.

<a id="source-model"></a>
## Source model

MobileNetV2 ([paper](https://arxiv.org/abs/1801.04381),
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)).
Neither source branch shipped an ONNX exporter here; the S-side
configuration consumes a `mobilenetv2.onnx` that the sources do not
reproduce.

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
shipped an ONNX exporter or weight provenance for the published MobileNetV2
artifacts. Regenerating the `.onnx` means authoring the export against the
upstream model first; until that exists, this directory documents the
deployment artifacts without reproducing them.
<a id="calibration"></a>
## Calibration

The S-side configuration points at `../calibration_data_bgr`
(float32); the calibration set is not shipped with the repository. A
regeneration must record the exact image list and preprocessing before its
output can be compared with the published artifact.
<a id="compile"></a>
## Compile

Reference configurations kept in this directory:

| Config | Target | Command (inside the OE container) |
| --- | --- | --- |
| `mobilenetv2_config.yaml` | s100 | `hb_compile --config mobilenetv2_config.yaml` |

S600 variants change only the march (`nash-p`) in the S-side YAML. The
march values above are read from the YAML files themselves (`bayes-e` X5,
`nash-e` S100). Compilation has not been re-run in this repository; a
regenerated artifact is not equivalent to the published one until target,
input metadata, output shape/dtype, and numerical results are compared.
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

- `mobilenetv2_config.yaml`

<a id="known-gaps"></a>
## Known gaps

- No ONNX exporter on either branch and no X5-side configuration; only
the S-side reference YAML is kept.
