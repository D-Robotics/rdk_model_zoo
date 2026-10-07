# YOLOWorld conversion

<a id="source-model"></a>
## Source model

The X5 source delivery provides the compiled `yolo_world.bin` protocol and the
offline embedding JSON, but no checkpoint, export script, calibration set,
Bayes-E YAML, or reproducible compiler recipe. The copied
`source/yoloworld_det.py` documents the runtime math provenance;
it is not a conversion tool.

<a id="toolchain-targets"></a>
<a id="export"></a>
## Toolchain targets and export

Target is RDK X5, input 640, image F32 NCHW plus text F32[1,32,512,1], and two
F32 output tensors. Rebuilding requires the source model/checkpoint and the
matching OpenExplorer package; the offline OE Docker image is discussed on the
D-Robotics developer forum at <https://forum.d-robotics.cc/t/topic/35229>.
Neither checkpoint nor exporter is published in this repository, so no export
command is given.

<a id="calibration"></a>
<a id="compile"></a>
## Calibration and compile

No source calibration dataset or quantization configuration is present. The
published artifact is consumed as-is; do not infer INT8 scales from the `.bin`
filename. A future conversion record must capture exporter/OE versions, source
checkpoint identity, vocabulary generation, input names/shapes, output names,
calibration data and SHA-256 before claiming parity.

<a id="validation"></a>
<a id="artifacts"></a>
## Validation and artifacts

Validate observed metadata against the two inputs and `classes_score` /
`bboxes` shapes before use. Run `evaluator/compare.py` on X5 for
raw and result parity against the source implementation. The only published
conversion artifact is the manifest model; the offline vocabulary is a separate
required input.

<a id="known-gaps"></a>
## Additional preparation

Checkpoint, export, calibration, compiler logs and publisher model hash are
unknown; the conversion is therefore not reproducible from this repository.
