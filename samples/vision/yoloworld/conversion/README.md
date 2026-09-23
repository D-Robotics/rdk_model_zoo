# YOLOWorld conversion

<a id="source-model"></a>
## Source model

The fixed X5 source provides the compiled `yolo_world.bin` protocol and the
offline embedding JSON, but no checkpoint, export script, calibration set,
Bayes-E YAML, or reproducible compiler recipe. The copied
`source/yoloworld_det.py` is retained only as provenance for the runtime math;
it is not a conversion tool.

<a id="toolchain-targets"></a>
<a id="export"></a>
## Toolchain targets and export

Target is RDK X5, input 640, image F32 NCHW plus text F32[1,32,512,1], and two
F32 output tensors. Rebuilding requires the source model/checkpoint and the
matching OpenExplorer package. Neither is published in this repository, so an
export command is intentionally not invented.

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
`bboxes` shapes before use. Run `evaluator/compare.py` on X5 for source/unified
raw and result parity. The only conversion artifact currently published is the
manifest model; the offline vocabulary is a separate required input.

<a id="known-gaps"></a>
## Known gaps

Checkpoint, export, calibration, compiler logs and publisher model hash are
unknown. Conversion status is therefore `not reproducible from this tree`; no
board conversion or accuracy result is claimed.
