English | [简体中文](README_cn.md)

# ASR model conversion

<a id="source-model"></a>
## Source model
The sample runs a Wav2Vec2 ASR model. The published S100/S600 HBM artifacts and the fixed vocabulary are available through the [model guide](../model/README.md). This directory ships no checkpoint-to-HBM script: bring a Wav2Vec2 checkpoint, an exporter and the per-target OE configuration as external inputs, and record the checkpoint URL/revision, architecture and training configuration.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets
The active manifest publishes separate S100 and S600 HBM files. Use the OE/compiler release and target configuration matching each artifact, and record the versions used. Do not treat S100P as S100, or rename one HBM to target another board.

<a id="export"></a>
## Export
Export takes the exact checkpoint and architecture/configuration, vocabulary/token ordering, preprocessing contract, exporter versions and ONNX opset as its inputs. Expected runtime input is float32 `[1,30000]`; expected output is `[1,T,3503]` logits. `T` is observed from the actual export/model, not taken from an unrelated Wav2Vec2 model. Verify blank ID 0 and the pinned vocabulary before comparing text.

<a id="calibration"></a>
## Calibration
Prepare a calibration set with a recorded dataset/license/split. Calibration data follows the runtime's 16 kHz mono, independent-window, variance-plus-1e-5, normalize-before-padding contract. The bundled short recording is a functional example, not a representative calibration or accuracy dataset. Record the resampler choice and final-chunk handling.

<a id="compile"></a>
## Compile
Compile with the matching target recipe in the OE environment. Inspect compiler input/output properties and record all model/config/data/toolchain hashes with the run. Integer output requires actual SCALE metadata supported by the runtime; never manufacture scale values from a filename.

<a id="validation"></a>
## Validation
Validate source and exported logits on the same prepared waveform; then compare compiled-model logits under their actual quantization metadata. Report numeric tolerances and decoding mode separately. CTC and legacy text differ even for identical logits. Dataset CER needs labeled transcripts and a declared text normalization policy.

<a id="artifacts"></a>
## Artifacts
A complete conversion handoff should contain checkpoint identity/license, ONNX and hash, ordered vocabulary/hash, frontend configuration, calibration manifest, per-target compiler configuration/logs, HBM/hash and validation report.

<a id="known-gaps"></a>
## Additional preparation
Obtain the checkpoint/export recipe, calibration corpus, compiler versions/configs and source/export/HBM validation as external inputs; this directory provides none of them. Use the published HBM with the inference workflow documented in the model guide.
