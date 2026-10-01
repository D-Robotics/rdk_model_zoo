# ASR model conversion

English | [简体中文](README_cn.md)

<a id="source-model"></a>
## Source model
The S source identifies a Wav2Vec2 ASR model, but does not provide a complete checkpoint-to-HBM recipe. The published HBM and fixed vocabulary can be consumed through the [model guide](../model/README.md); their availability does not establish an exportable source checkpoint. No checkpoint URL, revision or training configuration has been verified here.

<a id="toolchain-targets"></a>
## Toolchain and targets
The active manifest publishes separate S100 and S600 HBM files. A reproducible conversion requires the exact compatible OE/compiler version and target configuration for each. Neither has been established from the source placeholder. Do not treat S100P as S100, or rename one HBM to target another board.

<a id="export"></a>
## Export
Export is not executable from the provided material. Required inputs are the exact checkpoint and architecture/configuration, vocabulary/token ordering, preprocessing contract, exporter versions and ONNX opset. Expected runtime input is float32 `[1,30000]`; expected output is `[1,T,3503]` logits. `T` must be observed from the actual export/model, not guessed from an unrelated Wav2Vec2 model. Verify blank ID 0 and the pinned vocabulary before comparing text.

<a id="calibration"></a>
## Calibration
No source calibration set or generation script is supplied. A future set must have a recorded dataset/license/split and use the runtime's 16 kHz mono, independent-window, variance-plus-1e-5, normalize-before-padding contract. The bundled short recording is a functional example, not a representative calibration or accuracy dataset. Resampler choice and final-chunk handling must be recorded.

<a id="compile"></a>
## Compile
There is no verified compiler command/config to run. Obtain the matching target recipe, inspect compiler input/output properties and record all model/config/data/toolchain hashes before publishing a reproducible command. Integer output requires actual SCALE metadata supported by the runtime; never manufacture scale values from a filename.

<a id="validation"></a>
## Validation
Validate source and exported logits on the same prepared waveform; then compare compiled-model logits under their actual quantization metadata. Report numeric tolerances and decoding mode separately. CTC and legacy text differ even for identical logits. Dataset CER needs labeled transcripts and a declared text normalization policy. Neither OE compilation nor real model parity has run in this migration.

<a id="artifacts"></a>
## Artifacts
A complete conversion handoff should contain checkpoint identity/license, ONNX and hash, ordered vocabulary/hash, frontend configuration, calibration manifest, per-target compiler configuration/logs, HBM/hash and validation report. This directory contains documentation only; it does not claim those missing artifacts exist.

<a id="known-gaps"></a>
## Known gaps
Missing: exact checkpoint/export recipe, calibration corpus, compiler versions/configs and real source/export/HBM validation. Historical conversion directory (historical `../../../../platforms/s/samples/speech/asr/conversion/` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) is retained for traceability. Use published HBM for the documented runtime path; do not describe download-and-run as model conversion.
