# KWS conversion availability

English | [简体中文](README_cn.md)

<a id="source-model"></a>
## Source model

The source identifies an MDTC keyword model from the PaddlePaddle/PaddleAudio ecosystem, but includes no training checkpoint, export script or checkpoint digest. Its conversion page (historical `../../../../platforms/s/samples/speech/kws/conversion/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) was a placeholder. This directory makes the missing prerequisites explicit; it is not a conversion recipe.

<a id="toolchain-targets"></a>
## Toolchain and target

Only the compiled S100 HBM is published. No validated compiler version, model compiler configuration or adaptation for X5/S100P/S600 is recorded here. Runtime frontend versions do not establish compiler compatibility. Use the [published model](../model/README.md) for the existing runtime; do not substitute a compiler command copied from an image sample.

<a id="export"></a>
## Export prerequisite

A reproducible export first needs the exact trained wake-word checkpoint and architecture, preprocessing definitions, export tool versions and resulting graph I/O. Record weight and graph SHA-256, input/output names, shapes and probability semantics. Preserve the final activation if the exported output is already a probability; the canonical runtime does not add sigmoid.

<a id="calibration"></a>
## Calibration prerequisite

Representative positive/negative recordings and a permitted, separate calibration split are not supplied. The single bundled “hey snips” clip is not sufficient calibration or accuracy evidence. Match mono 16 kHz PCM scaling, 60000-sample truncation/padding and the fixed 80-bin fbank contract; record randomization, source IDs, frontend versions and feature hashes.

<a id="compile"></a>
## Compilation prerequisite

No executable compiler command is supplied because the necessary graph, toolchain version, target configuration and calibration set are absent. A future S100 recipe must specify the target, feature input layout, quantization precision and final output semantics, then retain compiler logs and output digest. Never rename another target's HBM or publish an unverified hash.

<a id="validation"></a>
## Validation plan

Validate the floating graph against its source on held-out positive/negative data, then compare compiled model metadata and scores against that graph. Keep score tolerances, threshold decisions, false accepts/rejects and latency separate. Host PaddleAudio parity only validates feature computation; it proves neither export correctness nor board behavior.

<a id="artifacts"></a>
## Expected future deliverables

A complete recipe must provide weight/source identities, export command, graph contract, calibration manifest and features, effective compiler configuration/logs, compiled digest and independent floating/compiled comparison report. None is claimed completed by this document. Existing HBM download and source performance remain available through the sample's model/evaluator guides.

<a id="known-gaps"></a>
## Known gaps

Checkpoint/export/calibration/compiler inputs are missing from the pinned source, and no new OE compile or board result exists. The canonical sample preserves this boundary instead of inventing reproducibility. Supply those actual inputs before attempting a new conversion; the present runtime remains usable with the published artifact subject to its SDK checks.
