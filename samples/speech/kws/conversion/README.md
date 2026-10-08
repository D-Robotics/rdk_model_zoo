English | [简体中文](README_cn.md)

# KWS conversion availability

<a id="source-model"></a>
## Source model

The pinned S100 source describes an MDTC keyword model from the PaddlePaddle/PaddleAudio ecosystem. Its training checkpoint and export script are not included. For inference with the published model, follow the [model guide](../model/README.md).

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and target

The published deployment is an S100 HBM. A conversion workflow needs the target's compiler environment and configuration; runtime frontend dependencies do not select or configure that toolchain. X5, S100P and S600 have no published KWS deployment.

<a id="export"></a>
## Export prerequisite

Export a model from its trained wake-word checkpoint and architecture, using the matching preprocessing definitions and export tools. Record weight and graph SHA-256, input/output names, shapes and probability semantics. Preserve the final activation when the graph already returns probabilities; the runtime does not add sigmoid.

<a id="calibration"></a>
## Calibration prerequisite

Prepare representative positive/negative recordings on a permitted calibration split. The bundled “hey snips” clip is a demonstration input. Match mono 16 kHz PCM scaling, 60000-sample truncation/padding and the fixed 80-bin fbank contract; record source IDs, frontend versions and feature hashes.

<a id="compile"></a>
## Compilation prerequisite

For S100 compilation, specify the target, feature input layout, quantization precision and final output semantics; retain compiler logs and output digest with the resulting HBM. Keep the artifact identity tied to S100.

<a id="validation"></a>
## Validation plan

Compare the floating graph with its source on held-out positive/negative data, then compare compiled model metadata and scores with the floating graph. Report score tolerances, threshold decisions, false accepts/rejects and latency separately.

<a id="artifacts"></a>
## Conversion outputs

Record weight/source identities, export command, graph contract, calibration manifest and features, compiler configuration/logs, compiled digest and floating/compiled comparison results. The published HBM download and source performance are documented in the [model](../model/README.md) and [evaluator](../evaluator/README.md) guides.

<a id="known-gaps"></a>
## Published runtime artifact

Use the published S100 HBM with the matching board SDK by following the [runtime guide](../runtime/python/README.md). A new conversion starts from the trained checkpoint, graph and calibration data described above.
