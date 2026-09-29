# LPRNet conversion

<a id="source-model"></a>
## Source model

The fixed source provides the X5 deployment artifact and protocol notes, but no in-repository ONNX export script, checkpoint, calibration set, or PTQ YAML. The source conversion README uses an external OE package and a user-supplied `your_lprnet_config.yaml`; that placeholder is not a reproducible file in this repository.

<a id="toolchain-targets"></a>
## Toolchain and targets

The source names an RDK X5 OpenExplorer environment with `hb_mapper` and `hrt_model_exec`. Only X5 is supported. No S march or S artifact exists for LPRNet.

<a id="export"></a>
## Export

No export recipe is included by the fixed source. An external user-owned ONNX/checkpoint pipeline must produce a graph whose deployment contract is `input float32 NCHW (1,3,24,94)` and `output float32 (1,68,18,1)` — the released `lpr.bin` native logits; CTC decoding consumes the `(68,18)` payload after singleton removal. This migration did not invent or run that pipeline.

<a id="calibration"></a>
## Calibration

No calibration dataset or source calibration producer is included. The checked-in `test_input.dat` is a runtime input fixture, not a calibration set. It must not be described as image data or reused as a representative calibration dataset.

<a id="compile"></a>
## Compile

After the user supplies the missing ONNX and YAML in this conversion directory, the source documents these OE commands; they are conditional templates, not a runnable recipe here:

```bash
# cwd: samples/vision/lprnet/conversion; input: external your_lprnet_config.yaml
hb_mapper checker --model-type onnx --config ./your_lprnet_config.yaml
hb_mapper makertbin --model-type onnx --config ./your_lprnet_config.yaml
# success requires the OE tools to report success and create an X5 .bin artifact
```

The exact output filename, calibration options, quantization settings, and source checkpoint remain unknown.

<a id="validation"></a>
## Post-conversion validation

Inspect the generated model with `hrt_model_exec model_info --model_file ./lpr.bin`, then compare its metadata to the runtime binding. No conversion or `hrt_model_exec` metadata inspection was run in this migration; the published artifact itself passed same-board source/unified comparison on one X5 8GB and one X5 4GB (2026-09-24, see the evaluator README), which validates runtime parity but not conversion reproducibility.

<a id="artifacts"></a>
## Artifacts

The only manifest-backed deployment artifact is `x5:lprnet:lpr.bin`; the runtime expects it at `../model/lpr.bin`. ONNX, checkpoint, calibration, and compiler logs are external and are not checked in.

<a id="known-gaps"></a>
## Known gaps

- No source export script, checkpoint version, calibration producer, PTQ YAML, or reproducible OE package is provided.
- `your_lprnet_config.yaml` is a source placeholder, not a repository file.
- Publisher SHA-256 is unknown; conversion is `not-run`. Board parity of the published artifact is recorded in the evaluator README and does not extend to rebuilding.
