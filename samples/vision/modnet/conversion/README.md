# MODNet conversion

<a id="source-model"></a>
## Source model

The source identifies the official [MODNet repository](https://github.com/ZHKKKe/MODNet) and paper, but it does not pin a checkpoint revision or provide an ONNX exporter in the fixed source tree. The source README mentions `onnx_export/` and `ptq_yamls/`, yet those paths are absent from the audited files; they are not copied or invented here.

<a id="toolchain-targets"></a>
## Toolchain and targets

The source describes RDK X5 OpenExplorer tools `hb_mapper`, `hb_perf`, and `hrt_model_exec`. Only the X5 `bayes-e` deployment asset is in the manifest. No S target or C++ conversion path is provided.

<a id="export"></a>
## Export

There is no source `onnx_export` script or pinned checkpoint/export configuration. An external owner must provide an ONNX graph with float32 RGB NCHW input `(1,3,512,512)` and float32 matte output `(1,1,512,512)`. This migration did not execute or reconstruct that missing path.

<a id="calibration"></a>
## Calibration

There is no source PTQ YAML or calibration producer in the audited tree. `test_data/person.jpg` is an inference fixture, not a representative calibration set. No calibration data was generated.

<a id="compile"></a>
## Compile

No complete compile command can be made reproducible because the source YAML and ONNX are missing. Once the user supplies both in a toolchain environment, the source's conceptual steps are `hb_mapper checker` followed by `hb_mapper makertbin`; the exact options, calibration, output prefix, and checkpoint are unknown. The compiled output must still satisfy the model binding before use.

<a id="validation"></a>
## Post-conversion validation

Use the target toolchain's `hb_perf` and `hrt_model_exec` only after an external model and configuration exist, then inspect input/output metadata against the runtime README. No export, PTQ, compile, or board test was run here.

<a id="artifacts"></a>
## Artifacts

The only manifest row is the manual `x5:modnet:modnet_512x512_rgb.bin`, expected at `../model/modnet_512x512_rgb.bin`. ONNX, checkpoint, calibration data, YAML, logs, and compiled model are external inputs.

<a id="known-gaps"></a>
## Known gaps

- `onnx_export/` and `ptq_yamls/` are mentioned by the source README but absent from the fixed source inventory.
- Checkpoint version, export arguments, calibration dataset, PTQ settings, and compiler output naming are unknown.
- The manifest has no URL or publisher SHA-256; conversion and board validation are `not-run`.
