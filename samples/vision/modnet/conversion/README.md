# MODNet conversion

<a id="source-model"></a>
## Source model

Use the [MODNet repository](https://github.com/ZHKKKe/MODNet) to obtain a trained checkpoint and export ONNX. Prepare an ONNX model with the input/output contract below, then quantize and compile it with the target toolchain.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain and targets

The source describes RDK X5 OpenExplorer tools `hb_mapper`, `hb_perf`, and `hrt_model_exec`. Only the X5 `bayes-e` deployment asset is in the manifest. No S target or C++ conversion path is provided.

<a id="export"></a>
## Export

There is no `onnx_export` script or pinned checkpoint/export configuration. An external owner must provide an ONNX graph with float32 RGB NCHW input `(1,3,512,512)` and float32 matte output `(1,1,512,512)`. Thisigration did not execute or reconstruct that missing path.

<a id="calibration"></a>
## Calibration

There is no PTQ YAML or calibration producer. `test_data/person.jpg` is an inference fixture, not a representative calibration set.

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
## Additional preparation

- `onnx_export/` and `ptq_yamls/` are mentioned by the source README but not included.
- Checkpoint version, export arguments, calibration dataset, PTQ settings, and compiler output naming are unknown.
- The manifest has no URL or publisher SHA-256.
