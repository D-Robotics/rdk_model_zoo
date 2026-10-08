# MODNet conversion

<a id="source-model"></a>
## Source model

Obtain a trained checkpoint and export ONNX using the official [MODNet project](https://github.com/ZHKKKe/MODNet). Paper: [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961). Prepare the checkpoint, exporter, PTQ configuration and calibration data, then compile with the X5 toolchain.

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

There is no `onnx_export` script or pinned checkpoint/export configuration. An external owner must provide an ONNX graph with float32 RGB NCHW input `(1,3,512,512)` and float32 matte output `(1,1,512,512)`.

<a id="calibration"></a>
## Calibration

There is no PTQ YAML or calibration producer. `test_data/person.jpg` is an inference fixture, not a representative calibration set.

<a id="compile"></a>
## Compile

Prepare `modnet.yaml` in the X5 OpenExplorer environment. Set its `onnx_model` field to the exported ONNX, then configure representative calibration data and the output prefix. This YAML is an external input; run the following commands once it is prepared:

```bash
hb_mapper checker --config modnet.yaml
hb_mapper makertbin --config modnet.yaml
```

<a id="validation"></a>
## Post-conversion validation

Inspect the compiled model performance with `hb_perf` in the OE environment:

```bash
hb_perf model_perf \
    --model ./modnet_512x512_rgb.bin \
    --input-shape input 1x3x512x512
```

Copy the model to X5 and measure a single-thread run on the board:

```bash
hrt_model_exec perf \
    --model_file ./modnet_512x512_rgb.bin \
    --thread_num 1
```

Input is float32 RGB NCHW `(1,3,512,512)`, normalized with `(pixel - 127.5) / 127.5` to [-1,1]. Output is float32 `(1,1,512,512)` alpha matte in [0,1]. Check runtime metadata and compare matte arrays with the evaluator.

<a id="artifacts"></a>
## Artifacts

The only manifest row is the manual `x5:modnet:modnet_512x512_rgb.bin`, expected at `../model/modnet_512x512_rgb.bin`. ONNX, checkpoint, calibration data, YAML, logs, and compiled model are external inputs.

<a id="known-gaps"></a>
## Additional preparation

- `onnx_export/` and `ptq_yamls/` are mentioned by the source README but not included.
- Checkpoint version, export arguments, calibration dataset, PTQ settings, and compiler output naming are unknown.
- The manifest has no URL or publisher SHA-256.
