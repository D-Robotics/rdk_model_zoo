English | [简体中文](README_cn.md)

# Conversion — CLIP X5

<a id="source-model"></a>
## Source Model

The published pair has an X5 BPU image encoder (`img_encoder.bin`) and a CPU ONNX text encoder (`text_encoder.onnx`). No upstream checkpoint version, export script, conversion YAML, or calibration dataset is included. The original CLIP BPE vocabulary and model protocol are preserved.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="toolchain-targets"></a>
## Toolchain & Targets

Regenerating the image `.bin` requires an OpenExplorer Docker or corresponding OE package compilation environment. No compiler version, `march` configuration, text ONNX export version, or target-specific YAML is recorded. The published target is X5 only.

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| x5 | not recorded | not recorded | no YAML included; use an OpenExplorer/OE package environment |

The D-Robotics developer forum carries an offline-image discussion at <https://forum.d-robotics.cc/t/topic/35229>.

<a id="export"></a>
## Export (ONNX)

No export command or source checkpoint is provided. The text encoder remains the published ONNX asset and the image encoder remains the published `.bin`. Known deployed contracts are:

| Encoder | Input | Output |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 tokens `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## Calibration

No calibration dataset, sample count, quantization configuration, or calibration script is included. Calibration is therefore not reproducible from this directory.

<a id="compile"></a>
## Compile

No conversion YAML or compile command is published for either encoder. The model downloader prepares the already published pair; it does not compile it.

<a id="validation"></a>
## Post-Conversion Validation

The validation path runs the published pair through `runtime/python/main.py`, computes cosine scores for the prompts, and writes a separate visualization; see the [Python runtime](../runtime/python/README.md) for the command.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `img_encoder.bin` | x5 image BPU | `samples/vision/clip/model/` |
| `text_encoder.onnx` | x5 text CPU ONNX | `samples/vision/clip/model/` |

<a id="known-gaps"></a>
## Additional preparation

- No source checkpoint, image ONNX export script, text ONNX export script, conversion YAML, compiler/version matrix, or calibration dataset is provided.
- The published `.bin` and `.onnx` files can be prepared from the manifest, but the conversion process cannot be reproduced from this sample.
- HBM/BPU and ONNX artifact SHA-256 values are `sha256: null (unknown)` in the active manifest.

## License

Conversion documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0. The published pair retains its publication provenance.
