English | [简体中文](./README_cn.md)

# Conversion — CLIP X5

<a id="source-model"></a>
## Source Model

The published pair has an X5 BPU image encoder (`img_encoder.bin`) and a CPU ONNX text encoder (`text_encoder.onnx`). The fixed source does not include an upstream checkpoint version, export script, conversion YAML, or calibration dataset. It preserves the original CLIP BPE vocabulary and model protocol; no new conversion claim is made here.

<a id="toolchain-targets"></a>
## Toolchain & Targets

The source notes OpenExplorer Docker or the corresponding OE package compilation environment for regenerating the image `.bin`. It does not record a compiler version, `march` configuration, text ONNX export version, or a target-specific YAML. The published target is X5 only.

| Target | march | OE version | Config |
| --- | --- | --- | --- |
| x5 | not recorded in source | not recorded in source | no source YAML; use OpenExplorer/OE package environment |

Resource pointer preserved from the source conversion notes: the D-Robotics developer forum offline-image discussion at <https://forum.d-robotics.cc/t/topic/35229>. This is a resource pointer, not a conversion result.

<a id="export"></a>
## Export (ONNX)

No export command or source checkpoint is provided. The text encoder remains the published ONNX asset and the image encoder remains the published `.bin`; a generic exporter command would exceed the source evidence. Known deployed contracts are:

| Encoder | Input | Output |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 tokens `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## Calibration

No calibration dataset, sample count, quantization configuration, or calibration script is present in the source. Calibration is therefore not reproducible from this directory and no calibration command is asserted.

<a id="compile"></a>
## Compile

No conversion YAML or compile command is published for either encoder. The model downloader prepares the already published pair; it does not compile it. Do not treat a generic OpenExplorer invocation as a verified recipe.

<a id="validation"></a>
## Post-Conversion Validation

The maintained validation path runs the published pair through `runtime/python/main.py`, computes cosine scores for the prompts, and writes a separate visualization. No conversion or board validation was run in this migration: status `not-run`.

<a id="artifacts"></a>
## Artifacts

| Artifact | Target | Lands at |
| --- | --- | --- |
| `img_encoder.bin` | x5 image BPU | `samples/vision/clip/model/` |
| `text_encoder.onnx` | x5 text CPU ONNX | `samples/vision/clip/model/` |

<a id="known-gaps"></a>
## Known Gaps

- No source checkpoint, image ONNX export script, text ONNX export script, conversion YAML, compiler/version matrix, or calibration dataset is provided.
- The published `.bin` and `.onnx` files can be prepared from the manifest, but the conversion process cannot be reproduced from this sample.
- HBM/BPU and ONNX artifact SHA-256 values are `sha256: null (unknown)` in the active manifest.
- Conversion and board smoke validation are not-run.

## License

Conversion documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0. The source pair retains its publication provenance.
