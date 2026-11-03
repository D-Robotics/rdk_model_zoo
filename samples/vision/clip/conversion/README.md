English | [简体中文](README_cn.md)

# Conversion — CLIP

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

The historical default pair has no matching source export recipe. The independent pinned ViT-B/32 workflow below exports both graphs. The text encoder remains the published ONNX asset and the image encoder remains the published `.bin`. Known deployed contracts are:

| Encoder | Input | Output |
| --- | --- | --- |
| Image BPU | float32 RGB NCHW `(1,3,224,224)` | float32 `(1,512)` |
| Text CPU ONNX | int32 tokens `(N,77)` | float32 `(N,512)` |

<a id="calibration"></a>
## Calibration

The historical default pair has no frozen calibration provenance. The independent workflow below prepares 200 real-image tensors with the native preprocessing.

<a id="compile"></a>
## Compile

The historical default pair has no matching conversion YAML. New per-target recipes are recorded with their converted profiles. The model downloader prepares the already published pair; it does not compile it.

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

- The historical default pair has no matching source checkpoint or conversion provenance.
- The published `.bin` and `.onnx` files can be prepared from the manifest, but the conversion process cannot be reproduced from this sample.
- The historical default pair retains its unknown publisher hashes. New pairs require exact hashes for both files.

## License

Conversion documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0. The published pair retains its publication provenance.

## Independent pinned ViT-B/32 export

The [upstream notice](UPSTREAM_NOTICE.md) fixes the official code, checkpoint,
BPE vocabulary and MIT license. Use the existing PyTorch 2.8.0 / ONNX 1.17.0 /
ONNX Runtime 1.19.2 export environment, with ftfy 6.3.1 and regex 2024.11.6
in an isolated dependency directory. No toolchain upgrade is required.

```bash
# Repository root; an exact upstream source checkout and 200 frozen photographs.
python3 samples/vision/clip/conversion/export_pair.py \
  --source-dir /path/to/CLIP-d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 \
  --checkpoint /path/to/ViT-B-32.pt --images /path/to/frozen/images \
  --opset 17 --output /path/to/new-run/export
```

This strictly loads the official checkpoint into its eager source model. Both
encoder wrappers, both ONNX graphs, native BPE tokenization and cosine matching
must pass three real photographs and seed-0 data at atol 1e-4 / rtol 1e-3.
The report records all graph, source and producer identities. The text graph
has dynamic prompt batches, int32 [prompts,77] input and float32 [prompts,512]
output. The image graph takes float32 [1,3,224,224] RGB in [0,1] and includes
the official mean/std. Its calibration directory contains 200 independent
real-image `.npy` tensors; random inputs are validation-only.

Compile only the image graph in the matching S SDK. The CPU text ONNX remains
unchanged. Inspect the final HBM ABI and original full NodeInfo/OE before
registering a complete pair. Native `--variant` selection authenticates both
files before either runtime loads. A complete download contains the BPU image
model, CPU text ONNX, BPE/tokenizer, MIT notice, host matching evidence and
separate BPU conversion / CPU host reports. There is no measured pipeline
latency or board accuracy until independent board acceptance.
