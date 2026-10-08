English | [简体中文](README_cn.md)

# Model preparation

<a id="artifacts"></a>
## Published artifact

| Target | Exact asset ID | Path under this directory | Publisher SHA-256 |
| --- | --- | --- | --- |
| S100 | `s:depth_anything_v2:s100/depth_any.hbm` | `s100/depth_any.hbm` | unknown (`null`) |

The S manifest is authoritative
for the published filename and URL. The source documentation also names S100P,
but the manifest carries no separate S100P asset. S100P, S600 and
X5 are refused, even if an external path is supplied. `auto` selects the sole
S100 contract; real execution still checks local identity before SDK loading.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── download_model.sh  # Shell command
```

<a id="preparation"></a>
## Explicit preparation

From repository root:

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
bash samples/vision/depth_anything_v2/model/download.sh --target s100
```

The first command only reads the manifest; the second downloads the selected
artifact. `download_model.sh` forwards the same flags and takes the explicit
`--target` argument (not positional SoC names). Inference never downloads a model
or installs dependencies.

<a id="accompanying-files"></a>
## Accompanying files

This model needs no class labels. The bundled `../test_data/furseal.jpg` and six
figures are visualization examples. Prepare calibration images and ground truth separately.
The HBM, source weights and ONNX are not bundled. See the
[conversion guide](../conversion/README.md) for missing reproduction prerequisites.
A similarly named upstream checkpoint is not guaranteed to match this artifact.

<a id="local-paths"></a>
## Paths and external copies

Default runtime model paths are resolved relative to this directory, independent
of current directory. The shell helpers first change to repository root, which
is the base for user-provided relative paths. Direct Python invocation instead
uses the caller's current directory for user paths.

Downloader `--output-dir` changes only the destination; it preserves the `s100/`
subdirectory and does not update runtime defaults. For an external copy:

```bash
bash samples/vision/depth_anything_v2/model/download.sh --target s100 \
  --output-dir /work/depth-models
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 \
  --asset-id s:depth_anything_v2:s100/depth_any.hbm \
  --model-path /work/depth-models/s100/depth_any.hbm \
  --output /work/depth-results/external-copy
```

An explicit model path requires the exact asset ID. That reference selects a
contract, not proof that arbitrary file bytes are the published HBM. The runtime
records the observed file digest and validates IO metadata. Custom artifacts
with different normalization, geometry or semantics require a new binding; a
filename change cannot make them compatible.

<a id="formats-checksums"></a>
## Format, checksums and provenance

HBM is the source heterogeneous model format. Expected public IO is float32 RGB
NCHW `[1,3,518,686]` and float32 depth `[1,518,686]`. Internal int16
quantization does not change the float public output contract. Tensor metadata
is validated at load; a binding failure must be investigated, not silenced by
reshaping an incompatible tensor.

Publisher SHA-256 is unknown. Downloader/runtime compute a local digest to bind
subsequent evidence, but cannot authenticate publisher identity. Keep URL,
observed hash and runtime version together with any board results.
