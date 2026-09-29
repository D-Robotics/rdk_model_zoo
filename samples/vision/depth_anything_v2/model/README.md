[English](README.md) | [简体中文](README_cn.md)

# Model preparation

<a id="artifacts"></a>
## Published artifact

| Target | Exact asset ID | Path under this directory | Publisher SHA-256 |
| --- | --- | --- | --- |
| S100 | `s:depth_anything_v2:s100/depth_any.hbm` | `s100/depth_any.hbm` | unknown (`null`) |

The [S manifest](../../../../platforms/s/docs/release/models.yaml) is authoritative
for the published filename and URL. Source prose also names S100P, but the
manifest has no separate S100P asset or compatibility evidence. S100P, S600 and
X5 are refused, even if an external path is supplied. `auto` selects the sole
S100 contract; real execution still checks local identity before SDK loading.

<a id="preparation"></a>
## Explicit preparation

From repository root:

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
bash samples/vision/depth_anything_v2/model/download.sh --target s100
```

The first command only reads the manifest; the second downloads the selected
artifact. `download_model.sh` forwards the same flags; historical positional SoC
arguments are replaced by explicit `--target`. Inference never downloads a model
or installs dependencies. No model was downloaded in this host migration.

<a id="accompanying-files"></a>
## Accompanying files

This model needs no class labels. The bundled `../test_data/furseal.jpg` and six
figures are byte-preserved source files, not calibration images or ground truth.
The HBM, source weights and ONNX are not bundled. See the
[conversion guide](../conversion/README.md) for missing reproduction prerequisites.
No claim that a similarly named upstream checkpoint matches this artifact is made.

<a id="local-paths"></a>
## Paths and external copies

Default runtime model paths are resolved relative to this directory, independent
of current directory. The shell wrappers first change to repository root, which
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
NCHW `[1,3,518,686]` and float32 depth `[1,518,686]`. Source internal int16
quantization does not establish integer public output. Actual HBM metadata has
not been observed in this migration; binding failure must be investigated, not
silenced by reshaping an incompatible tensor.

Publisher SHA-256 is unknown. Downloader/runtime compute a local digest to bind
subsequent evidence, but cannot independently authenticate publisher identity.
A nonempty file, matching shape or successful download does not prove model
accuracy or board compatibility. Keep URL, observed hash and runtime version
with later board results; current board verification remains `not-run`.
