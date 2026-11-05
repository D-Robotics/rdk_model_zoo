English | [简体中文](README_cn.md)

# Model preparation

<a id="artifacts"></a>
## Published artifacts

| Target | Exact asset ID | Path under this directory | Publisher SHA-256 |
| --- | --- | --- | --- |
| S100 | `s:depth_anything_v2:s100/depth_any.hbm` | `s100/depth_any.hbm` | unknown (`null`) |
| S100 Small O1 | `s:depth_anything_v2:s100/depth_anything_v2_small_518x686_nashe_20261102_o1.hbm` | `s100/depth_anything_v2_small_518x686_nashe_20261102_o1.hbm` | `8b012b7eab63358a4f9fc8f70035134723c0135c09cc51a2d194ef243018af4b` |

The S manifest is authoritative
for the published filename and URL. The source documentation also names S100P,
but the manifest carries no separate S100P asset. S100P, S600 and
X5 are refused, even if an external path is supplied. `auto` selects the legacy
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
[conversion guide](../conversion/README.md) for the historical model prerequisites and the independent public Small recipe.
A similarly named upstream checkpoint is not guaranteed to match the historical artifact.

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
contract. Use the HBM matching that input/output contract. The runtime
records the observed file digest and validates IO metadata. Custom artifacts
with different normalization, geometry or semantics require a new binding; a
filename change cannot make them compatible.

<a id="formats-checksums"></a>
## Formats and checksums

HBM is the source heterogeneous model format. Expected public IO is float32 RGB
NCHW `[1,3,518,686]` and float32 depth `[1,518,686]`. Internal int16
quantization does not change the float public output contract. Tensor metadata
is validated at load; a binding failure must be investigated, not silenced by
reshaping an incompatible tensor.

For the historical `depth_any.hbm`, publisher SHA-256 is unknown. Downloader/runtime compute a local digest to bind
subsequent evidence, but cannot authenticate publisher identity. Keep URL,
observed hash and runtime version together with any board results.


## Independent Small selection

The optional `--variant small-518x686-o1` selects only the manifest row for `s100/depth_anything_v2_small_518x686_nashe_20261102_o1.hbm`. Both the downloader and runtime require that exact row to exist; a prepared exporter or recipe is not a published model. The default remains `s100/depth_any.hbm`. Public Small weights are Apache-2.0 and have a separately pinned source/checkpoint identity in the conversion guide; the old artifact's unknown encoder/hash and historical metrics stay unchanged.

The Small O1 file is 126,860,200 bytes. Its host HBM inspection verified nash-e and float32 `[1,3,518,686]` → `[1,518,686]`. See [conversion provenance](../conversion/small_518x686_o1.provenance.json) for the source checkpoint, graph, calibration, toolchain and report hashes. Board validation remains `not-run`.

```bash
bash samples/vision/depth_anything_v2/model/download.sh --target s100 --variant small-518x686-o1
```

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100p | `small-518x686-int16-20261103-o1` | model | 200 |
| s600 | `small-518x686-int16-20261103-o1` | model | 200 |

```bash
python3 samples/vision/depth_anything_v2/model/download.py --target s100p --asset-id s:depth_anything_v2:s100p/depth_anything_v2_small_518x686_nashm_20261103_o1.hbm
python3 samples/vision/depth_anything_v2/runtime/python/main.py --target s100p --variant small-518x686-int16-20261103-o1 --asset-id s:depth_anything_v2:s100p/depth_anything_v2_small_518x686_nashm_20261103_o1.hbm --model-path /path/to/depth_anything_v2_small_518x686_nashm_20261103_o1.hbm --test-img samples/vision/depth_anything_v2/test_data/furseal.jpg
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
