# EfficientNet model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EfficientNet_B2_224x224_nv12.bin` | x5 | b2 | single | download (`x5:efficientnet:EfficientNet_B2_224x224_nv12.bin`) |
| `EfficientNet_B3_224x224_nv12.bin` | x5 | b3 | single | download (`x5:efficientnet:EfficientNet_B3_224x224_nv12.bin`) |
| `EfficientNet_B4_224x224_nv12.bin` | x5 | b4 | single | download (`x5:efficientnet:EfficientNet_B4_224x224_nv12.bin`) |
| `s100/efficientnet_lite0_224x224_nv12.hbm` | s100 | lite0 | single | download (`s:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm`) |
| `s100/efficientnet_lite1_240x240_nv12.hbm` | s100 | lite1 | single | download (`s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm`) |
| `s100/efficientnet_lite2_260x260_nv12.hbm` | s100 | lite2 | single | download (`s:efficientnet:s100/efficientnet_lite2_260x260_nv12.hbm`) |
| `s100/efficientnet_lite3_300x300_nv12.hbm` | s100 | lite3 | single | download (`s:efficientnet:s100/efficientnet_lite3_300x300_nv12.hbm`) |
| `s100/efficientnet_lite4_380x380_nv12.hbm` | s100 | lite4 | single | download (`s:efficientnet:s100/efficientnet_lite4_380x380_nv12.hbm`) |
| `s600/efficientnet_lite0_224x224_nv12.hbm` | s600 | lite0 | single | download (`s:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm`) |
| `s600/efficientnet_lite1_240x240_nv12.hbm` | s600 | lite1 | single | download (`s:efficientnet:s600/efficientnet_lite1_240x240_nv12.hbm`) |
| `s600/efficientnet_lite2_260x260_nv12.hbm` | s600 | lite2 | single | download (`s:efficientnet:s600/efficientnet_lite2_260x260_nv12.hbm`) |
| `s600/efficientnet_lite3_300x300_nv12.hbm` | s600 | lite3 | single | download (`s:efficientnet:s600/efficientnet_lite3_300x300_nv12.hbm`) |
| `s600/efficientnet_lite4_380x380_nv12.hbm` | s600 | lite4 | single | download (`s:efficientnet:s600/efficientnet_lite4_380x380_nv12.hbm`) |

Each reference is an exact row of `docs/release/x5/models.yaml` or
`docs/release/s/models.yaml`; the manifest is the authority for URL and
format. X5 consumes one packed NV12 tensor; S100/S600 consume separate Y
and UV tensors — a bare filename cannot select the protocol, so the runtime
always pairs `--model-path` with the exact reference. The lite series has
**per-variant geometry** (224/240/260/300/380); the variant, not a default
size, selects it. Select the manifest reference that matches the target and variant.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── __init__.py  # Python script
├── download.py  # Prepare model files
└── download.sh  # Model preparation command
```

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target/variant — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/efficientnet/model/download.sh x5 b2
bash samples/vision/efficientnet/model/download.sh s100 lite4
bash samples/vision/efficientnet/model/download.sh s600 lite0
```

The Python form is equivalent:
`python3 samples/vision/efficientnet/model/download.py --target s100 --variant lite1`.
Omitting the variant resolves each target's source default (x5 `b2`,
s100/s600 `lite0`), matching the runtime's omitted-variant behavior; an
explicit `--variant` always selects exactly.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)` for these rows. Run the downloader before inference to place the selected artifact under this directory.

<a id="accompanying-files"></a>
## Accompanying files

Classification runs use the shared ImageNet class file
`datasets/imagenet/imagenet_classes.names` (X5 and S alike); it is the
runtime `--label-file` default and is committed to the repository, so no
download is needed. The ImageNet-1k label files under `test_data/`
(`imagenet_classes.names`, `imagenet1000_labels.txt`, `imagenet_1k.json`)
can be selected explicitly via `--label-file`.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live under the sample's `model/` directory
(X5 flat; S-series under `model/s100/` and `model/s600/`), relative to the
sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations. Models kept under the board's system directory
`/opt/hobot/model/<soc>/basic/` are not managed by this sample — pass the
explicit `--model-path` when using such a copy.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `EfficientNet_B2_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EfficientNet_B3_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EfficientNet_B4_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `s100/efficientnet_lite0_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s100/efficientnet_lite1_240x240_nv12.hbm` | nash-e `.hbm`, split Y/UV input (240x240), F32 `[1,1000]` logits output | null (unknown) |
| `s100/efficientnet_lite2_260x260_nv12.hbm` | nash-e `.hbm`, split Y/UV input (260x260), F32 `[1,1000]` logits output | null (unknown) |
| `s100/efficientnet_lite3_300x300_nv12.hbm` | nash-e `.hbm`, split Y/UV input (300x300), F32 `[1,1000]` logits output | null (unknown) |
| `s100/efficientnet_lite4_380x380_nv12.hbm` | nash-e `.hbm`, split Y/UV input (380x380), F32 `[1,1000]` logits output | null (unknown) |
| `s600/efficientnet_lite0_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s600/efficientnet_lite1_240x240_nv12.hbm` | nash-p `.hbm`, split Y/UV input (240x240), F32 `[1,1000]` logits output | null (unknown) |
| `s600/efficientnet_lite2_260x260_nv12.hbm` | nash-p `.hbm`, split Y/UV input (260x260), F32 `[1,1000]` logits output | null (unknown) |
| `s600/efficientnet_lite3_300x300_nv12.hbm` | nash-p `.hbm`, split Y/UV input (300x300), F32 `[1,1000]` logits output | null (unknown) |
| `s600/efficientnet_lite4_380x380_nv12.hbm` | nash-p `.hbm`, split Y/UV input (380x380), F32 `[1,1000]` logits output | null (unknown) |

The manifest SHA-256 fields are `null (unknown)`. The downloader prints each artifact's computed digest after transfer; keep it with that artifact identity.
