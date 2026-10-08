English | [简体中文](README_cn.md)

# MobileNetV4 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `MobileNetV4_conv_small_224x224_nv12.bin` | x5 | small | single | download (`x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin`) |
| `MobileNetV4_conv_medium_224x224_nv12.bin` | x5 | medium | single | download (`x5:mobilenetv4:MobileNetV4_conv_medium_224x224_nv12.bin`) |
| `s100/mobilenetv4_small_224x224_nv12.hbm` | s100 | small | single | download (`s:mobilenetv4:s100/mobilenetv4_small_224x224_nv12.hbm`) |
| `s600/mobilenetv4_small_224x224_nv12.hbm` | s600 | small | single | download (`s:mobilenetv4:s600/mobilenetv4_small_224x224_nv12.hbm`) |
| `s100/mobilenetv4_medium_256x256_nv12.hbm` | s100 | medium | single | download (`s:mobilenetv4:s100/mobilenetv4_medium_256x256_nv12.hbm`) |
| `s600/mobilenetv4_medium_256x256_nv12.hbm` | s600 | medium | single | download (`s:mobilenetv4:s600/mobilenetv4_medium_256x256_nv12.hbm`) |

Each reference is an exact row of `docs/release/x5/models.yaml` or
`docs/release/s/models.yaml`; the manifest is the authority for URL and
format. X5 consumes one packed NV12 tensor; S100/S600 consume separate Y
and UV tensors — the runtime pairs `--model-path` with the exact target reference so the input protocol matches the artifact.

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
bash samples/vision/mobilenetv4/model/download.sh x5 small
bash samples/vision/mobilenetv4/model/download.sh x5 medium
bash samples/vision/mobilenetv4/model/download.sh s100 medium
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv4/model/download.py --target s100 --variant small`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)` for these rows. Run the downloader before inference to place the selected artifact under this directory.

<a id="accompanying-files"></a>
## Accompanying files

Classification runs use the shared ImageNet class file
`datasets/imagenet/imagenet_classes.names` (X5 and S series alike); it is
the runtime `--label-file` default and is committed to the repository, so no
download is needed. The `imagenet_classes.names` under `test_data/` matches
that file byte for byte and can be selected explicitly via `--label-file`.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live under the sample's `model/` directory
(X5 flat; S-series under `model/s100/` and `model/s600/`), relative to the
sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `MobileNetV4_conv_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input, F32 `[1,1000,1,1]` logits output | null (unknown) |
| `MobileNetV4_conv_medium_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input, F32 `[1,1000,1,1]` logits output | null (unknown) |
| `s100/mobilenetv4_small_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s600/mobilenetv4_small_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s100/mobilenetv4_medium_256x256_nv12.hbm` | nash `.hbm`, split Y/UV input (256x256), F32 `[1,1000]` logits output | null (unknown) |
| `s600/mobilenetv4_medium_256x256_nv12.hbm` | nash `.hbm`, split Y/UV input (256x256), F32 `[1,1000]` logits output | null (unknown) |

The manifest SHA-256 fields are `null (unknown)`. The downloader prints each artifact's computed digest after transfer; keep it with that artifact identity.
