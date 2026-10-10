English | [简体中文](README_cn.md)

# MobileNetV4 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `mobilenetv4_conv_small_bayese_224x224_nv12.bin` | x5 | small | single | download (`x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm` | s100 | small | single | download (`s:mobilenetv4:s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm` | s100p | small | single | download (`s:mobilenetv4:s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm` | s600 | small | single | download (`s:mobilenetv4:s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm`) |
| `mobilenetv4_conv_medium_bayese_224x224_nv12.bin` | x5 | medium | single | download (`x5:mobilenetv4:mobilenetv4_conv_medium_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm` | s100 | medium | single | download (`s:mobilenetv4:s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm` | s100p | medium | single | download (`s:mobilenetv4:s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm` | s600 | medium | single | download (`s:mobilenetv4:s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm`) |

Each reference is an exact row of `docs/release/x5/models.yaml` or
`docs/release/s/models.yaml`; the manifest is the authority for URL and
format. X5 consumes one packed NV12 tensor; S100/S100P/S600 consume separate Y
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
bash samples/vision/mobilenetv4/model/download.sh s100p medium
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv4/model/download.py --target s100 --variant small`.
The downloader checks content length and the manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer, which must equal the SHA-256 listed in the table below. Run the downloader before inference to place the selected artifact under this directory.

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
(X5 flat; S-series under `model/s100/`, `model/s100p/` and `model/s600/`), relative to the
sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `mobilenetv4_conv_small_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `3978f150ce77e725c80c1624f39300a56b8def69543449f6dd3fc9da9a46ff1b` |
| `s100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `ae8730fab716b4a7046304e10c855bc9db5c8d2385a569e349b315f8f98446d5` |
| `s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `34114e3f00030a9a01aea4f9634387be6441b57603267f59ef9f9d9190d5ec10` |
| `s600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `a2bb74d2051c8495827d14071676a7320cbb2637507b424822388608269298a2` |
| `mobilenetv4_conv_medium_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `0a1dc43e2a6d9b789112307f2f2c681d3abfd1406aaef3fde3c3a1d3870f9025` |
| `s100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `adec594b442eb5efcd2ff1694b86f087d630e2a933871bffaeebea7ae956ffca` |
| `s100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `b95804313b774a49c3ebc48a3f3ca278964c93df456822ac47b9ee6abc133107` |
| `s600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `5b1b0abc231e6d0272a39de32885ef46eb3755e808763def5efbccaaa73d415b` |

Every artifact is a post-training INT8 quantization of a pinned
[timm](https://github.com/huggingface/pytorch-image-models) checkpoint
(`mobilenetv4_conv_small.e2400_r224_in1k` and
`mobilenetv4_conv_medium.e500_r224_in1k`, Apache-2.0), compiled with the
file name's march token: `bayese` (bayes-e, X5), `nashe` (nash-e, S100),
`nashm` (nash-m, S100P) and `nashp` (nash-p, S600). The network input is
normalized in the model; the runtime feeds NV12 produced from a 224x224
center crop (see [conversion](../conversion/README.md#preprocessing)).
