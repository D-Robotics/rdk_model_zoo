English | [简体中文](README_cn.md)

# MobileNetV1 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `mobilenetv1_100_bayese_224x224_nv12.bin` | x5 | 100 | single | download (`x5:mobilenetv1:mobilenetv1_100_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv1_100_nashe_224x224_nv12.hbm` | s100 | 100 | single | download (`s:mobilenetv1:s100/mobilenetv1_100_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv1_100_nashm_224x224_nv12.hbm` | s100p | 100 | single | download (`s:mobilenetv1:s100p/mobilenetv1_100_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv1_100_nashp_224x224_nv12.hbm` | s600 | 100 | single | download (`s:mobilenetv1:s600/mobilenetv1_100_nashp_224x224_nv12.hbm`) |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | x5 | 125 | single | download (`x5:mobilenetv1:mobilenetv1_125_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv1_125_nashe_224x224_nv12.hbm` | s100 | 125 | single | download (`s:mobilenetv1:s100/mobilenetv1_125_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv1_125_nashm_224x224_nv12.hbm` | s100p | 125 | single | download (`s:mobilenetv1:s100p/mobilenetv1_125_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv1_125_nashp_224x224_nv12.hbm` | s600 | 125 | single | download (`s:mobilenetv1:s600/mobilenetv1_125_nashp_224x224_nv12.hbm`) |

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
bash samples/vision/mobilenetv1/model/download.sh x5 100
bash samples/vision/mobilenetv1/model/download.sh x5 125
bash samples/vision/mobilenetv1/model/download.sh s100p 125
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv1/model/download.py --target s100 --variant 100`.
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
| `mobilenetv1_100_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `23392a747c6c50b97f2ffe7412b0e2b3ded9245355fe158b988712007d501056` |
| `s100/mobilenetv1_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `8a27a19dbe4d9878328bb928aa8af395367b64a688629e82388e55410f25472d` |
| `s100p/mobilenetv1_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `34ba7820078612d9e070c601cedc9006345353547cbb4c6abdda991bdf1b06b3` |
| `s600/mobilenetv1_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `160c05acb159ae6170786bab32be42305ed62613468f45828f4f7fbe3939532b` |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `c4e3ef8bfcd41d25ca7c8e743cf443eba0841187eca78d737cb15d71ae53af80` |
| `s100/mobilenetv1_125_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `da510f1184db1b12cba0f9861f92133a5bede5eaf1088e4b7696813de9be205e` |
| `s100p/mobilenetv1_125_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `00f10280df4376f0de9f2c440fb70444d66155b24145ddff86c0b66102e07718` |
| `s600/mobilenetv1_125_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `993f487a9861a7281479d9a0cf825a253df7752cd2bb3ec3ff5dc2509bcfff64` |

Every artifact is a post-training INT8 quantization of a pinned
[timm](https://github.com/huggingface/pytorch-image-models) checkpoint
(`mobilenetv1_100.ra4_e3600_r224_in1k` and
`mobilenetv1_125.ra4_e3600_r224_in1k`, Apache-2.0), compiled with the
file name's march token: `bayese` (bayes-e, X5), `nashe` (nash-e, S100),
`nashm` (nash-m, S100P) and `nashp` (nash-p, S600). The network input is
normalized in the model; the runtime feeds NV12 produced from the
center crop at the model input size (see [conversion](../conversion/README.md#preprocessing)).
The S100, S100P and S600 builds of `100` additionally use the toolchain's
weight bias correction (still INT8; see [conversion](../conversion/README.md#compile)).
