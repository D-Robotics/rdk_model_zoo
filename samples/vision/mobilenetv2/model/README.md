English | [简体中文](README_cn.md)

# MobileNetV2 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `mobilenetv2_100_bayese_224x224_nv12.bin` | x5 | 100 | single | download (`x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv2_100_nashe_224x224_nv12.hbm` | s100 | 100 | single | download (`s:mobilenetv2:s100/mobilenetv2_100_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv2_100_nashm_224x224_nv12.hbm` | s100p | 100 | single | download (`s:mobilenetv2:s100p/mobilenetv2_100_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv2_100_nashp_224x224_nv12.hbm` | s600 | 100 | single | download (`s:mobilenetv2:s600/mobilenetv2_100_nashp_224x224_nv12.hbm`) |
| `mobilenetv2_140_bayese_224x224_nv12.bin` | x5 | 140 | single | download (`x5:mobilenetv2:mobilenetv2_140_bayese_224x224_nv12.bin`) |
| `s100/mobilenetv2_140_nashe_224x224_nv12.hbm` | s100 | 140 | single | download (`s:mobilenetv2:s100/mobilenetv2_140_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv2_140_nashm_224x224_nv12.hbm` | s100p | 140 | single | download (`s:mobilenetv2:s100p/mobilenetv2_140_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv2_140_nashp_224x224_nv12.hbm` | s600 | 140 | single | download (`s:mobilenetv2:s600/mobilenetv2_140_nashp_224x224_nv12.hbm`) |

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
bash samples/vision/mobilenetv2/model/download.sh x5 100
bash samples/vision/mobilenetv2/model/download.sh x5 140
bash samples/vision/mobilenetv2/model/download.sh s100p 140
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv2/model/download.py --target s100 --variant 100`.
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
| `mobilenetv2_100_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `e044c3acb08403f2a3aee6342f4f131b798496773ebc7e4ae34a623ecff3e242` |
| `s100/mobilenetv2_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `e31d0445ead5f361dc0dae6a4c959d671a9c878a4f5a4d508d7fd5b9cc6672b5` |
| `s100p/mobilenetv2_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `906082fa0df94c88ad00187b6120f802a9e2047fc9a33745f737516e8f634159` |
| `s600/mobilenetv2_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `2509ba5e2a47ae97a7a43cc83fbb60c03b0cc5899f4580e8b4cc82ce38eafe1b` |
| `mobilenetv2_140_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `bf2a8920efbd7382e0fd63841d6217f9d0c311ea96cbaeee737d07dfe62e5aa9` |
| `s100/mobilenetv2_140_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `77155795844def681c4951ea4ee7e229a3cf1b6efd7ef8c71a0678731e189085` |
| `s100p/mobilenetv2_140_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `887d47c7825e1829177ffdf35f2b2b8d895b7d8c02f0fa1e1f9e2b09308ce63f` |
| `s600/mobilenetv2_140_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `b868f6aacfa7a8380f7950458df7da5a240d67d2a2f6c2dec7e08785f631cd60` |

Every artifact is a post-training INT8 quantization of a pinned
[timm](https://github.com/huggingface/pytorch-image-models) checkpoint
(`mobilenetv2_100.ra_in1k` and
`mobilenetv2_140.ra_in1k`, Apache-2.0), compiled with the
file name's march token: `bayese` (bayes-e, X5), `nashe` (nash-e, S100),
`nashm` (nash-m, S100P) and `nashp` (nash-p, S600). The network input is
normalized in the model; the runtime feeds NV12 produced from the
center crop at the model input size (see [conversion](../conversion/README.md#preprocessing)).
