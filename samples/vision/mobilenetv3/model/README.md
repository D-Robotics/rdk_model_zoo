English | [简体中文](README_cn.md)

# MobileNetV3 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm` | s100 | large | single | download (`s:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm` | s100p | large | single | download (`s:mobilenetv3:s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm` | s600 | large | single | download (`s:mobilenetv3:s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm`) |
| `s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm` | s100 | small | single | download (`s:mobilenetv3:s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm`) |
| `s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm` | s100p | small | single | download (`s:mobilenetv3:s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm`) |
| `s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm` | s600 | small | single | download (`s:mobilenetv3:s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm`) |

Each reference is an exact row of
`docs/release/s/models.yaml`; the manifest is the authority for URL and
format. S100/S100P/S600 consume separate Y
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
bash samples/vision/mobilenetv3/model/download.sh s100 large
bash samples/vision/mobilenetv3/model/download.sh s100 small
bash samples/vision/mobilenetv3/model/download.sh s100p small
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv3/model/download.py --target s100 --variant large`.
The downloader checks content length and the manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer, which must equal the SHA-256 listed in the table below. Run the downloader before inference to place the selected artifact under this directory.

<a id="accompanying-files"></a>
## Accompanying files

Classification runs use the shared ImageNet class file
`datasets/imagenet/imagenet_classes.names`; it is
the runtime `--label-file` default and is committed to the repository, so no
download is needed. The `imagenet_classes.names` under `test_data/` matches
that file byte for byte and can be selected explicitly via `--label-file`.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live under the sample's `model/` directory
(under `model/s100/`, `model/s100p/` and `model/s600/`), relative to the
sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `9190d5334b2f1488bbfa3cc4f918bb21ec788d4fcc08a55074b5287cb7baf24d` |
| `s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `96e6bda58009330c96d139bdf858326ddf5481299b68d6ecadd3b2b3bc7a1658` |
| `s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `9b2b2422566214ebdbad27cedebbbaf5668edd9272c3efc51adb1113be8dbad4` |
| `s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `c8cfa5cfe34d46b99e235d9823bbbf464e67808d59d9e3281593c87cc07bd6fc` |
| `s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `a67df2c4154d44c9ebe25e5250d16409a9a1ad31c953cc311b703f396151bcec` |
| `s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `8f2177eacf3496aeea1c4234146ea73eaad99882ff74b0bd0afa47b266849b08` |

Every artifact is a post-training quantization (INT16 activations, INT8 weights) of a pinned
[timm](https://github.com/huggingface/pytorch-image-models) checkpoint
(`mobilenetv3_large_100.ra_in1k` and
`mobilenetv3_small_100.lamb_in1k`, Apache-2.0), compiled with the
file name's march token: `nashe` (nash-e, S100),
`nashm` (nash-m, S100P) and `nashp` (nash-p, S600). The network input is
normalized in the model; the runtime feeds NV12 produced from the
center crop at the model input size (see [conversion](../conversion/README.md#preprocessing)).
