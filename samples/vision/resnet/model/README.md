English | [简体中文](README_cn.md)

# ResNet model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `resnet18_bayese_224x224_nv12.bin` | x5 | 18 | single | download (`x5:resnet:resnet18_bayese_224x224_nv12.bin`) |
| `s100/resnet18_nashe_224x224_nv12.hbm` | s100 | 18 | single | download (`s:resnet:s100/resnet18_nashe_224x224_nv12.hbm`) |
| `s100p/resnet18_nashm_224x224_nv12.hbm` | s100p | 18 | single | download (`s:resnet:s100p/resnet18_nashm_224x224_nv12.hbm`) |
| `s600/resnet18_nashp_224x224_nv12.hbm` | s600 | 18 | single | download (`s:resnet:s600/resnet18_nashp_224x224_nv12.hbm`) |
| `resnet34_bayese_224x224_nv12.bin` | x5 | 34 | single | download (`x5:resnet:resnet34_bayese_224x224_nv12.bin`) |
| `s100/resnet34_nashe_224x224_nv12.hbm` | s100 | 34 | single | download (`s:resnet:s100/resnet34_nashe_224x224_nv12.hbm`) |
| `s100p/resnet34_nashm_224x224_nv12.hbm` | s100p | 34 | single | download (`s:resnet:s100p/resnet34_nashm_224x224_nv12.hbm`) |
| `s600/resnet34_nashp_224x224_nv12.hbm` | s600 | 34 | single | download (`s:resnet:s600/resnet34_nashp_224x224_nv12.hbm`) |
| `resnet50_bayese_224x224_nv12.bin` | x5 | 50 | single | download (`x5:resnet:resnet50_bayese_224x224_nv12.bin`) |
| `s100/resnet50_nashe_224x224_nv12.hbm` | s100 | 50 | single | download (`s:resnet:s100/resnet50_nashe_224x224_nv12.hbm`) |
| `s100p/resnet50_nashm_224x224_nv12.hbm` | s100p | 50 | single | download (`s:resnet:s100p/resnet50_nashm_224x224_nv12.hbm`) |
| `s600/resnet50_nashp_224x224_nv12.hbm` | s600 | 50 | single | download (`s:resnet:s600/resnet50_nashp_224x224_nv12.hbm`) |
| `resnet101_bayese_224x224_nv12.bin` | x5 | 101 | single | download (`x5:resnet:resnet101_bayese_224x224_nv12.bin`) |
| `s100/resnet101_nashe_224x224_nv12.hbm` | s100 | 101 | single | download (`s:resnet:s100/resnet101_nashe_224x224_nv12.hbm`) |
| `s100p/resnet101_nashm_224x224_nv12.hbm` | s100p | 101 | single | download (`s:resnet:s100p/resnet101_nashm_224x224_nv12.hbm`) |
| `s600/resnet101_nashp_224x224_nv12.hbm` | s600 | 101 | single | download (`s:resnet:s600/resnet101_nashp_224x224_nv12.hbm`) |
| `resnet152_bayese_224x224_nv12.bin` | x5 | 152 | single | download (`x5:resnet:resnet152_bayese_224x224_nv12.bin`) |
| `s100/resnet152_nashe_224x224_nv12.hbm` | s100 | 152 | single | download (`s:resnet:s100/resnet152_nashe_224x224_nv12.hbm`) |
| `s100p/resnet152_nashm_224x224_nv12.hbm` | s100p | 152 | single | download (`s:resnet:s100p/resnet152_nashm_224x224_nv12.hbm`) |
| `s600/resnet152_nashp_224x224_nv12.hbm` | s600 | 152 | single | download (`s:resnet:s600/resnet152_nashp_224x224_nv12.hbm`) |

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
# output: exit 0, observed digest printed; no partial files left behind
bash samples/vision/resnet/model/download.sh x5 18
bash samples/vision/resnet/model/download.sh x5 50
bash samples/vision/resnet/model/download.sh s100p 152
```

The Python form is equivalent:
`python3 samples/vision/resnet/model/download.py --target s100 --variant 18`.
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
| `resnet18_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `98a16c1136c2b993cd63b6d3e2cc4eb54b44a563d2ef95f0e0319f0003c45483` |
| `s100/resnet18_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `be56e79b3b53920de23b41c6ffea9738ea1acdb1c36570aa2a48b04eb712d6fc` |
| `s100p/resnet18_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `1d38d8457e4fc63f7148244a5d36411a797d6671059d59f071fe7ef416858782` |
| `s600/resnet18_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `488144886f10bfe6539b003fe1f7df594e33ee39fdf59b06f8a9efc8c2072f64` |
| `resnet34_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `06b882b882b5e05b6a8f5ce021ce82fb0c023b459eb60439dfe8bbbcc2eddb07` |
| `s100/resnet34_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `fbac49e060e2c39c1d971e844604c593f17482307b0b81d7665cf8a71e7e7abf` |
| `s100p/resnet34_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `5b3a8d906bcbc3fdcba3a11a6bb9636fd548c704948583345e9ec7af5c5b9a4b` |
| `s600/resnet34_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `f57bf280157b858b553ccb4e8f9cd1f52f14f4fdacccc83ce4bf20943857b72f` |
| `resnet50_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `6910443e0bcaaf9b2a3411633674af38e577260b15129a73dd6e68fcb972250b` |
| `s100/resnet50_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `118bc79923f698afd91666f19f32ae8651ee46e3f1aedfdc1f11bff1e8eb74e9` |
| `s100p/resnet50_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `0bdfebc791334efcbe7b61e6551d3900786aade6f2239ccb214e30d3b6ee6375` |
| `s600/resnet50_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `6bcd64de9dc28c87229d5c1f45046eddccec7ba3c331a4a1215ab22133a30877` |
| `resnet101_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `67db7c52d02b858526328fdfc541a9c50119da98882de9bd836754b9e882c116` |
| `s100/resnet101_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `9ae4a198c1b5becdf1dd564eb5627a3f2b8ec9ecc29315d360ce41ac321c7142` |
| `s100p/resnet101_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `47f07158654e29c60336d9bb37634d0146fe86e924554b33d70cd29605732d72` |
| `s600/resnet101_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `3662582a82aaace43d742924ecd74750d754385c5df8d55e0d0643e632f27495` |
| `resnet152_bayese_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | `76a029ff8b0db1ac98ffdbcdeb0bfae7552b03a5491e72815e7f3f46a4547497` |
| `s100/resnet152_nashe_224x224_nv12.hbm` | nash-e `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `a4b93d4659b0bc5e1bbdad9e0b6814291c837d24148ca29597bc1a9149c699f0` |
| `s100p/resnet152_nashm_224x224_nv12.hbm` | nash-m `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `892b1c9849ceb10cb555c1abf8a1b0a8abfec715c09962b74b9f1feb5c80ee34` |
| `s600/resnet152_nashp_224x224_nv12.hbm` | nash-p `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | `382582948c00baafe070ee702b71104d3f4bd48aa9e4651bfb387e77c625c6f9` |

Every artifact is a post-training INT8 quantization of a pinned
[timm](https://github.com/huggingface/pytorch-image-models) checkpoint
(`resnet18.a1_in1k`, `resnet34.a1_in1k`, `resnet50.a1_in1k`, `resnet101.a1_in1k` and
`resnet152.a1_in1k`, Apache-2.0), compiled with the
file name's march token: `bayese` (bayes-e, X5), `nashe` (nash-e, S100),
`nashm` (nash-m, S100P) and `nashp` (nash-p, S600). The network input is
normalized in the model; the runtime feeds NV12 produced from the
center crop at the model input size (see [conversion](../conversion/README.md#preprocessing)).
