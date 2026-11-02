English | [简体中文](README_cn.md)

# EdgeNeXt model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EdgeNeXt_base_224x224_nv12.bin` | x5 | base | single | download (`x5:edgenext:EdgeNeXt_base_224x224_nv12.bin`) |
| `EdgeNeXt_small_224x224_nv12.bin` | x5 | small | single | download (`x5:edgenext:EdgeNeXt_small_224x224_nv12.bin`) |
| `EdgeNeXt_x_small_224x224_nv12.bin` | x5 | x_small | single | download (`x5:edgenext:EdgeNeXt_x_small_224x224_nv12.bin`) |
| `EdgeNeXt_xx_small_224x224_nv12.bin` | x5 | xx_small | single | download (`x5:edgenext:EdgeNeXt_xx_small_224x224_nv12.bin`) |

Each reference is an exact row of `docs/release/x5/models.yaml`; the
manifest is the authority for URL and format. X5 consumes one packed NV12
tensor, so pair `--model-path` with its exact manifest reference. Choose a target and variant listed in the sample support matrix, then prepare that artifact.

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
bash samples/vision/edgenext/model/download.sh x5 base
bash samples/vision/edgenext/model/download.sh x5 xx_small
```

The Python form is equivalent:
`python3 samples/vision/edgenext/model/download.py --target x5 --variant small`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. Run it before inference; it prints the computed digest after transfer. The manifest SHA-256 is `null (unknown)` for this row.

<a id="accompanying-files"></a>
## Accompanying files

Use the checked-in one-label-per-line ImageNet class file at run time: `datasets/imagenet/imagenet_classes.names`.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live flat under the sample's `model/`
directory, relative to the sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `EdgeNeXt_base_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_x_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_xx_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest SHA-256 fields are `null (unknown)`. The downloader prints each artifact's computed digest after transfer; keep that digest with its artifact identity.

<!-- converted-batch-20261102 -->
## Converted-only additions (2026-11-02)

These independently converted timm weights have passed strict checkpoint loading,
reparameterization and two-input PyTorch/ONNX host comparison, checker and compiler.
Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are **not-run**; historical metrics
and screenshots do not describe these artifacts. Existing defaults are retained.
Pillow is needed for the explicit bicubic shorter-edge 256 / rounded center crop
224 profile. Normalization is fused in Mapper from the actual checkpoint config;
output is float32 1,000-class logits followed by native softmax.

| Variant | Artifact | Checkpoint revision | Board |
|---|---|---|---|
| `timm-xx-small-in1k` | `edgenext_xx_small_in1k_bayese_224x224_nv12.bin` | [`e10d04b6e70f`](https://huggingface.co/timm/edgenext_xx_small.in1k/blob/e10d04b6e70f33cfb54ebd4bce2ede078f4c9ba2/README.md) | not-run |
| `timm-x-small-in1k` | `edgenext_x_small_in1k_bayese_224x224_nv12.bin` | [`8525b3ecc3c7`](https://huggingface.co/timm/edgenext_x_small.in1k/blob/8525b3ecc3c7a25c07406163de827d8814f5f1ef/README.md) | not-run |
| `timm-small-usi-in1k` | `edgenext_small_usi_in1k_bayese_224x224_nv12.bin` | [`81bd8f2ee563`](https://huggingface.co/timm/edgenext_small.usi_in1k/blob/81bd8f2ee56330bdf6a7a2e47e1b2b82eb34f978/README.md) | not-run |
| `timm-base-usi-in1k` | `edgenext_base_usi_in1k_bayese_224x224_nv12.bin` | [`ce0946556d0e`](https://huggingface.co/timm/edgenext_base.usi_in1k/blob/ce0946556d0ed93eaa61c686955e3118814573bb/README.md) | not-run |

```bash
python3 samples/vision/edgenext/model/download.py --target x5 --variant timm-xx-small-in1k
python3 samples/vision/edgenext/runtime/python/main.py --target x5 --variant timm-xx-small-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
