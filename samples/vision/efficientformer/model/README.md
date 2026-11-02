English | [简体中文](README_cn.md)

# EfficientFormer model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EfficientFormer_l1_224x224_nv12.bin` | x5 | l1 | single | download (`x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin`) |
| `EfficientFormer_l3_224x224_nv12.bin` | x5 | l3 | single | download (`x5:efficientformer:EfficientFormer_l3_224x224_nv12.bin`) |

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
bash samples/vision/efficientformer/model/download.sh x5 l1
bash samples/vision/efficientformer/model/download.sh x5 l3
```

The Python form is equivalent:
`python3 samples/vision/efficientformer/model/download.py --target x5 --variant l1`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)` for these rows. Run the downloader before inference to place the selected artifact under this directory.

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
| `EfficientFormer_l1_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EfficientFormer_l3_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

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
| `timm-l1-snap-dist-in1k` | `efficientformer_l1_snap_dist_in1k_bayese_224x224_nv12.bin` | [`8738a5819080`](https://huggingface.co/timm/efficientformer_l1.snap_dist_in1k/blob/8738a58190800aaf5995f453b05f643df453a565/README.md) | not-run |

```bash
python3 samples/vision/efficientformer/model/download.py --target x5 --variant timm-l1-snap-dist-in1k
python3 samples/vision/efficientformer/runtime/python/main.py --target x5 --variant timm-l1-snap-dist-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
