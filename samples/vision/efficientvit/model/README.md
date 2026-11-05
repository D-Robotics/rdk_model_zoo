English | [简体中文](README_cn.md)

# EfficientViT model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EfficientViT_m5_224x224_nv12.bin` | x5 | m5 | single | download (`x5:efficientvit:EfficientViT_m5_224x224_nv12.bin`) |

The reference is an exact row of `docs/release/x5/models.yaml`; the
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
bash samples/vision/efficientvit/model/download.sh x5
```

The Python form is equivalent:
`python3 samples/vision/efficientvit/model/download.py --target x5`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)` for these rows. Run the downloader before inference to place the selected artifact under this directory.

<a id="accompanying-files"></a>
## Accompanying files

Use the checked-in one-label-per-line ImageNet class file at run time: `datasets/imagenet/imagenet_classes.names`.

<a id="local-paths"></a>
## Local paths

After preparation, the artifact lives flat under the sample's `model/`
directory, relative to the sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at this
location.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `EfficientViT_m5_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest SHA-256 field is `null (unknown)`. The downloader prints the artifact's computed digest after transfer; keep it with that artifact identity.

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
| `timm-m5-r224-in1k` | `efficientvit_m5_r224_in1k_bayese_224x224_nv12.bin` | [`e45d80f65cd8`](https://huggingface.co/timm/efficientvit_m5.r224_in1k/blob/e45d80f65cd8ef0964ab526dd4f8025abbff4e19/README.md) | not-run |

```bash
python3 samples/vision/efficientvit/model/download.py --target x5 --variant timm-m5-r224-in1k
python3 samples/vision/efficientvit/runtime/python/main.py --target x5 --variant timm-m5-r224-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100 | `timm-m5-r224-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-m5-r224-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-m5-r224-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/efficientvit/model/download.py --target s100 --variant timm-m5-r224-in1k-s-int16-o1-20261103
python3 samples/vision/efficientvit/runtime/python/main.py --target s100 --variant timm-m5-r224-in1k-s-int16-o1-20261103 --asset-id s:efficientvit:s100/efficientvit_m5_r224_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/efficientvit_m5_r224_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/efficientvit/test_data/hook.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
