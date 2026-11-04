English | [简体中文](README_cn.md)

# EfficientFormerV2 model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EfficientFormerv2_s0_224x224_nv12.bin` | x5 | s0 | single | download (`x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin`) |
| `EfficientFormerv2_s1_224x224_nv12.bin` | x5 | s1 | single | download (`x5:efficientformerv2:EfficientFormerv2_s1_224x224_nv12.bin`) |
| `EfficientFormerv2_s2_224x224_nv12.bin` | x5 | s2 | single | download (`x5:efficientformerv2:EfficientFormerv2_s2_224x224_nv12.bin`) |

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
bash samples/vision/efficientformerv2/model/download.sh x5 s0
bash samples/vision/efficientformerv2/model/download.sh x5 s2
```

The Python form is equivalent:
`python3 samples/vision/efficientformerv2/model/download.py --target x5 --variant s1`.
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
| `EfficientFormerv2_s0_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EfficientFormerv2_s1_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EfficientFormerv2_s2_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

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
| `timm-s0-snap-dist-in1k` | `efficientformerv2_s0_snap_dist_in1k_bayese_224x224_nv12.bin` | [`aa226e91a076`](https://huggingface.co/timm/efficientformerv2_s0.snap_dist_in1k/blob/aa226e91a076e760bc7957e5990f19f6ae322c5f/README.md) | not-run |
| `timm-s1-snap-dist-in1k` | `efficientformerv2_s1_snap_dist_in1k_bayese_224x224_nv12.bin` | [`0c1fc60a0e89`](https://huggingface.co/timm/efficientformerv2_s1.snap_dist_in1k/blob/0c1fc60a0e89b6309d9de451cdacc11ac0a8b987/README.md) | not-run |
| `timm-s2-snap-dist-in1k` | `efficientformerv2_s2_snap_dist_in1k_bayese_224x224_nv12.bin` | [`1c56a7635500`](https://huggingface.co/timm/efficientformerv2_s2.snap_dist_in1k/blob/1c56a76355000c79568559d34ba3fa24416b5107/README.md) | not-run |

```bash
python3 samples/vision/efficientformerv2/model/download.py --target x5 --variant timm-s0-snap-dist-in1k
python3 samples/vision/efficientformerv2/runtime/python/main.py --target x5 --variant timm-s0-snap-dist-in1k --dry-run
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
| s100 | `timm-s0-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-s0-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-s0-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-s1-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-s1-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-s1-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s100 | `timm-s2-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s100p | `timm-s2-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |
| s600 | `timm-s2-snap-dist-in1k-s-int16-o1-20261103` | model | 200 |

```bash
python3 samples/vision/efficientformerv2/model/download.py --target s100 --variant timm-s0-snap-dist-in1k-s-int16-o1-20261103
python3 samples/vision/efficientformerv2/runtime/python/main.py --target s100 --variant timm-s0-snap-dist-in1k-s-int16-o1-20261103 --asset-id s:efficientformerv2:s100/efficientformerv2_s0_snap_dist_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --model-path /path/to/efficientformerv2_s0_snap_dist_in1k_nashe_224x224_nv12_float_int16_20261103_o1.hbm --test-img samples/vision/efficientformerv2/test_data/goldfish.JPEG
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
