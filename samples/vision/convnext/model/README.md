English | [简体中文](README_cn.md)

# ConvNeXt model artifacts

Prepare the model artifact with the downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `ConvNeXt_atto_224x224_nv12.bin` | x5 | atto | single | download (`x5:convnext:ConvNeXt_atto_224x224_nv12.bin`) |
| `ConvNeXt_nano_224x224_nv12.bin` | x5 | nano | single | download (`x5:convnext:ConvNeXt_nano_224x224_nv12.bin`) |
| `ConvNeXt_pico_224x224_nv12.bin` | x5 | pico | single | download (`x5:convnext:ConvNeXt_pico_224x224_nv12.bin`) |
| `ConvNeXt_femto_224x224_nv12.bin` | x5 | femto | single | download (`x5:convnext:ConvNeXt_femto_224x224_nv12.bin`) |

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
bash samples/vision/convnext/model/download.sh x5
bash samples/vision/convnext/model/download.sh x5 nano
```

The Python form is equivalent:
`python3 samples/vision/convnext/model/download.py --target x5`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. Run it before inference; it prints the computed digest after transfer. The manifest SHA-256 is `null (unknown)` for this row.

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
| `ConvNeXt_atto_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `ConvNeXt_nano_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `ConvNeXt_pico_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `ConvNeXt_femto_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest SHA-256 field is `null (unknown)`. The downloader prints the artifact's computed digest after transfer; keep it with that artifact identity.

## Atto D2 converted artifact (2026-11-02)

`atto-d2` is a separate conversion of [timm/convnext_atto.d2_in1k](https://huggingface.co/timm/convnext_atto.d2_in1k/tree/435c0a8d02d5b872e90c8cde36887465982cb399) (Apache-2.0), pinned to `435c0a8d02d5b872e90c8cde36887465982cb399`. The default `atto` and the four historical artifacts retain their identities and measurements.

The new 4,528,264-byte BIN has SHA-256 `c334e6c87ae202a0ef8d78b4506ecdab7fafb4684ce720fc4407f9bf990ea101`. Export and host PyTorch/ONNX comparison passed; board smoke, accuracy, Runtime 1/2 and C++ 1/2 are **not-run**. Historical performance tables do not describe this file. Its preprocessing is Pillow bicubic resize (shorter edge 256), rounded center crop 224, then OpenCV BT.601 video-range NV12. Mapper fuses ImageNet normalization. Output is float32 logits for 1,000 classes; runtime applies softmax. Pillow is required for this variant.

```bash
# From the repository root; download and verify the registered artifact.
python3 samples/vision/convnext/model/download.py --target x5 --variant atto-d2
# Inspect the contract on a host; no SDK, download or board inference.
python3 samples/vision/convnext/runtime/python/main.py --target x5 --variant atto-d2 --dry-run
# On a prepared X5 board; not executed for this converted-only release.
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 --asset-id x5:convnext:convnext_atto_d2_bayese_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/convnext_atto_d2_bayese_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG
```

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
| `timm-femto-d1-in1k` | `convnext_femto_d1_in1k_bayese_224x224_nv12.bin` | [`1e0c02df687c`](https://huggingface.co/timm/convnext_femto.d1_in1k/blob/1e0c02df687c47abf0819e1a4f858293e17e0c50/README.md) | not-run |
| `timm-pico-d1-in1k` | `convnext_pico_d1_in1k_bayese_224x224_nv12.bin` | [`0d879cb0cfd9`](https://huggingface.co/timm/convnext_pico.d1_in1k/blob/0d879cb0cfd9b851a13f007d3ed170dd049f1476/README.md) | not-run |
| `timm-nano-d1h-in1k` | `convnext_nano_d1h_in1k_bayese_224x224_nv12.bin` | [`fe9cba6df2f1`](https://huggingface.co/timm/convnext_nano.d1h_in1k/blob/fe9cba6df2f17352639e23ca89e1aa6b37f51462/README.md) | not-run |

```bash
python3 samples/vision/convnext/model/download.py --target x5 --variant timm-femto-d1-in1k
python3 samples/vision/convnext/runtime/python/main.py --target x5 --variant timm-femto-d1-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->
