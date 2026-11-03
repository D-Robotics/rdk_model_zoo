English | [简体中文](README_cn.md)

# EfficientSAM Model Preparation

<a id="artifacts"></a>
## Artifacts

| Target | Encoder | Decoder | Format |
|---|---|---|---|
| x5 | `efficient_sam_vitt_encoder_512x512_default_none.bin` | `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | BIN |
| s100 | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` | HBM |
| s100p | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` | HBM |
| s600 | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` | HBM |

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── download_model.sh  # Shell command
```

<a id="preparation"></a>
## Preparation

From the repository root, run the explicit target command using Python and PyYAML on a host or board. It downloads both manifest assets into `samples/vision/efficient_sam/model/` (X5) or its `nash-e/`, `nash-m/`, `nash-p/` subdirectory (S) and reports observed digests. It requires network access and runs as an explicit preparation step before inference.

```bash
python3 samples/vision/efficient_sam/model/download.py --target x5
python3 samples/vision/efficient_sam/model/download.py --target s100
python3 samples/vision/efficient_sam/model/download.py --target s100p
python3 samples/vision/efficient_sam/model/download.py --target s600
# Run only the command for your target; each command prepares one encoder/decoder pair.
```

Use the exact URLs and filename identities in the [X5 manifest](../../../../docs/release/x5/models.yaml) and [S manifest](../../../../docs/release/s/models.yaml). If download access fails, transfer the same two files from another machine into the paths listed above; the runtime never substitutes a different board's pair. Existing files are verified and are not overwritten. An observed digest identifies the local bytes; with the publisher SHA unknown, it does not independently prove their origin. Run `--list-models` on the sample runtime to obtain exact stage asset IDs for custom paths.

<a id="accompanying-files"></a>
## Accompanying Files

`../test_data/dogs.jpg` is the supplied input fixture. `../test_data/efficient_sam_binary_mask.png` is a preserved source reference mask. Neither file is a model asset.

<a id="local-paths"></a>
## Local Paths

The runtime resolves the selected pair below `samples/vision/efficient_sam/model/` from the exact manifest filenames. Custom encoder or decoder paths must each be paired with their exact corresponding `--encoder-asset-id` or `--decoder-asset-id`.

<a id="formats-checksums"></a>
## Formats and Checksums

All published rows are `bin` for x5 and `hbm` for S targets. The active manifests record `sha256: null (unknown)` for every asset; no digest is guessed or copied.

## Converted S100 profile (2026-11-03)

`vitt-512-int16-20261103-o1` · S100 / nash-e · OE 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5) · O1.

Select this independent build explicitly. Historical defaults and measurements are retained. Actual HBM outputs are float32.

| Role | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| encoder | `s:efficient_sam:nash-e/efficient_sam_vitt_encoder_512x512_nashe_int16_20261103_o1.hbm` | `5fa5d84783124b653a211272574fe71fd4b3e5b42bc3bb8e84ec7ec7f0e4e238` | 24795880 |
| decoder | `s:efficient_sam:nash-e/efficient_sam_vitt_decoder_512_nashe_int16_20261103_o1.hbm` | `d106a8c9ba8c2a038961e5c34ac18b62b352fc708be6299009c763eb40ef1836` | 22544128 |

```bash
python3 samples/vision/efficient_sam/model/download.py --target s100 --variant vitt-512-int16-20261103-o1
python3 samples/vision/efficient_sam/runtime/python/main.py --target s100 --variant vitt-512-int16-20261103-o1 --test-img samples/vision/efficient_sam/test_data/dogs.jpg
```

Calibration uses 200 independent COCO training photographs. Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are `not-run`. Source, checkpoint, ONNX, effective configuration, native recipe and complete OE identities are recorded in [converted.json](../conversion/converted.json).
