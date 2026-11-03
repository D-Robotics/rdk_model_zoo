English | [简体中文](README_cn.md)

# MobileSAM Model Preparation

<a id="artifacts"></a>
## Artifacts

| Target | Encoder | Decoder | Format |
|---|---|---|---|
| x5 | `mobile_sam_image_encoder_norm_512x512_allint16.bin` | `mobile_sam_decoder_512_box_default.bin` | BIN |
| s100 | `nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm` | `nash-e/mobile_sam_decoder_512_nashe.hbm` | HBM |
| s100p | `nash-m/mobile_sam_image_encoder_norm_512x512_nashm.hbm` | `nash-m/mobile_sam_decoder_512_nashm.hbm` | HBM |
| s600 | `nash-p/mobile_sam_image_encoder_norm_512x512_nashp.hbm` | `nash-p/mobile_sam_decoder_512_nashp.hbm` | HBM |

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

From the repository root, run the explicit target command using Python and PyYAML on a host or board. It downloads both manifest assets into `samples/vision/mobile_sam/model/` (X5) or its `nash-e/`, `nash-m/`, `nash-p/` subdirectory (S) and reports observed digests. Network access is required for this step; inference never downloads.

```bash
python3 samples/vision/mobile_sam/model/download.py --target x5
python3 samples/vision/mobile_sam/model/download.py --target s100
python3 samples/vision/mobile_sam/model/download.py --target s100p
python3 samples/vision/mobile_sam/model/download.py --target s600
# Run only the command for your target; each command prepares one encoder/decoder pair.
```

Use the exact URLs and filename identities in the [X5 manifest](../../../../docs/release/x5/models.yaml) and [S manifest](../../../../docs/release/s/models.yaml). If download access fails, transfer the same two files from another machine into the paths listed above; the runtime never substitutes a different board's pair. Existing files are verified and are not overwritten. An observed digest identifies the local bytes; with the publisher SHA unknown, it does not independently prove their origin. Run `--list-models` on the sample runtime to obtain exact stage asset IDs for custom paths.

<a id="accompanying-files"></a>
## Accompanying Files

`../test_data/dogs.jpg` is the supplied image fixture. `../test_data/mobile_sam_binary_mask.png` is a preserved source reference mask. The default box is `[185,120,380,445]` in resized image coordinates.

<a id="local-paths"></a>
## Local Paths

The runtime resolves the exact selected pair below `samples/vision/mobile_sam/model/`. Custom paths require their corresponding exact encoder and decoder asset IDs.

<a id="formats-checksums"></a>
## Formats and Checksums

The x5 rows use `bin`; S rows use `hbm`. The historical published rows record `sha256: null (unknown)`; the dated converted profile below records verified SHA-256 values.

## Converted S100 profile (2026-11-03)

`tinyvit-512-int16-20261103-o1` · S100 / nash-e · OE 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5) · O1.

Select this independent build explicitly. Historical defaults and measurements are retained. Actual HBM outputs are float32.

| Role | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| encoder | `s:mobile_sam:nash-e/mobile_sam_image_encoder_norm_512x512_nashe_int16_20261103_o1.hbm` | `19f6f39b85c276bbd5f608bbfa3829a03a9507fe0b70a27963c01a03aa9bb8d6` | 49086032 |
| decoder | `s:mobile_sam:nash-e/mobile_sam_decoder_512_nashe_int16_20261103_o1.hbm` | `ea123331de46a21f92a022ba6c6730305ba75ba744efa405f33d08120397e6ad` | 14175872 |

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100 --variant tinyvit-512-int16-20261103-o1
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100 --variant tinyvit-512-int16-20261103-o1 --test-img samples/vision/mobile_sam/test_data/dogs.jpg
```

Calibration uses 198 independent COCO training photographs. Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are `not-run`. Source, checkpoint, ONNX, effective configuration, native recipe and complete OE identities are recorded in [converted.json](../conversion/converted.json).

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100p | `tinyvit-512-int16-20261103-o1` | encoder, decoder | 198 |
| s600 | `tinyvit-512-int16-20261103-o1` | encoder, decoder | 198 |

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100p --variant tinyvit-512-int16-20261103-o1
python3 samples/vision/mobile_sam/runtime/python/main.py --target s100p --variant tinyvit-512-int16-20261103-o1 --encoder-model-path /path/to/mobile_sam_image_encoder_norm_512x512_nashm_int16_20261103_o1.hbm --encoder-asset-id s:mobile_sam:nash-m/mobile_sam_image_encoder_norm_512x512_nashm_int16_20261103_o1.hbm --decoder-model-path /path/to/mobile_sam_decoder_512_nashm_int16_20261103_o1.hbm --decoder-asset-id s:mobile_sam:nash-m/mobile_sam_decoder_512_nashm_int16_20261103_o1.hbm --test-img samples/vision/mobile_sam/test_data/dogs.jpg
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
