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

<a id="preparation"></a>
## Preparation

From the repository root, run the explicit target command using the board Python environment. It downloads both manifest assets into `samples/vision/mobile_sam/model/` (X5) or its `nash-e/`, `nash-m/`, `nash-p/` subdirectory (S) and reports observed digests. Network access is required for this step; inference never downloads.

```bash
python3 samples/vision/mobile_sam/model/download.py --target s100
# expect: two files under samples/vision/mobile_sam/model/nash-e/
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

The x5 rows use `bin`; S rows use `hbm`. Every active manifest row records `sha256: null (unknown)`. No checksum is invented.
