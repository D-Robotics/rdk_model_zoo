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

<a id="preparation"></a>
## Preparation

From the repository root, run the explicit target command using the board Python environment. It downloads both manifest assets into `samples/vision/efficient_sam/model/` (X5) or its `nash-e/`, `nash-m/`, `nash-p/` subdirectory (S) and reports observed digests. It requires network access and does not run during inference.

```bash
python3 samples/vision/efficient_sam/model/download.py --target s100
# expect: two files under samples/vision/efficient_sam/model/nash-e/
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
