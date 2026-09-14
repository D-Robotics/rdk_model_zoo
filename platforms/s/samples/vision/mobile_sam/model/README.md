English | [简体中文](./README_cn.md)

# Model (MobileSAM)

This directory downloads and stores the pre-quantized MobileSAM encoder and decoder HBM models.

## Directory Structure

```text
.
├── download_model.sh      # Download pre-compiled models
└── README.md              # Documentation
```

## Download Models

To download the pre-compiled MobileSAM models for RDK S100/S100P/S600, run:

```bash
bash download_model.sh            # auto-detect board
bash download_model.sh nash-e     # explicit march
```

The script auto-detects the board and downloads models under `./nash-e/`, `./nash-m/` or `./nash-p/`.

## Notes

- The RDK S100/S100P/S600 inference model format is `.hbm`.
- MobileSAM uses a dual-model pipeline: an image encoder plus a box-prompt decoder, both compiled per march.
- The model suffix differs by platform: `nashe` for S100, `nashm` for S100P, `nashp` for S600.

## Published Models

| Board | march | Encoder `.hbm` | Decoder `.hbm` |
|---|---|---|---|
| S100 | nash-e | `nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm` | `nash-e/mobile_sam_decoder_512_nashe.hbm` |
| S100P | nash-m | `nash-m/mobile_sam_image_encoder_norm_512x512_nashm.hbm` | `nash-m/mobile_sam_decoder_512_nashm.hbm` |
| S600 | nash-p | `nash-p/mobile_sam_image_encoder_norm_512x512_nashp.hbm` | `nash-p/mobile_sam_decoder_512_nashp.hbm` |

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).