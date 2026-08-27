English | [简体中文](./README_cn.md)

# Model (EfficientSAM)

This directory downloads and stores the pre-quantized EfficientSAM encoder and decoder HBM models.

## Directory Structure

```text
.
├── download_model.sh      # Download pre-compiled models
└── README.md              # Documentation
```

## Download Models

To download the pre-compiled EfficientSAM models for RDK S100/S100P/S600, run:

```bash
bash download_model.sh            # auto-detect board
bash download_model.sh nash-e     # explicit march
```

The script auto-detects the board and downloads models under `./nash-e/`, `./nash-m/` or `./nash-p/`.

## Notes

- The RDK S100/S100P/S600 inference model format is `.hbm`.
- EfficientSAM uses a dual-model pipeline: an image encoder plus a prompt decoder, both compiled per march.
- The model suffix differs by platform: `nashe` for S100, `nashm` for S100P, `nashp` for S600.

## Published Models

| Board | march | Encoder `.hbm` | Decoder `.hbm` |
|---|---|---|---|
| S100 | nash-e | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` |
| S100P | nash-m | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` |
| S600 | nash-p | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` |

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).