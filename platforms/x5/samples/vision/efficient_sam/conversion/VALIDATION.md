# Validation Evidence

Date: 2026-07-06
Environment: OE Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`, `hb_mapper` version `1.24.3`; RDK X5 board with `hbm_runtime`.

## Encoder Quantization

The accepted encoder model is:

```text
model/efficient_sam_vitt_encoder_512x512_default_none.bin
```

Final encoder-output cosine from `hb_mapper makertbin`:

```text
/image_encoder/neck/neck.3/Mul: 0.968013
```

Status: PASS for the segmentation/embedding gate `>= 0.95`.

## Decoder Quantization

The accepted decoder model is:

```text
model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin
```

The decoder ONNX is exported as a fixed-prompt decoder with positive points `(248,210)` and `(302,315)` baked into the graph. This avoids a board-side PyTorch prompt decoder and keeps the final demo fully on `hbm_runtime`.

Decoder output cosine from `hb_mapper makertbin`:

```text
low_res_masks:    0.965641
iou_predictions:  0.997313
```

Status: PASS for the segmentation output gate `>= 0.95`.

## Board Validation

Verified on RDK X5 with `hbm_runtime`:

```text
encoder.bin -> image_embeddings -> fixedprompt_decoder.bin -> low_res_masks/iou_predictions -> full mask
```

Observed output from `python3 main.py`:

```text
Predicted IoU: 0.5895
mask index:    0
```

Generated artifacts:

```text
test_data/efficient_sam_full_mask_result.jpg
test_data/efficient_sam_binary_mask.png
```

The overlay image was visually checked and contains a complete dog mask, not an encoder heatmap or partial decoder artifact.