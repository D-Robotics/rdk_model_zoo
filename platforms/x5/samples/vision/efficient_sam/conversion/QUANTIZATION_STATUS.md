# EfficientSAM-Tiny Quantization Status

This sample publishes two RDK X5 `.bin` models for the board-side EfficientSAM-Tiny demo:

- `efficient_sam_vitt_encoder_512x512_default_none.bin`
- `efficient_sam_vitt_decoder_fixedprompt_512_default.bin`

The encoder consumes a `512x512` RGB featuremap tensor and emits `image_embeddings`. The fixed-prompt decoder consumes `image_embeddings` and emits `low_res_masks` plus `iou_predictions`. The point prompts are baked into the decoder ONNX during export, so the runtime sample uses a static dual-model pipeline.

## Published Flow

```text
input image -> encoder.bin -> image_embeddings -> fixedprompt_decoder.bin -> low_res_masks/iou_predictions -> full mask
```

## Notes

- Generated ONNX files, `*_quant_info.json`, calibration tensors, and `.bin` outputs are not committed with the sample.
- Use `model/download_model.sh` to fetch the published `.bin` files.
- Use `conversion/README.md` to regenerate the models from the upstream EfficientSAM checkpoint when needed.