English | [简体中文](./README_cn.md)

# MobileSAM Python Runtime

This directory provides the Python demo for MobileSAM full-mask inference on RDK X5. Both the TinyViT image encoder and the box-prompt mask decoder are quantized `.bin` models and run through `hbm_runtime`.

## Workflow

1. `pre_process`: resize the input image to `512x512`, apply MobileSAM mean/std normalization, and prepare `normalized_images` as NCHW float32 featuremap.
2. `forward`: run `mobile_sam_image_encoder_norm_512x512_allint16.bin`, then feed `image_embeddings` and the box prompt to `mobile_sam_decoder_512_box_default.bin`.
3. `post_process`: select the mask with the highest predicted IoU, upsample low-resolution logits to `512x512`, and threshold at `0`.
4. Save both the mask overlay and binary mask image.

## Requirements

- RDK X5 board with `hbm_runtime`.
- Python packages: `numpy`, `opencv-python`.

## Run

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
```

`run.sh` checks the required `.bin` files and calls `../../model/download_model.sh` when a model is missing. Direct `python3 main.py` usage is also supported:

```bash
python3 main.py
```

Optional arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--encoder-model-path` | `../../model/mobile_sam_image_encoder_norm_512x512_allint16.bin` | Quantized encoder model path. |
| `--decoder-model-path` | `../../model/mobile_sam_decoder_512_box_default.bin` | Quantized decoder model path. |
| `--test-img` | `../../test_data/dogs.jpg` | Input image path. |
| `--box` | `185,120,380,445` | Box prompt in resized `512x512` coordinates. |
| `--img-save-path` | `../../test_data/mobile_sam_full_mask_result.jpg` | Overlay result output path. |
| `--mask-save-path` | `../../test_data/mobile_sam_binary_mask.png` | Binary mask output path. |
| `--priority` | `0` | Optional runtime scheduling priority. |

RDK X5 has one BPU core, so this demo does not expose a BPU core selection argument.

Example:

```bash
python3 main.py \
  --encoder-model-path ../../model/mobile_sam_image_encoder_norm_512x512_allint16.bin \
  --decoder-model-path ../../model/mobile_sam_decoder_512_box_default.bin \
  --test-img ../../test_data/dogs.jpg \
  --box 185,120,380,445 \
  --img-save-path ../../test_data/mobile_sam_full_mask_result.jpg \
  --mask-save-path ../../test_data/mobile_sam_binary_mask.png
```

## Output

- Overlay result: `../../test_data/mobile_sam_full_mask_result.jpg`
- Binary mask: `../../test_data/mobile_sam_binary_mask.png`
