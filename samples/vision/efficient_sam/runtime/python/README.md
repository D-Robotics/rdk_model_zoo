English | [简体中文](./README_cn.md)

# EfficientSAM-Tiny Python Runtime

This directory provides the Python demo for EfficientSAM-Tiny full-mask inference on RDK X5. Both the image encoder and fixed-prompt decoder are quantized `.bin` models and run through `hbm_runtime`.

## Workflow

1. `pre_process`: resize the input image to `512x512` and prepare `batched_images` as NCHW float32 featuremap.
2. `forward`: run `efficient_sam_vitt_encoder_512x512_default_none.bin`, then feed `image_embeddings` to `efficient_sam_vitt_decoder_fixedprompt_512_default.bin`.
3. `post_process`: select the mask with the highest predicted IoU, upsample low-resolution logits to `512x512`, and threshold at `0`.
4. Save both the mask overlay and binary mask image.

## Requirements

- RDK X5 board with `hbm_runtime`.
- Python packages: `numpy`, `opencv-python`.

## Run

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
```

`run.sh` checks the required `.bin` files and calls `../../model/download_model.sh` when a model is missing. Direct `python3 main.py` usage is also supported:

```bash
python3 main.py
```

Optional arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--encoder-model-path` | `../../model/efficient_sam_vitt_encoder_512x512_default_none.bin` | Quantized encoder model path. |
| `--decoder-model-path` | `../../model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | Quantized decoder model path. |
| `--test-img` | `../../test_data/dogs.jpg` | Input image path. |
| `--img-save-path` | `../../test_data/efficient_sam_full_mask_result.jpg` | Overlay result output path. |
| `--mask-save-path` | `../../test_data/efficient_sam_binary_mask.png` | Binary mask output path. |
| `--priority` | `0` | Optional runtime scheduling priority. |

RDK X5 has one BPU core, so this demo does not expose a BPU core selection argument.

Example:

```bash
python3 main.py \
  --encoder-model-path ../../model/efficient_sam_vitt_encoder_512x512_default_none.bin \
  --decoder-model-path ../../model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin \
  --test-img ../../test_data/dogs.jpg \
  --img-save-path ../../test_data/efficient_sam_full_mask_result.jpg \
  --mask-save-path ../../test_data/efficient_sam_binary_mask.png
```

## Output

- Overlay result: `../../test_data/efficient_sam_full_mask_result.jpg`
- Binary mask: `../../test_data/efficient_sam_binary_mask.png`
