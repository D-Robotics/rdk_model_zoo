English | [简体中文](./README_cn.md)

# DINOv2 Python Runtime

Board-side inference demo for the DINOv2 ViT-S/14 int16 `.hbm` model, built
on `hbm_runtime`.

## Input Contract

- Input name: `input`
- Input format: float32 NCHW RGB, fully preprocessed on the board CPU
  (square resize to 224, `/255`, ImageNet mean/std normalization).
- The `.hbm` takes the final float tensor directly; no image preprocessing is
  compiled into the graph.

## Outputs

- `cls_feat`: `(1, 384)` global image embedding.
- `patch_feat`: `(1, 256, 384)` dense per-patch features.

## Usage

```bash
bash run.sh                 # default: cls_feat output, auto-detected model
bash run.sh patch_feat      # inspect the dense output instead
```

Equivalent direct invocation:

```bash
python3 main.py \
  --model-path ../../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
  --test-img ../../test_data/dog.jpg \
  --second-img ../../test_data/bus.jpg \
  --output cls_feat
```

The model march is auto-detected from the on-board SoC when `--model-path`
is not given.

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--model-path` | auto (SoC-detected) | Path to the `.hbm` model. |
| `--test-img` | `../../test_data/dog.jpg` | First test image. |
| `--second-img` | `../../test_data/bus.jpg` | Second image for the similarity demo. |
| `--image-size` | 224 | Square input resolution. |
| `--output` | `cls_feat` | Output to inspect: `cls_feat` or `patch_feat`. |
| `--priority` | 0 | Runtime priority (0-255). |
| `--bpu-cores` | 0 | BPU core indexes. |

## License

See [../../../../LICENSE](../../../../LICENSE).
