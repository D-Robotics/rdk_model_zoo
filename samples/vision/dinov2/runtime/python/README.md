English | [简体中文](./README_cn.md)

# DINOv2 Python Runtime

Board-side inference demo for the DINOv2 ViT-S/14 int16 `.hbm` model, built
on `hbm_runtime`.

## Environment and Files

Supported boards are RDK S100 (`nash-e`), RDK S100P (`nash-m`), and RDK S600
(`nash-p`). Run on an RDK image that provides `hbm_runtime`, with Python 3,
NumPy, and OpenCV installed.

```text
runtime/python/
├── README.md       # English guide
├── README_cn.md    # 中文说明
├── dinov2.py       # preprocessing and HBM wrapper
├── main.py         # command-line demo and stdout summary
└── run.sh          # model download and demo launcher
```

## Input Contract

- Input name: `input`
- Input format: one contiguous float32 NCHW RGB tensor with the fixed shape
  `1x3x224x224`. The model has a fixed 224 input contract.
- On the board CPU, OpenCV converts BGR to RGB, bicubic-resizes the short side
  to 256 while preserving aspect ratio, center-crops 224 by 224, applies
  `/255` and ImageNet mean/std normalization, then writes contiguous float32
  NCHW.
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

## Stdout

The demo first prints model metadata. It then prints the selected embedding
summary: output name, shape, dtype, mean, std, min, max, and L2 norm. When
the `--second-img` file exists, it also prints cosine similarity between the
two image embeddings.

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--model-path` | auto (SoC-detected) | Path to the `.hbm` model. |
| `--test-img` | `../../test_data/dog.jpg` | First test image. |
| `--second-img` | `../../test_data/bus.jpg` | Second image for the similarity demo. |
| `--output` | `cls_feat` | Output to inspect: `cls_feat` or `patch_feat`. |
| `--priority` | 0 | Runtime priority (0-255). |
| `--bpu-cores` | 0 | BPU core indexes. |

## License

See [../../../../../LICENSE](../../../../../LICENSE).
