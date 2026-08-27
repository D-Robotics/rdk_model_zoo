English | [简体中文](./README_cn.md)

# Runtime (MobileSAM)

Board-side full-mask inference for MobileSAM using `hbm_runtime`.

## Directory Structure

```text
.
├── main.py
├── mobile_sam.py
├── run.sh
├── README.md
└── README_cn.md
```

## Dependencies

```bash
pip install numpy opencv-python
```

`hbm_runtime` is provided by the board's RDK-S runtime, not pip.

## Quick Start

```bash
bash run.sh
```

`run.sh` auto-detects the board, downloads the matching `.hbm` pair if missing,
then runs `python3 main.py`.

## Manual Run

```bash
python3 main.py --box 100,50,400,460 --bpu-cores 0 1
```

`main.py` runs NCHW RGB preprocessing, encoder and decoder inference, and mask
upsampling, then saves the overlay and binary mask.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--encoder-model-path` | auto | Override the encoder `.hbm` path. |
| `--decoder-model-path` | auto | Override the decoder `.hbm` path. |
| `--test-img` | `test_data/dogs.jpg` | Input image. |
| `--img-save-path` | `test_data/mobile_sam_full_mask_result.jpg` | Overlay output. |
| `--mask-save-path` | `test_data/mobile_sam_binary_mask_result.png` | Binary mask output (reference `mobile_sam_binary_mask.png` is preserved). |
| `--box` | `185,120,380,445` | Box prompt `x1,y1,x2,y2` in 512×512 coords. |
| `--priority` | `0` | Model scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Output

- `mobile_sam_full_mask_result.jpg` — overlay of the mask + contour on the resized image.
- `mobile_sam_binary_mask_result.png` — the binary mask (the committed `mobile_sam_binary_mask.png` is the reference, left untouched).

## Files

- `main.py` — entry point; resolves the board, loads both `.hbm` models, runs inference.
- `mobile_sam.py` — `MobileSAMSegment` pipeline (preprocess → encoder → decoder → postprocess) using `hbm_runtime.HB_HBMRuntime`.
- `run.sh` — launcher with auto board detection + download-if-missing.

## API

`MobileSAMSegment` exposes the standard interfaces:

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

This directory is licensed under the [Apache 2.0 License](../../../../../LICENSE).