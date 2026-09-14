English | [简体中文](./README_cn.md)

# Runtime (EfficientSAM)

Board-side full-mask inference for EfficientSAM using `hbm_runtime`.

## Directory Structure

```text
.
├── main.py
├── efficient_sam.py
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
then runs `python3 main.py`. The box prompt is baked into the decoder ONNX, so no
prompt argument is needed.

## Manual Run

```bash
python3 main.py --bpu-cores 0 1
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--encoder-model-path` | auto | Override the encoder `.hbm` path. |
| `--decoder-model-path` | auto | Override the decoder `.hbm` path. |
| `--test-img` | `test_data/dogs.jpg` | Input image. |
| `--img-save-path` | `test_data/efficient_sam_full_mask_result.jpg` | Overlay output. |
| `--mask-save-path` | `test_data/efficient_sam_binary_mask_result.png` | Binary mask output (reference `efficient_sam_binary_mask.png` is preserved). |
| `--priority` | `0` | Model scheduling priority. |
| `--bpu-cores` | `0` | BPU core indexes. |

## Output

- `efficient_sam_full_mask_result.jpg` — mask + contour overlay.
- `efficient_sam_binary_mask_result.png` — binary mask (the committed `efficient_sam_binary_mask.png` is the reference, left untouched).

## Files

- `main.py` — entry; resolves board, loads both `.hbm`, runs inference.
- `efficient_sam.py` — `EfficientSAMSegment` pipeline using `hbm_runtime.HB_HBMRuntime`.
- `run.sh` — launcher with auto board detection + download-if-missing.

## API

`EfficientSAMSegment` exposes the standard interfaces:

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