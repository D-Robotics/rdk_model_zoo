English | [简体中文](./README_cn.md)

# Depth Anything V2 Python Runtime

This directory provides the Python inference entry for Depth Anything V2 monocular depth estimation on RDK S100/S100P.

## Directory Structure

```text
.
├── main.py
├── depth_anything_v2.py
├── run.sh
├── README.md
└── README_cn.md
```

## Dependencies

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 torch==2.3.1
```

## Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | HBM model path | `../../model/s100/depth_any.hbm` |
| `--test-img` | Test image path | `../../test_data/furseal.jpg` |
| `--img-save-path` | Output depth map path | `result.jpg` |
| `--priority` | hbm_runtime scheduling priority | `0` |
| `--bpu-cores` | BPU core indexes | `0` |

## Quick Start

```bash
cd samples/vision/depth_anything_v2/runtime/python
bash run.sh
```

## Manual Run

```bash
python3 main.py \
  --model-path ../../model/s100/depth_any.hbm \
  --test-img ../../test_data/furseal.jpg \
  --img-save-path result.jpg
```

`main.py` reads the input image, runs NCHW RGB preprocessing, HBM inference, depth-map interpolation and normalization, then saves the colorized depth map.

## API

The `DepthAnythingV2` wrapper exposes the standard interfaces:

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
