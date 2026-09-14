[English](./README.md) | [简体中文](./README_cn.md)

# Python Runtime

## Requirements

- RDK X5 board environment
- BSP-provided `hbm_runtime` matching the installed `libdnn`
- Python 3, NumPy, and OpenCV

Do not install the unrelated PyPI package named `hbm_runtime`.

## Run

Default model, image, and output directory:

```bash
bash run.sh
```

Specify all paths explicitly:

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

The default model is `yolo26n_depth_bayese_768x768_nv12.bin`, the default input is `test_data/bus.jpg`, and the default output directory is `test_data/python_result`.

## Outputs

- `log_depth.npy`: raw calibrated log-depth.
- `depth_native.npy`: relative depth restored to source resolution.
- `depth.png`: colorized depth visualization.
- `overlay.png`: source image and depth visualization overlay.
- `report.json`: model, input, geometry, output-shape, and latency metadata.

## Code Interface

`yolo26_depth.py` provides the reusable `Yolo26Depth` class. Model-specific letterbox restoration remains local, while NV12 conversion reuses `utils/py_utils/preprocess.py`.

Follow the [source-reference documentation guide](../../../../../docs/source_reference/README.md) to generate API documentation.
